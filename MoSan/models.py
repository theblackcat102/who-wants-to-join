import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(positive_predictions, negative_predictions, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.
    Parameters
    ----------
    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    References
    ----------
    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """

    loss = (1.0 - torch.sigmoid(positive_predictions -
                                negative_predictions))

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


class AttentionModule(nn.Module):
    def __init__(self, user_dim=128, context_dim=128, hidden_dim=128 ,K=9):
        super(AttentionModule, self).__init__()
        self.context_weight = nn.Linear(context_dim, hidden_dim)
        self.user_weight = nn.Linear(user_dim, hidden_dim, bias=False)

        self.weight_vector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        self.drop = nn.Dropout(0.1)
        self.K = K

    def forward(self, c, u, attn_mask=None):
        '''
            attn_mask : B x group_size
        '''
        B, N = c.shape
        _, G, H = u.shape
        c = c.reshape(B, 1, N)
        c = c.repeat(1, G, 1) # B x G X N
        c = c.view(-1, N)
        _u = u.view(-1, H)

        proj = F.relu(  self.context_weight(c) + self.user_weight(_u) )
        attn = self.weight_vector(self.drop(proj))
        attn = attn.view(-1, self.K)


        attention_score = F.softmax(attn, dim=1)
        if attn_mask is not None:
            _B, _G = attn_mask.shape 
            attn_mask = attn_mask[:, :_G-1]
            attn_mask = attn_mask.reshape(_B, -1)
            attn_mask = attn_mask.repeat(_G, 1) # B*(_G) x G 
            attention_score = torch.mul(attention_score, attn_mask)

        attention_score = attention_score.unsqueeze(-1)
        weighted_sum = attention_score * u
        g_latent = weighted_sum.sum(1)

        return g_latent


class MoSAN(nn.Module):
    def __init__(self, 
        group_size=398,
        venue_size=2396,
        user_size=46895,
        user_dim=128, context_dim=128, hidden_dim=128, group=10):
        super(MoSAN, self).__init__()
        # offset for padding
        self.user_embed = nn.Embedding(user_size+1, user_dim, padding_idx=0)
        self.venue_embed = nn.Embedding(venue_size, user_dim)
        self.attn = AttentionModule(user_dim=user_dim, 
            context_dim=context_dim, hidden_dim=hidden_dim, K=group-1)
        
    
    def forward(self, context, group_users, pos_venue, neg_venue, attn_mask=None):
        '''
            context = (B x N)
            group_users = (B x group_size-1 x N)
            attn_mask = (B x group_size )
        '''
        _B, _N = context.shape
        # print(context.shape, group_users.shape, pos_venue.shape)
        context = context.view(_B*_N)
        group_users = group_users.view(_B*_N, -1)
        context = self.user_embed(context)

        pos_venue = self.venue_embed(pos_venue).squeeze(1)
        neg_venue = self.venue_embed(neg_venue).squeeze(1)

        group_users = self.user_embed(group_users)


        g_latent = self.attn(context, group_users, attn_mask)


        g_latent = g_latent.view(_B, _N, -1)

        if attn_mask is not None:
            g_latent = g_latent.transpose(1, 2)
            g_latent = g_latent * attn_mask.unsqueeze(1)
            g_latent = g_latent.transpose(1, 2)
        g_latent = g_latent.sum(1)
        pos = torch.bmm(pos_venue.view(_B, 1, -1), g_latent.view(_B, -1, 1)).squeeze()
        neg = torch.bmm(neg_venue.view(_B, 1, -1), g_latent.view(_B, -1, 1)).squeeze()
        return pos, neg


if __name__ == "__main__":
    a = MoSAN(group=20)
    factor = torch.randint(0, 4684, (32, 20)).long()
    context = torch.randint(0, 4684, (32, 20, 19)).long()
    mask = torch.ones((32, 20))
    mask[:, 10:] = torch.zeros((32, 10))
    pos_v, neg_v = torch.randint(0, 2395, (32, 1)).long(), torch.randint(0, 2395, (32, 1)).long()
    pairwise_match = a(factor, context, pos_v, neg_v, attn_mask=mask)
    nlloss = bpr_loss(*pairwise_match)
    print(nlloss)