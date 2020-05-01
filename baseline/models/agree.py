'''
Created on Nov 10, 2017
Updated on Apr 18, 2020
@author: Lianhai Miao, Zhi Rui Tam
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelfAttention(nn.Module):

    def __init__(self, embed_dim, hidden):
        super(SelfAttention, self).__init__()
        self.temperature = hidden ** 0.5
        self.query = nn.Linear(embed_dim, hidden, bias=False)
        self.key = nn.Linear(embed_dim, hidden, bias=False)
        self.value = nn.Linear(embed_dim, hidden, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # 1 x C x D
        q = self.query(query)
        # 1 x U x D
        k = self.key(key)
        v = self.value(value)

        # 1 x C x U
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)

        return attn, output


class AGREE(nn.Module):
    def __init__(self, num_users, num_groups, embedding_dim, drop_ratio, w_group=False):
        super(AGREE, self).__init__()
        self.num_users = num_users
        self.embeddings = nn.Embedding(num_users, embedding_dim)
        self.norm = nn.BatchNorm1d(3*embedding_dim)
        self.w_group = w_group
        if w_group:
            self.group_embed = nn.Embedding(num_groups, embedding_dim)
        # self.attention_ = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.attention = SelfAttention(embedding_dim, embedding_dim)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)

    def forward_group_embed(self, user_latent, group, target_user, candidate_len=-1):
        
        if candidate_len < 0:
            candidate_len = user_latent.shape[-2]

        assert group.shape[-1] == 1
        assert target_user.shape[-1] == 1


        target_latent = self.embeddings(target_user)
        
        
        attention_embeds = torch.cat((user_latent, target_latent.repeat(1, candidate_len, 1) ), dim=-1)
        # print(attention_embeds.shape)

        # at_wt = self.attention_(attention_embeds)
        # user_embeds_with_attention = torch.sum(at_wt*user_latent, dim=1)
        # print(user_embeds_with_attention.shape)
        _, user_embeds_with_attention = self.attention( target_latent.repeat(1, candidate_len, 1), user_latent, user_latent )
        user_embeds_with_attention = torch.sum(user_embeds_with_attention, dim=1)
        # print(user_embeds_with_attention.shape)
        g_embeds = user_embeds_with_attention# + group_latent
        if self.w_group:
            group_latent = self.group_embed(group).squeeze(1)
            g_embeds += group_latent
        target_latent = target_latent.squeeze(1)

        element_embeds = g_embeds* target_latent  # Element-wise product
        new_embeds = torch.cat((element_embeds, g_embeds, target_latent.squeeze(1)), dim=1)
        return new_embeds

    def forward(self, input_users, group, target_user):
        '''
            Input users: list of known users
            Group : group index 
            Target user: user you wish to decide whether to invite or not
        '''
        if input_users.shape[-1] > 1:
            candidate_len = input_users.shape[-1]
            user_latent = self.embeddings(input_users)
            new_embeds = self.forward_group_embed(user_latent, group, target_user, candidate_len=candidate_len)
        else:
            user_latent = self.embeddings(input_users).squeeze(1)
            target_latent = self.embeddings(target_user).squeeze(1)
            element_embeds = user_latent * target_latent
            new_embeds = torch.cat((element_embeds, user_latent, target_latent), dim=1)

        # new_embeds = self.norm(new_embeds)
        y = torch.sigmoid(self.predictlayer(new_embeds))


        return y

    def predict_rank(self, input_users, group, candidates, top_k =5):
        B = input_users.shape[0]
        y_pred = torch.zeros((B, self.num_users))
        y_pred.zero_()
        assert group.shape[-1] == 1
        candidate_len = input_users.shape[-1]

        for batch_idx in range(B):
            candidate_index = (candidates[batch_idx] == 1 ).nonzero()
            candidate_size = len(candidate_index)

            user_latent = self.embeddings(input_users[[batch_idx]])
            user_latent = user_latent.repeat(candidate_size, 1, 1) # C x K x D

            new_embeds = self.forward_group_embed(user_latent, group[[batch_idx]], 
                candidate_index, candidate_len)
            
            # target_latent = self.embeddings(candidate_index) # C x D


            # attention_embeds = torch.cat((user_latent, target_latent.repeat(1, candidate_len, 1) ), dim=-1)
            # # print(attention_embeds.shape)

            # at_wt = self.attention(attention_embeds) # C x K x 1

            # user_embeds_with_attention = torch.sum(at_wt*user_latent, dim=1)
            # g_embeds = user_embeds_with_attention
            # if self.w_group:
            #     group_latent = self.group_embed(group[batch_idx]).squeeze(1)
            #     group_latent = group_latent.repeat(candidate_size, 1) # C x D
            #     g_embeds += group_latent

            # target_latent = target_latent.squeeze(1)
            # element_embeds = g_embeds* target_latent  # Element-wise product
            # new_embeds = torch.cat((element_embeds, g_embeds, target_latent), dim=1)
            # new_embeds = self.norm(new_embeds)

            rank = torch.sigmoid(self.predictlayer(new_embeds)).flatten()

            best_idx = torch.argsort(rank, descending=True)
            y_pred[ batch_idx, candidate_index[best_idx[:top_k]] ] = 1

        return y_pred




class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.temperature = embedding_dim ** 0.5
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # print(x.shape)
        out = self.linear(x/ self.temperature)
        weight = F.softmax(out, dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":
    B = 32
    known_users = torch.randint(0, 10, (B, 10)).long()
    groups = torch.randint(0, 10, (B, 1)).long()
    query_users = torch.randint(0, 10, (B, 3)).long()
    query_mask = torch.zeros(B, 10)
    for batch in range(B):
        query_mask[batch, query_users[batch]] = 1.0

    model = AGREE(10, 10, 16, 0.1)

    model.predict_rank(known_users, groups, query_mask)

    # query_users = torch.randn(8, 10, 64)
    # known_users = torch.randn(8, 2, 64)
    # selfattn = SelfAttention(64, 64)
    # _, output = selfattn(known_users, query_users, query_users)
    # print(output.shape)