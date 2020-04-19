'''
Created on Nov 10, 2017
Updated on Apr 18, 2020
@author: Lianhai Miao, Zhi Rui Tam
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AGREE(nn.Module):
    def __init__(self, num_users, num_groups, embedding_dim, drop_ratio):
        super(AGREE, self).__init__()
        self.num_users = num_users
        self.embeddings = nn.Embedding(num_users, embedding_dim)
        self.group_embed = nn.Embedding(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)

    def forward(self, input_users, group, target_user):
        '''
            Input users: list of known users
            Group : group index 
            Target user: user you wish to decide whether to invite or not
        '''
        if input_users.shape[-1] > 1:
            user_latent = self.embeddings(input_users)
            candidate_len = input_users.shape[-1]

            assert group.shape[-1] == 1
            assert target_user.shape[-1] == 1

            group_latent = self.group_embed(group).squeeze(1)
            target_latent = self.embeddings(target_user)
            
            attention_embeds = torch.cat((user_latent, target_latent.repeat(1, candidate_len, 1) ), dim=-1)
            # print(attention_embeds.shape)

            at_wt = self.attention(attention_embeds)
            user_embeds_with_attention = torch.sum(at_wt*user_latent, dim=1)

            g_embeds = user_embeds_with_attention + group_latent
            target_latent = target_latent.squeeze(1)

            element_embeds = g_embeds* target_latent  # Element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds, target_latent.squeeze(1)), dim=1)

        else:
            user_latent = self.embeddings(input_users).squeeze(1)
            target_latent = self.embeddings(target_user).squeeze(1)
            element_embeds = user_latent * target_latent
            new_embeds = torch.cat((element_embeds, user_latent, target_latent), dim=1)

        y = torch.sigmoid(self.predictlayer(new_embeds))


        return y

    def predict_rank(self, input_users, group, candidates, top_k =5):
        B = input_users.shape[0]
        y_pred = torch.zeros((B, self.num_users))
        y_pred.zero_()
        assert group.shape[-1] == 1
        candidate_len = input_users.shape[-1]

        for batch_idx in range(B):
            user_latent = self.embeddings(input_users[[batch_idx]])



            group_latent = self.group_embed(group[batch_idx]).squeeze(1)

            candidate_index = (candidates[batch_idx] == 1 ).nonzero()
            candidate_size = len(candidate_index)
            
            target_latent = self.embeddings(candidate_index) # C x D

 
            user_latent = user_latent.repeat(candidate_size, 1, 1) # C x K x D
            group_latent = group_latent.repeat(candidate_size, 1) # C x D

            attention_embeds = torch.cat((user_latent, target_latent.repeat(1, candidate_len, 1) ), dim=-1)
            # print(attention_embeds.shape)

            at_wt = self.attention(attention_embeds) # C x K x 1

            user_embeds_with_attention = torch.sum(at_wt*user_latent, dim=1)
            g_embeds = user_embeds_with_attention + group_latent
            target_latent = target_latent.squeeze(1)
            element_embeds = g_embeds* target_latent  # Element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds, target_latent), dim=1)

            rank = torch.sigmoid(self.predictlayer(new_embeds)).flatten()

            best_idx = torch.argsort(rank, descending=True)
            y_pred[ batch_idx, candidate_index[best_idx[:top_k]] ] = 1

        return y_pred




class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # print(x.shape)
        out = self.linear(x)
        weight = F.softmax(out, dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
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