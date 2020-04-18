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

        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y



class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
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
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":
    B = 32
    known_users = torch.randint(0, 10, (B, 6)).long()
    groups = torch.randint(0, 10, (B, 1)).long()
    query_users = torch.randint(0, 10, (B, 1)).long()

    model = AGREE(10, 10, 16, 0.1)

    model(known_users, groups, query_users)