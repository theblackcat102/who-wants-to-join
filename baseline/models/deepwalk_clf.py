import torch
from torch import nn
import random

class DeepwalkClf(torch.nn.Module):

    def __init__(self, embeddings, user_size, mode='classifier'):
        super(DeepwalkClf, self).__init__()
        self.user_size = user_size
        embed_size, embed_dim = embeddings.weight.shape
        self.embeddings = embeddings
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim)
        )

        if mode == 'classifier':
            self.output = nn.Linear(embed_dim, user_size)
            print('initialize output proj using embeddings')
            self.output.weight.data.copy_(self.embeddings.weight.data)
        self.log_sigmoid = nn.LogSigmoid()
        self.embed_dim = embed_dim

    def forward_rank(self, candidates, masked_target, target, margin=1, neg_sample=5):
        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)

        latent = self.fc(pooled)

        batch_size = candidates.shape[0]
        total_neg = 0
        total_pos = 0

        for batch_idx in range(batch_size):
            user_latent = latent[batch_idx]
            masked_target[batch_idx] = masked_target[batch_idx] - target[batch_idx]
            negative_index = (masked_target[batch_idx] == 1 ).nonzero()
            neg_latent = self.embeddings(negative_index.squeeze(-1))

            shuffle_idx = list(range(len(neg_latent)))
            random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[:neg_sample]

            neg_latent = neg_latent[shuffle_idx]

            loss = (margin-torch.mm(user_latent.unsqueeze(-1).T, neg_latent.T).flatten()).clamp(min=0)
            neg = self.log_sigmoid(loss)
            total_neg += neg.sum()

            pos_latent = self.embeddings((target[batch_idx] == 1).nonzero().squeeze(-1))

            # print(user_latent.unsqueeze(-1).T.shape, pos_latent.T.shape)
            pos = self.log_sigmoid(torch.mm(user_latent.unsqueeze(-1).T, pos_latent.T).flatten())
            total_pos += pos.sum()

        loss = (total_pos+total_neg) / (batch_size*2)
        # loss = total_neg/(N*2)#log_sigmoid(total_pos + total_neg)/ (batch_size*2)
        return -loss
    
    def forward(self, candidates):
        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)

        latent = self.fc(pooled)
        return self.output(latent)

    def predict(self, candidates, top_k=5):
        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)
        predict = self.fc(pooled)
        values, index = torch.topk(predict, k=top_k)
        return index

    def predict_rank(self, candidates, masked_target, top_k=5):
        batch_size = candidates.shape[0]

        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)
        latent = self.fc(pooled)

        y_pred = torch.zeros((batch_size, self.user_size))
        y_pred.zero_()

        for batch_idx in range(batch_size):
            user_latent = latent[batch_idx]
            candidate_index = (masked_target[batch_idx] == 1 ).nonzero()
            pos_latent = self.embeddings(candidate_index.squeeze(-1))
            
            rank = torch.sigmoid(torch.mm(user_latent.unsqueeze(-1).T, pos_latent.T)).flatten()
            best_idx = torch.argsort(rank, descending=True)
            y_pred[ batch_idx, candidate_index[best_idx[:top_k]] ] = 1

        return y_pred

