from src.aminer import Aminer
from torch_geometric.data import DataLoader
from src.layers import StackedGCNDBLP
import os
import os.path as osp
import torch.nn as nn
import torch
import pickle
from tqdm import tqdm
from src.utils import dict2table, confusion, str2bool
import numpy as np
import random

# check if geometric batch largest value is index or value

"""
    current default value is top 10
"""

class RankingTrainer():

    def __init__(self, batch_size=64):
        if osp.exists('dblp_hete_shuffle_idx.pkl'):
            with open('dblp_hete_shuffle_idx.pkl', 'rb') as f:
                shuffle_idx = pickle.load(f)
        else:
            shuffle_idx = [idx for idx in range(len(dataset))]
            split_pos = int(len(dataset)*0.7)
            train_idx = shuffle_idx[:split_pos]
            random.shuffle(train_idx)
            shuffle_idx[:split_pos] = train_idx
            print(shuffle_idx[split_pos: split_pos+10])
            with open('dblp_hete_shuffle_idx.pkl', 'wb') as f:
                pickle.dump(shuffle_idx, f)

        dataset = Aminer()
        dataset = dataset[shuffle_idx]

        split_pos = int(len(dataset)*0.7)
        train_idx = shuffle_idx[:split_pos]
        valid_idx_ = shuffle_idx[split_pos:]
        test_pos = int(len(valid_idx_)*0.333)
        test_idx = valid_idx_[:test_pos]
        valid_idx = valid_idx_[test_pos:]

        self.train_dataset = dataset[train_idx]
        self.test_dataset = dataset[test_idx]
        self.valid_dataset = dataset[valid_idx]

    def calculate_loss(self, outputs, batch, batch_size):
        # calculate triple ranking loss 
        total_neg = 0
        total_pos = 0
        log_sigmoid = nn.LogSigmoid()
        # iterate through all datasets
        batch_size = torch.max(batch.batch)+1
        for batch_idx in range(batch_size):
            paper_idx = (batch.batch == batch_idx)
            data = outputs[paper_idx]

            # author node in this data
            author_node_idx = (batch.x[:, -1] == 0) & paper_idx
            candidate_idx = author_node_idx & (batch.x[:, 1 ] == 1)

            candidate_embed = outputs[ candidate_idx ]
            latent = candidate_embed.sum(0)
            target_embed = outputs[ (batch.y==1) & paper_idx]

            pos = log_sigmoid(torch.mm(latent.unsqueeze(-1).T, target_embed.T).flatten())
            total_pos += pos.sum()
            # not label not known users and node type = user
            negative_node_idx = (batch.y == 0) & (batch.x[:, 1 ] == 0) & (batch.x[:, 2 ] == 0) & paper_idx

            negative_embed = outputs[ negative_node_idx ]
            if len(negative_embed) == 0:
                # print('no negative found!')
                continue

            shuffle_idx = list(range(len(negative_embed)))
            random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[:len(target_embed)]

            negative_embed = negative_embed[shuffle_idx]

            # torch.dot(target_embed[0],  latent)
            neg = log_sigmoid(-torch.mm(latent.unsqueeze(-1).T, negative_embed.T).flatten())
            total_neg += neg.sum()

        loss = (total_pos+total_neg) / (batch_size*2)
        return -loss
    
    def inference(self, outputs, x, batch, top_k=10, user_size=874608):
        B = torch.max(batch)+1
        y_pred = torch.FloatTensor(B, user_size)
        y_pred.zero_()
        # y_target = torch.FloatTensor(B, user_size)
        # y_target.zero_()
        for batch_idx in range(B):
            paper_idx = (batch == batch_idx)

            data = outputs[paper_idx]
            author_node_idx = (x[:, -1] == 0) & paper_idx
            known_author_idx = author_node_idx & (x[:, 1 ] == 1)

            candidate_embed = outputs[ known_author_idx ]
            target_embed = outputs[ author_node_idx & (x[:, 1 ] == 0) ]
            x_id = x[ author_node_idx & (x[:, 1 ] == 0), 0 ]
            latent = candidate_embed.sum(0)
            if len(target_embed) == 0:
                continue
            rank = torch.sigmoid(torch.mm(target_embed, latent.unsqueeze(-1))).flatten()

            best_idx = torch.argsort(rank, descending=True)
            # true_id = x[ (y==1) & paper_idx, 0 ]
            # y_target[batch_idx, true_id ] = 1
            # print(best_idx[:top_k], x_id[best_idx[:top_k]], true_id )
            y_pred[ batch_idx, x_id[best_idx[:top_k]] ] = 1

        return y_pred

    def evaluate(self, dataloader, model):
        model.eval()
        recalls = []
        precisions = []
        B = self.batch_size
        user_size = 874608
        val_loss = 0
        cnt = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, dynamic_ncols=True):
                x, edge_index = batch.x, batch.edge_index
                x = x.cuda()
                edge_index = edge_index.cuda()
                outputs = model(edge_index, x)
                val_loss += self.calculate_loss(outputs, batch, B)

                outputs = outputs.cpu()       
                x = x.cpu()
                y_pred = self.inference(outputs, x, batch.batch, user_size=user_size)

                B = torch.max(batch.batch)+1
                y_target = torch.FloatTensor(B, user_size)
                y_target.zero_()
                for batch_idx in range(B):
                    paper_idx = (batch.batch == batch_idx)
                    user_id = batch.x[ (batch.y==1) & paper_idx, 0 ]
                    y_target[batch_idx, user_id] = 1

                # print(y_pred.sum(), y_target.sum())

                TP, FP, TN, FN = confusion(y_pred, y_target)
                recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
                precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
                # print(precision, recall)
                precisions.append(precision)
                recalls.append(recall)
                cnt += 1

        avg_recalls = np.mean(recalls)
        avg_precisions = np.mean(precisions)
        f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)
        model.train()
        return f1, avg_recalls, avg_precisions, val_loss/cnt


    def train(self, batch_size=64, epochs=100):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
        self.batch_size = batch_size
        # test_loader = DataLoader(self.test_dataset,
        #                          batch_size=args.batch_size,
        #                          shuffle=False)

        model = StackedGCNDBLP(
                   author_size=874608,#len(author2id),
                   paper_size=3605603,#len(paper2id),
                   conf_size=12770, output_channels=8)
        model = model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0005, weight_decay=5e-4)

        with tqdm(total=len(train_loader)*epochs, dynamic_ncols=True) as pbar:
            for epoch in range(epochs):

                for batch in train_loader:
                    optimizer.zero_grad()
                    x, edge_index = batch.x, batch.edge_index
                    x = x.cuda()
                    edge_index = edge_index.cuda()
                    outputs = model(edge_index, x)
                    loss = self.calculate_loss(outputs, batch, batch_size)
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(loss.item(), epoch))
                if epoch % 5 == 0:
                    print(self.evaluate(valid_loader, model))

if __name__ == "__main__":
    trainer = RankingTrainer()
    trainer.train()
