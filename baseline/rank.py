from src.aminer import Aminer
from torch_geometric.data import DataLoader
from src.layers import StackedGCNDBLP
from datetime import datetime
import os, sys
import os.path as osp
import torch.nn as nn
import torch
import pickle
from copy import deepcopy
from tqdm import tqdm
from src.utils import dict2table, confusion, str2bool
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import random

# check if geometric batch largest value is index or value

"""
    current default value is top 10
"""
class RankingTrainer():

    def __init__(self, name, model, dataset, shuffle_idx, 
        user_size, batch_size=64, top_k=5, args=None):
        self.name = name
        self.model = model
        self.user_size = user_size
        self.top_k = top_k

        dataset = dataset[shuffle_idx]

        split_pos = int(len(dataset)*0.7)
        train_idx = shuffle_idx[:split_pos]
        valid_idx_ = shuffle_idx[split_pos:]
        # 7: 1: 2 ; train : valid : test
        valid_pos = int(len(valid_idx_)*0.3333)
        valid_idx = valid_idx_[:valid_pos]
        test_idx = valid_idx_[valid_pos:]

        self.train_dataset = dataset[train_idx]
        self.test_dataset = dataset[test_idx]
        self.valid_dataset = dataset[valid_idx]
        self.log_path = osp.join(
            "logs", "baseline",
            self.name+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.writer = SummaryWriter(log_dir=self.log_path)

        if args:
            self.writer.add_text('Text', dict2table(vars(args)), 0)


    def calculate_loss(self, outputs, batch, batch_size, margin=5):
        # calculate triple ranking loss 
        total_neg = 0
        total_pos = 0
        N = 0
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
            shuffle_idx = list(range(len(target_embed)))
            random.shuffle(shuffle_idx)
            target_embed = target_embed[shuffle_idx]

            pos = log_sigmoid(torch.mm(latent.unsqueeze(-1).T, target_embed.T).flatten())

            # pos = torch.mm(latent.unsqueeze(-1).T, target_embed.T).flatten()
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
            loss = (margin-torch.mm(latent.unsqueeze(-1).T, negative_embed.T).flatten()).clamp(min=0)
            neg = log_sigmoid(loss)
            total_neg += neg.sum()

        loss = (total_pos+total_neg) / (batch_size*2)
        # loss = total_neg/(N*2)#log_sigmoid(total_pos + total_neg)/ (batch_size*2)
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

    def evaluate(self, dataloader, model, user_size=874608):
        model.eval()
        recalls = []
        precisions = []
        B = self.batch_size
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, dynamic_ncols=True):
                x, edge_index = batch.x, batch.edge_index
                x = x.cuda()
                edge_index = edge_index.cuda()
                outputs = model(edge_index, x)
                val_loss += self.calculate_loss(outputs, batch, B)

                outputs = outputs.cpu()       
                x = x.cpu()
                y_pred = self.inference(outputs, x, batch.batch, 
                    user_size=user_size, top_k=self.top_k)

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

        avg_recalls = np.mean(recalls)
        avg_precisions = np.mean(precisions)
        f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)
        model.train()
        return f1, avg_recalls, avg_precisions, val_loss/len(dataloader)

    def train(self, batch_size=64, epochs=50):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
        self.batch_size = batch_size
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        best_f1 = 0
        model = self.model
        model = model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0005, weight_decay=5e-4)
        n_iter = 0
        with tqdm(total=len(train_loader)*epochs, dynamic_ncols=True) as pbar:
            for epoch in range(epochs):

                for batch in train_loader:
                    optimizer.zero_grad()
                    x, edge_index = batch.x, batch.edge_index
                    x = x.cuda()
                    edge_index = edge_index.cuda()
                    outputs = model(edge_index, x)
                    loss = self.calculate_loss(outputs, batch, batch_size)
                    self.writer.add_scalar(
                        "Train/loss", loss.item(), n_iter)
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(loss.item(), epoch))
                    n_iter += 1
                if epoch % 5 == 0:
                    f1, recalls, precisions, val_loss = self.evaluate(valid_loader, model, 
                        user_size=self.user_size)
                    self.writer.add_scalar(
                        "Val/loss", val_loss.item(), n_iter)
                    self.writer.add_scalar("Val/F1", f1, n_iter)
                    self.writer.add_scalar("Val/Recalls", recalls, n_iter)
                    self.writer.add_scalar("Val/Precisions", precisions, n_iter)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_checkpoint = {
                            'epoch': epoch+1,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'f1': f1
                        }
                    print(f1)

        f1, recalls, precisions, val_loss = self.evaluate(valid_loader, model, 
            user_size=self.user_size)
        self.writer.add_scalar(
            "Val/loss", val_loss.item(), n_iter)
        self.writer.add_scalar("Val/F1", f1, n_iter)
        self.writer.add_scalar("Val/Recalls", recalls, n_iter)
        self.writer.add_scalar("Val/Precisions", precisions, n_iter)

        if f1 > best_f1:
            best_f1 = f1
            best_checkpoint = {
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'f1': f1
            }
        print(f1)
        model.load_state_dict(best_checkpoint["model"])
        f1, recalls, precisions, loss = self.evaluate(test_loader, model)    
        print(f1, recalls, precisions)
        self.writer.add_scalar("Test/F1", f1, n_iter)
        self.writer.add_scalar("Test/Recalls", recalls, n_iter)
        self.writer.add_scalar("Test/Precisions", precisions, n_iter)
        self.writer.flush()
        return f1, recalls, precisions, loss.data

def evaluate_dblp(parser):
    print("Aminer")
    group = parser.add_argument_group('rank parameters')
    group.add_argument('--author-dim', type=int, default=16)
    group.add_argument('--paper-dim', type=int, default=16)
    group.add_argument('--conf-dim', type=int, default=8)
    group.add_argument('--input-dim', type=int, default=32)
    group.add_argument('--output-dim',type=int, default=8)
    group.add_argument('--dropout', type=float, default=0.1)
    group.add_argument('--layers', nargs='+', type=int, default=[32, 32])
    args = parser.parse_args()
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

    model = StackedGCNDBLP(
                author_size=874608,#len(author2id),
                paper_size=3605603,#len(paper2id),
                conf_size=12770, output_channels=args.output_dim,
                user_dim=args.author_dim,
                paper_dim=args.paper_dim,
                conf_dim=args.conf_dim,
                input_channels=args.input_dim,
                layers=args.layers,
                dropout=args.dropout)
    if args.weights:
        print('load pretrain weights')
        weight = torch.load(args.weights)
        model.load_state_dict(weight)

    dataset = Aminer()
    values = {
            'f1': [],
            'recall': [],
            'precision': [],
            'loss': []
        }
    for i in range(args.repeat_n):
        trainer = RankingTrainer('aminer',model, dataset, shuffle_idx, 
            user_size=874608, top_k=args.top_k, args=args)
        f1, recalls, precisions, loss = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        values['f1'].append(f1)
        values['recall'].append(recalls)
        values['precision'].append(precisions)
        values['loss'].append(float(loss))

    results = {}
    for key, value in values.items():
        results['avg_'+key] = np.mean(value)
        results['std_'+key] = np.std(value)
    results['results'] = values
    results['arguments'] = vars(args)
    with open('cf_rank_'+sys.argv[1]+'_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

def evaluate_meetup(parser):
    print("Meetup")
    from src.layers import StackedGCNMeetup
    from src.meetup import Meetup, locations_id, MEETUP_FOLDER
    group = parser.add_argument_group('arguments')
    group.add_argument('--city', type=str, default='SF',
                        choices=['NY', 'SF'])
    group.add_argument('--min-size', type=int, default=5)
    group.add_argument('--max-size', type=int, default=100)
    group.add_argument('--pred-ratio', type=float, default=0.8)
    group.add_argument('--maxhop', type=int, default=2)
    group.add_argument('--input-dim', type=int, default=8)
    group.add_argument('--dropout', type=float, default=0.1)
    group.add_argument('--layers', nargs='+', type=int, default=[8, 8])
    group.add_argument('--output-dim',type=int, default=8)
    args = parser.parse_args()
    with open(osp.join(MEETUP_FOLDER, 'topic2id.pkl'), 'rb') as f:
        topic2id = pickle.load(f)
    with open(osp.join(MEETUP_FOLDER, 'cat2id.pkl'), 'rb') as f:
        cat2id = pickle.load(f)
    # with open(osp.join(MEETUP_FOLDER, 'group2topic.pkl'), 'rb') as f:
    #     group2topic = pickle.load(f)
    dataset = Meetup(city_id=locations_id[args.city],
                         cutoff=args.maxhop, min_size=args.min_size,
                         max_size=args.max_size)
    category_size = 40
    topic_size = len(topic2id)
    group_size = len(dataset.group2id)

    args = parser.parse_args()
    if osp.exists(args.city+'_shuffle_idx.pkl'):
        with open(args.city+'_shuffle_idx.pkl', 'rb') as f:
            shuffle_idx = pickle.load(f)
    model = StackedGCNMeetup(user_size=len(dataset.user2id),
                                 category_size=category_size,
                                 topic_size=topic_size,
                                 group_size=group_size,
                                 input_channels=args.input_dim,
                                 output_channels=args.output_dim,
                                 layers=args.layers,
                                 dropout=args.dropout)
    if args.weights:
        print('load pretrain weights')
        weight = torch.load(args.weights)
        model.load_state_dict(weight)

    values = {
            'f1': [],
            'recall': [],
            'precision': [],
            'loss': []
        }
    for i in range(args.repeat_n):

        trainer = RankingTrainer('meetup_'+str(args.city)+'_', model, dataset, shuffle_idx, 
            user_size=len(dataset.user2id), top_k=args.top_k, args=args)
        f1, recalls, precisions, loss = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        values['f1'].append(f1)
        values['recall'].append(recalls)
        values['precision'].append(precisions)
        values['loss'].append(float(loss))

    results = {}
    for key, value in values.items():
        results['avg_'+key] = np.mean(value)
        results['std_'+key] = np.std(value)
    results['results'] = values
    results['arguments'] = vars(args)
    with open('cf_rank_'+sys.argv[1]+'_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


def evaluate_amazon(parser):
    print("Amazon")
    from src.layers import StackedGCNAmazon
    from src.amazon import AmazonCommunity
    group = parser.add_argument_group('arguments')
    group.add_argument('--user-dim', type=int, default=16)
    group.add_argument('--cat-dim', type=int, default=8)
    group.add_argument('--min-size', type=int, default=5)
    group.add_argument('--max-size', type=int, default=100)
    group.add_argument('--pred-ratio', type=float, default=0.8)
    group.add_argument('--maxhop', type=int, default=2)
    group.add_argument('--input-dim', type=int, default=32)
    group.add_argument('--dropout', type=float, default=0.1)
    group.add_argument('--layers', nargs='+', type=int, default=[32, 32])
    group.add_argument('--output-dim',type=int, default=8)
    args = parser.parse_args()
    dataset = AmazonCommunity(cutoff=args.maxhop, min_size=args.min_size,
                                  max_size=args.max_size)
    with open('data/amazon/cat2id.pkl', 'rb') as f:
        cat2id = pickle.load(f)

    category_size = len(cat2id)

    args = parser.parse_args()

    with open('amazon_hete_shuffle_idx.pkl', 'rb') as f:
        shuffle_idx = pickle.load(f)
    model = StackedGCNAmazon(len(dataset.user2id),
                                 category_size=category_size,
                                 user_dim=args.user_dim,
                                 category_dim=args.cat_dim,
                                 input_channels=args.input_dim,
                                 layers=args.layers,
                                 dropout=args.dropout)
    if args.weights:
        print('load pretrain weights')
        weight = torch.load(args.weights)
        model.load_state_dict(weight)


    values = {
            'f1': [],
            'recall': [],
            'precision': [],
            'loss': []
        }
    for i in range(args.repeat_n):
        trainer = RankingTrainer('amazon_', model, dataset, shuffle_idx, 
            user_size=len(dataset.user2id), top_k=args.top_k, args=args)
        f1, recalls, precisions, loss = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        values['f1'].append(f1)
        values['recall'].append(recalls)
        values['precision'].append(precisions)
        values['loss'].append(float(loss))

    results = {}
    for key, value in values.items():
        results['avg_'+key] = np.mean(value)
        results['std_'+key] = np.std(value)
    results['results'] = values
    results['arguments'] = vars(args)
    with open('cf_rank_'+sys.argv[1]+'_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)



if __name__ == "__main__":
    import argparse
    dataset_function_map = {
        'aminer': evaluate_dblp,
        'meetup': evaluate_meetup,
        'amazon': evaluate_amazon,
    }
    parser = argparse.ArgumentParser(
        description='CF rank method for group expansion')
    parser.add_argument('dataset', type=str, default='aminer', choices=['aminer', 'meetup', 'amazon'])
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weights', type=str, default='', 
        help='load pretrain GNN weights')
    parser.add_argument('--repeat-n', type=int, default=1)

    if sys.argv[1] in ['aminer', 'meetup', 'amazon']:
        dataset_function_map[sys.argv[1]](parser)    
    else:
        print('Valid dataset are aminer, meetup, amazon')        
