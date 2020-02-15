import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
import sys
from baseline.utils import ExtractSubstructureContextPair
from baseline.dataloader import DataLoaderSubstructContext
from sklearn.metrics import roc_auc_score
import pandas as pd
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

def pool_func(x, batch, mode = "sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

criterion = nn.BCEWithLogitsLoss()

def train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device):
    model_substruct.train()

    balanced_loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        #print(batch)
        # creating substructure representation
        substruct_rep = model_substruct(batch.edge_index_substruct, batch.x_substruct)[batch.center_substruct_idx]
        
        ### creating context representations
        overlapped_node_rep = model_context(batch.edge_index_context, batch.x_context)[batch.overlap_context_substruct_idx]

        #Contexts are represented by 
        if args.mode == "cbow":
            # positive context representation
            context_rep = pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode = args.context_pooling)
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_context_rep = torch.cat([context_rep[cycle_index(len(context_rep), i+1)] for i in range(args.neg_samples)], dim = 0)
            
            pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
            pred_neg = torch.sum(substruct_rep.repeat((args.neg_samples, 1))*neg_context_rep, dim = 1)

        elif args.mode == "skipgram":

            expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(substruct_rep))], dim = 0)
            pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim = 1)

            #shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(args.neg_samples):
                shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), i+1)]
                shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

            shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
            pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((args.neg_samples, 1)), dim = 1)

        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()

        loss = loss_pos + args.neg_samples*loss_neg
        loss.backward()
        #To write: optimizer
        optimizer_substruct.step()
        optimizer_context.step()

        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += 0.5* (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))

    return balanced_loss_accum/step, acc_accum/step

def pretrain_aminer(parser):
    from dataset.aminer import Aminer
    from src.layers import StackedGCNDBLP

    group = parser.add_argument_group('aminer parameters')
    group.add_argument('--author-dim', type=int, default=16)
    group.add_argument('--paper-dim', type=int, default=16)
    group.add_argument('--conf-dim', type=int, default=8)
    group.add_argument('--input-dim', type=int, default=32)
    group.add_argument('--dropout', type=float, default=0.1)
    group.add_argument('--layers', nargs='+', type=int, default=[32, 32])
    group.add_argument('--output-dim',type=int, default=8)

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = Aminer(transform=ExtractSubstructureContextPair(1, center=False))
    if osp.exists('dblp_hete_shuffle_idx.pkl'):
        with open('dblp_hete_shuffle_idx.pkl', 'rb') as f:
            shuffle_idx = pickle.load(f)

    split_pos = int(len(dataset)*0.7)

    dataset.list_of_data = np.array(dataset.list_of_data)[shuffle_idx[:split_pos]]
    
    loader = DataLoaderSubstructContext(dataset, batch_size=16, 
        shuffle=True, num_workers=8)


    model_substruct = StackedGCNDBLP(
                author_size=874608,#len(author2id),
                paper_size=3605603,#len(paper2id),
                conf_size=12770, output_channels=args.output_dim,
                user_dim=args.author_dim,
                paper_dim=args.paper_dim,
                conf_dim=args.conf_dim,
                input_channels=args.input_dim,
                layers=args.layers,
                dropout=args.dropout).to(device)
    model_context = StackedGCNDBLP(
                author_size=874608,#len(author2id),
                paper_size=3605603,#len(paper2id),
                conf_size=12770, output_channels=args.output_dim,
                user_dim=args.author_dim,
                paper_dim=args.paper_dim,
                conf_dim=args.conf_dim,
                input_channels=args.input_dim,
                layers=args.layers,
                dropout=args.dropout).to(device)

    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc = train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device)
        print(train_loss, train_acc)

    if not args.model_file == "":
        torch.save(model_substruct.state_dict(), args.model_file + ".pth")

def pretrain_meetup(parser):
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

    with open(osp.join(MEETUP_FOLDER, 'topic2id.pkl'), 'rb') as f:
        topic2id = pickle.load(f)
    # with open(osp.join(MEETUP_FOLDER, 'group2topic.pkl'), 'rb') as f:
    #     group2topic = pickle.load(f)


    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = Meetup(city_id=locations_id[args.city],
                         cutoff=args.maxhop, min_size=args.min_size,
                         max_size=args.max_size,
                         transform=ExtractSubstructureContextPair(1, center=False))
    category_size = 40
    topic_size = len(topic2id)
    group_size = len(dataset.group2id)

    if osp.exists(args.city+'_shuffle_idx.pkl'):
        with open(args.city+'_shuffle_idx.pkl', 'rb') as f:
            shuffle_idx = pickle.load(f)

    split_pos = int(len(dataset)*0.7)

    dataset.processed_file_idx = np.array(dataset.processed_file_idx)[shuffle_idx[:split_pos]]
    
    loader = DataLoaderSubstructContext(dataset, batch_size=8, 
        shuffle=True, num_workers=8)


    model_substruct = StackedGCNMeetup(user_size=len(dataset.user2id),
                                 category_size=category_size,
                                 topic_size=topic_size,
                                 group_size=group_size,
                                 input_channels=args.input_dim,
                                 output_channels=args.output_dim,
                                 layers=args.layers,
                                 dropout=args.dropout).to(device)
    model_context = StackedGCNMeetup(user_size=len(dataset.user2id),
                                 category_size=category_size,
                                 topic_size=topic_size,
                                 group_size=group_size,
                                 input_channels=args.input_dim,
                                 output_channels=args.output_dim,
                                 layers=[8],
                                 dropout=args.dropout).to(device)

    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc = train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device)
        print(train_loss, train_acc)

    if not args.model_file == "":
        torch.save(model_substruct.state_dict(), args.model_file + ".pth")    

def pretrain_amazon(parser):
    print("Amazon")
    from src.layers import StackedGCNAmazon
    from dataset.amazon import AmazonCommunity    
    group = parser.add_argument_group('amazon parameters')
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
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = AmazonCommunity(transform=ExtractSubstructureContextPair(1, center=False))
    if osp.exists('dblp_hete_shuffle_idx.pkl'):
        with open('dblp_hete_shuffle_idx.pkl', 'rb') as f:
            shuffle_idx = pickle.load(f)
    with open('data/amazon/cat2id.pkl', 'rb') as f:
        cat2id = pickle.load(f)

    category_size = len(cat2id)

    split_pos = int(len(dataset)*0.7)

    dataset.processed_file_idx = np.array(dataset.processed_file_idx)[shuffle_idx[:split_pos]]
    
    loader = DataLoaderSubstructContext(dataset, batch_size=16, 
        shuffle=True, num_workers=8)


    model_substruct = StackedGCNAmazon(len(dataset.user2id),
                                 category_size=category_size,
                                 user_dim=args.user_dim,
                                 category_dim=args.cat_dim,
                                 input_channels=args.input_dim,
                                 layers=args.layers,
                                 dropout=args.dropout).to(device)
    model_context = StackedGCNAmazon(len(dataset.user2id),
                                 category_size=category_size,
                                 user_dim=args.user_dim,
                                 category_dim=args.cat_dim,
                                 input_channels=args.input_dim,
                                 layers=args.layers,
                                 dropout=args.dropout).to(device)

    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc = train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device)
        print(train_loss, train_acc)

    if not args.model_file == "":
        torch.save(model_substruct.state_dict(), args.model_file + ".pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='CF rank method for group expansion')
    parser.add_argument('dataset', type=str, default='aminer', choices=['aminer', 'meetup', 'amazon'])    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--mode', type=str, default = "cbow", help = "cbow or skipgram")
    parser.add_argument('--model-file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--context_pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    parser.add_argument('--center', type=int, default=0,
                        help='center (default: 0).')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')

    dataset_function_map = {
        'aminer': pretrain_aminer,
        'meetup': pretrain_meetup,
        'amazon': pretrain_amazon,
    }
    if sys.argv[1] in ['aminer', 'meetup', 'amazon']:
        dataset_function_map[sys.argv[1]](parser)
    else:
        print('Valid dataset are aminer, meetup, amazon')        


