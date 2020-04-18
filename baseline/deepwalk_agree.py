import torch
from torch import nn
import pickle
import numpy as np
from dataset.aminer import Aminer
import numpy as np
import pickle, os, glob
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.utils import dict2table, confusion, str2bool
import random
from baseline.models.agree import AGREE
from baseline.dataset import DatasetConvert, AgreeDataset



def train_epoch(dataloader, model, optimizer):
    with tqdm(total=len(dataloader), dynamic_ncols=True) as pbar:
        for batch in dataloader:
            inputs, pos, neg, group_ids = batch
            inputs, pos, neg, group_ids = inputs.cuda(), pos.cuda(), neg.cuda(), group_ids.cuda()
            # seq   B x user-size  B x user-size
            group_ids = group_ids.unsqueeze(-1)

            pos_predict = model(inputs, group_ids, pos)
            neg_predict = model(inputs, group_ids, neg)
            optimizer.zero_grad()
            loss = torch.mean((pos_predict - neg_predict -1) **2)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("loss={:.4f}".format(loss.item()))

    return model, optimizer




if __name__ == "__main__":
    user_size = 399211
    import argparse
    parser = argparse.ArgumentParser(
        description='CF rank method for group expansion')
    parser.add_ = parser.add_argument
    parser.add_('--top-k', type=int, default=5)
    parser.add_('--city', type=str, default='SF',
                        choices=['NY', 'SF'])
    parser.add_('--dataset', type=str, default='aminer',
                    choices=['meetup', 'aminer'])
    parser.add_('--user-node', type=int, default=0, 
                help='integer which user node id is represented in')
    parser.add_('--epochs', type=int, default=5, 
                help='training epochs')
    parser.add_('--seq-len', type=int, default=5, 
                help='known users seq len')
    parser.add_('--neg-sample', type=int, default=5, 
                help='negative users')
    parser.add_('--rank-margin', type=float, default=1.0, 
                help='negative loss margin')
    parser.add_('--batch-size', type=int, default=64, 
                help='known users seq len')
    parser.add_('--lr', type=float, default=1e-4, 
                help='training lr')
    parser.add_('--embeddings', type=str,
                help='graphvite embedding pickle')
    args = parser.parse_args()

    with open('graphvite_embeddings/aminer_deduplicate_train_edgelist.txt-64-DeepWalk.pkl', 'rb') as f:
        graphvite_embeddings = pickle.load(f)

    dataset = Aminer()
    data_size = len(dataset)
    if os.path.exists('.cache/{}_user2idx.pkl'.format(str(dataset))):
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'rb') as f:
            user2idx, idx2user, group2id = pickle.load(f)
    else:
        user2idx, idx2user, group2id = reindex_name2id(graphvite_embeddings, dataset)
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'wb') as f:
            pickle.dump((user2idx, idx2user, group2id), f)


    user_size = len(idx2user)
    # print(len(user2idx), len(idx2user))
    train_split, val, test = int(data_size*0.7), int(data_size*0.1), int(data_size*0.2)

    indexes = np.array(list(range(data_size)), dtype=np.long)[train_split+val:]
    test_dataset = dataset[list(indexes)]
    test_dataset = DatasetConvert(test_dataset, user_size, user2idx, group2id, max_seq=6)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4)


    indexes = np.array(list(range(data_size)), dtype=np.long)[:train_split]
    train_dataset = dataset[list(indexes)]
    group_dataset = AgreeDataset(train_dataset, user_size, user2idx, group2id, max_seq=6, mode='group')
    group_dataloader = torch.utils.data.DataLoader(group_dataset, batch_size=args.batch_size, num_workers=4)
    user_dataset = AgreeDataset(train_dataset, user_size, user2idx, group2id, max_seq=6, mode='user')
    user_dataloader = torch.utils.data.DataLoader(group_dataset, batch_size=args.batch_size, num_workers=4)

    model = AGREE(len(user2idx), len(group2id), 64, 0.1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for e in range(10):
        model, optimizer = train_epoch(user_dataloader, model, optimizer)
        model, optimizer = train_epoch(group_dataloader, model, optimizer)

    precisions, recalls = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels, masked_target, group_id = batch
            inputs, masked_target, group_id = inputs.cuda(), masked_target.cuda(), group_id.cuda()
            y_pred = model.predict_rank(inputs, group_id.unsqueeze(-1), masked_target, top_k=args.top_k)

            TP, FP, TN, FN = confusion(y_pred, labels)
            recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
            precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
            # print(precision, recall)
            precisions.append(precision)
            recalls.append(recall)
 
    avg_recalls = np.mean(recalls)
    avg_precisions = np.mean(precisions)
    f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)

    print( f1, avg_recalls, avg_precisions)