import torch
torch.manual_seed(666)
from torch import nn
import pickle
import numpy as np
from dataset.aminer import Aminer
from dataset.meetup import Meetup, locations_id
import numpy as np
import pickle, os, glob
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.utils import dict2table, confusion, str2bool
import random
from baseline.models.deepwalk_clf import DeepwalkClf
from baseline.dataset import DatasetConvert

def reindex_name2id(graphvite_embeddings, dataset):
    # -1 : pad, -2 : UNK
    idx2user, user2idx = { 0: -1, 1: -2 }, { -1: 0, -2: 1 }
    name2id = graphvite_embeddings['name2id']
    for name, id_ in name2id.items():
        node_type, node_id = name.split('_')
        node_type, node_id = int(node_type), int(node_id)

        if node_type == 0 and node_id not in user2idx:
            idx = len(idx2user)
            idx2user[idx] = node_id
            user2idx[node_id] = idx

    group2id = {-2: 0 }

    for data in dataset:
        for node in data.x[ data.x[:, 2] == 0]:
            if int(node[0]) not in user2idx:
                idx = len(idx2user)
                idx2user[idx] = int(node[0])
                user2idx[int(node[0])] = idx
        if hasattr(data, 'titleid'):
            group2id[int(data.titleid[0])] = len(group2id)
        else:
            group_id = data.x[ data.x[:, -1] == 2, :][0][2]
            group2id[int(group_id)] = len(group2id)

    return user2idx, idx2user, group2id



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
    if args.dataset == 'aminer':
        dataset = Aminer()
    else:
        dataset = Meetup(city_id=locations_id[args.city])

    with open(args.embeddings, 'rb') as f:
        graphvite_embeddings = pickle.load(f)

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
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
        num_workers=4, shuffle=False)


    indexes = np.array(list(range(data_size)), dtype=np.long)[:train_split]
    train_dataset = dataset[list(indexes)]
    dataset = DatasetConvert(train_dataset, user_size, user2idx, group2id, max_seq=6)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    # Setup from embedding files
    embeddings = nn.Embedding(len(user2idx), 64, padding_idx=0)
    name2id = graphvite_embeddings['name2id']
    vectors = graphvite_embeddings['embedding']
    embed_weight = embeddings.weight.data.numpy()
    for name, id_ in name2id.items():
        node_type, node_id = name.split('_')
        node_type, node_id = int(node_type), int(node_id)
        if node_type == 0:
            vector = vectors[id_]
            embed_weight[user2idx[node_id]] = vector

    embeddings.weight.data.copy_(torch.from_numpy(embed_weight))
    mode = 'ranking'

    model = DeepwalkClf(embeddings, user_size, mode=mode)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
    max_index_id = 0
    criterion = nn.BCEWithLogitsLoss()
    epochs = args.epochs
    model.train()
    with tqdm(total=epochs, dynamic_ncols=True) as pbar:
        for epoch in range(epochs):
            for d in dataloader:
                inputs, labels, masked_target, group_id = d
                # seq   B x user-size  B x user-size
                inputs, labels, masked_target = inputs.cuda(), labels.cuda(), masked_target.cuda()
                optimizer.zero_grad()

                if mode == 'classifier':
                    outputs = model(inputs)
                    outputs = outputs * masked_target
                    loss = criterion(outputs, labels)
                else:
                    loss = model.forward_rank(inputs, masked_target, labels, 
                        margin=args.rank_margin,
                        neg_sample=args.neg_sample)
                # print(masked_target.shape, outputs.shape)
                #   only allow candidate loss
                loss.backward()
                optimizer.step()
                pbar.set_description("loss={:.4f}".format(loss.item()))

            pbar.update(1)

    B = 64
    precisions, recalls = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels, masked_target, group_id = d
            inputs, masked_target = inputs.cuda(), masked_target.cuda()
            B = inputs.shape[0]

            # classification only
            if mode == 'classifier':
                pred_index = model.predict(inputs, top_k=args.top_k)

                y_pred = torch.FloatTensor(B, user_size)
                y_pred.zero_()
                for batch_idx in range(B):
                    y_pred[batch_idx, pred_index[batch_idx]] = 1
            else:
                y_pred = model.predict_rank(inputs, masked_target, top_k=args.top_k)

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