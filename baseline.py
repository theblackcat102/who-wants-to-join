import numpy as np
import torch
import os, glob
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from src.trainer import confusion
from src.dataset import SNAPCommunity
from tqdm import tqdm
from collections import defaultdict
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle


def convert_pkl2txt(filename, outputfilename):
    print('converting embedding')
    if os.path.exists(outputfilename):
        return None

    output_f = open(outputfilename, 'w')
    with open(filename, 'rb') as f:
        e = pickle.load(f)
        matrix = e['embedding']
        output_f.write('{} {}\n'.format(len(e['name2id']) ,len(matrix[0])))
        for key, idx in tqdm(e['name2id'].items()):
            output_f.write('{} '.format(key))
            output_f.write('{}\n'.format( ' '.join(map(str, list(matrix[idx])) )))
    output_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nearest Cluster Evaluation')
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--max-size', type=int, default=100)
    parser.add_argument('--sample-ratio', type=float, default=0.8)
    parser.add_argument('--max-group', type=int, default=500)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--embed-type', type=str, default='pkl',choices=['pkl', 'txt'])
    parser.add_argument('--method', type=str, default='LINE',choices=['LINE', 'DeepWalk', 'node2vec'])


    args = parser.parse_args()
    with open(args.dataset+'_shuffle_idx.pkl', 'rb') as f:
        shuffle_idx = pickle.load(f)

    if args.embed_type == 'txt':
        tmp_file = "{}/{}.rand.embeddings".format(args.dataset, args.dataset)
    else:
        tmp_file = "graphv/{}.{}.{}.embeddings".format(args.dataset, args.dim, args.method)
        convert_pkl2txt( 'graphv/{}-{}-{}.pkl'.format(args.dataset, args.dim,args.method), tmp_file )
    print("load key vector")
    model = KeyedVectors.load_word2vec_format(tmp_file)
    print("load validation dataset")
    dataset = SNAPCommunity(args.dataset, cutoff=3)

    dataset = dataset[shuffle_idx]

    split_pos = int(len(dataset)*0.7)
    train_idx = shuffle_idx[:split_pos]
    valid_idx_ = shuffle_idx[split_pos:]
    test_pos = int(len(valid_idx_)*0.666)
    test_idx = valid_idx_[:test_pos]
    valid_idx = valid_idx_[test_pos:]

    train_dataset = dataset[train_idx]

    test_dataset = dataset[test_idx]
    valid_dataset = dataset[valid_idx]
    
    user2id = dataset.user2id
    id2user = defaultdict(str)
    for key, item in user2id.items():
        id2user[item] = key
    print(id2user[5])

    B, user_size = 1, len(user2id)
    stats = []
    for data in tqdm(valid_dataset, dynamic_ncols=True):
        existing_users_idx = data.x[:, 1] 
        input_users_id = data.x[:, 0].flatten()
        existing_users_id = input_users_id[ existing_users_idx == 1 ].numpy()
        pred_users_rank = defaultdict(float)
        user_words = [ id2user[user] for user in existing_users_id if id2user[user] in model.vocab ]

        target_users = input_users_id[ data.y.flatten() == 1  ].flatten().unsqueeze(0)

        if len(user_words) == 0:
            raise ValueError('No valid exist user')

        ms = model.most_similar(positive=user_words, topn=args.topk)
        for new_user, score in ms:
            pred_users_rank[new_user] += score
        pred_users = []
        for user, score in pred_users_rank.items():
            pred_users.append((user, score))
        pred_users.sort(key=lambda x:x[1], reverse=True)
        pred_users = torch.from_numpy(np.array([[ user2id[u[0]] for u in pred_users[:args.max_size] ]])).long()


        y_onehot = torch.FloatTensor(B, user_size)
        y_onehot.zero_()

        y_pred = pred_users.flatten().unsqueeze(0)
        y_onehot.scatter_(1, y_pred, 1)

        y_target = torch.FloatTensor(B, user_size)
        y_target.zero_()

        y_target.scatter_(1, target_users, 1)

        TP, FP, TN, FN = confusion(y_onehot.long().float(), y_target.long().float())
        stats.append([TP, FP, TN, FN])

    stats = np.array(stats)
    recall = np.sum(stats[:, 0])/ (np.sum(stats[:, 0])+ np.sum(stats[:, 3]))
    precision = np.sum(stats[:, 0])/ (np.sum(stats[:, 0])+ np.sum(stats[:, 1]))

    if recall != 0:
        f1 = 2*(recall*precision)/(recall+precision)
        accuracy = (np.sum(stats[:, 0]) + np.sum(stats[:, 2]))/ (np.sum(stats))
        print('Accuracy: ',accuracy)
        print('Recall: ',recall)
        print('Precision: ',precision)
        print('F1: ',f1)
    else:
        print('Recall: ',recall)

