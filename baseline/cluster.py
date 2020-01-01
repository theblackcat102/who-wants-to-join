import numpy as np
import torch
import os, glob
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from .dataset import AMinerDataset, TOKENS, seq_collate, SocialDataset
from .models import Seq2Seq
from .utils import str2bool, confusion
from tqdm import tqdm
from collections import defaultdict
import argparse
from torch.utils.data import Dataset, DataLoader
from .utils import str2bool, confusion
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
    parser.add_argument('--embed-type', type=str, default='txt',choices=['pkl', 'txt'])
    parser.add_argument('--method', type=str, default='LINE',choices=['LINE', 'DeepWalk', 'node2vec'])


    args = parser.parse_args()
    if args.embed_type == 'txt':
        tmp_file = "{}/{}.rand.embeddings".format(args.dataset, args.dataset)
    else:
        tmp_file = "graphv/{}.{}.embeddings".format(args.dataset, args.method)
        convert_pkl2txt( 'graphv/{}-64-{}.pkl'.format(args.dataset, args.method), tmp_file )

    print("load key vector")
    model = KeyedVectors.load_word2vec_format(tmp_file)
    print("load validation dataset")
    dataset = SocialDataset(train=False, sample_ratio=args.sample_ratio,
         max_size=args.max_group, query='group', dataset=args.dataset, 
         min_freq = 5)
    member_map = dataset.member_map
    inverse_map = {}
    print('Build inverse map')
    for key, item in dataset.member_map.items():
        inverse_map[item] = key
    print(inverse_map[5])

    stats = dataset.get_stats()
    user_size= stats['member']
    
    dataloader = DataLoader(dataset, 
        # sampler=self.dist_sampler, 
        collate_fn=seq_collate,
        batch_size=1, num_workers=1, shuffle=False)
    B = 1
    stats = []
    print('Start nearest neighbour prediction')
    for batch in tqdm(dataloader, dynamic_ncols=True):
        existing_users, target_users, cnts = batch
        existing_users = existing_users.numpy()
        pred_users_rank = defaultdict(float)
        user_words = [ inverse_map[user] for user in existing_users.flatten() if user > 4 and inverse_map[user] in model.vocab ]
        if len(user_words) == 0:
            print(existing_users)
            continue

        ms = model.most_similar(positive=user_words, topn=args.topk)
        for new_user, score in ms:
            pred_users_rank[new_user] += score
        pred_users = []
        for user, score in pred_users_rank.items():
            pred_users.append((user, score))
        pred_users.sort(key=lambda x:x[1], reverse=True)
        pred_users = torch.from_numpy(np.array([[ member_map[u[0]] for u in pred_users[:args.max_size] ]])).long()


        y_onehot = torch.FloatTensor(B, user_size)
        y_onehot.zero_()

        y_pred = pred_users.flatten().unsqueeze(0)
        y_onehot.scatter_(1, y_pred, 1)
        y_onehot[:, :4] = 0.0

        y_target = torch.FloatTensor(B, user_size)
        y_target.zero_()

        y_target.scatter_(1, target_users, 1)
        y_target[:, :4] = 0.0

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


