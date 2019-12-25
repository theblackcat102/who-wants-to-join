from random import randint, shuffle
from random import random as rand
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pickle
import random
from collections import Counter
from .utils import format_token, extract_relation, create_inv_map
from copy import deepcopy
os.makedirs('.cache', exist_ok=True)

def parse_date(date_str):
    return datetime.strptime(date_str, 'YYYYMMDDhhmmss')

TOKENS = {
    'UNK': 0,
    'EOS': 1,
    'PAD': 2,
    'BOS': 3,
}

city_map = {
    'chicago': ['Chicago', 'Chicago Heights', 'Chicago Park', 'Chicago Ridge','East Chicago','West Chicago'],
    'nyc': ['New York','New York Mills','West New York'],
    'sf': ['san francisco','San Francisco','South San Francisco']
}
'''
    EOS : 2
    PAD : 1
    BOS : 0
'''


def create_relation(df, id_key, foreign_key):
    data = defaultdict(list)
    for idx, row in df.iterrows():
        data[row[foreign_key]].append(row[id_key])
    return data


def create_mapping(df, id_key):
    data = defaultdict(int)
    for key, value in TOKENS.items():
        data[key] = len(data)

    for unique_key in tqdm(df[id_key].unique()):
        if unique_key not in data:
            data[unique_key] = len(data)
    return data


def create_dict_map(maping):
    data = defaultdict(int)
    for key, value in TOKENS.items():
        data[key] = len(data)

    for key, _ in maping.items():
        data[key] = len(data)
    return data

def create_user_map(maping):
    data = defaultdict(int)
    for key, value in TOKENS.items():
        data[key] = len(data)

    for _, users in maping.items():
        for u in users:
            if u not in data:
                data[u] = len(data)
    return data

def load_embeddings(filename):
    embeddings = {}
    with open(filename, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                size, dims = line.split(' ')
                continue
            embed = line.strip().split(' ')
            key = int(embed[0])
            embedding = [ float(v) for v in embed[1:] ]
            if str(key) in embeddings:
                print('found repeated keys')

            embeddings[str(key)] = embedding
    return embeddings, int(size), int(dims)



def seq_collate(batches):
    max_exists_user = max([len(u[0]) for u in batches ])
    max_pred_user = max([len(u[1]) for u in batches ])
    existing_users, pred_users, tags, cnts = [], [], [], []
    for batch in batches:
        if len(batch) == 4:
            tag = None
            existing_user, pred_user, cnt, pad_idx = batch
        else:
            existing_user, pred_user, cnt, tag, pad_idx = batch
        existing_users.append( np.array([pad_idx]*(max_exists_user - len(existing_user)) + existing_user))
        pred_users.append( np.array(pred_user + [pad_idx]*(max_pred_user - len(pred_user))))
        if tag is not None:
            tags.append(tag)
        cnts.append(cnt)
    pred_users = np.array(pred_users)
    pred_users = torch.from_numpy(np.array(pred_users)).long()
    existing_users = torch.from_numpy(np.array(existing_users)).long()
    cnts = torch.from_numpy(np.array(cnts)).long()
    if len(tags) > 0:
        tags = torch.from_numpy(np.array(tags)).long()
        return existing_users, pred_users, cnts, tags
    return existing_users, pred_users, cnts

class SocialDataset(Dataset):
    def __init__(self, dataset='amazon', split_ratio=0.8, sample_ratio=0.5, 
        order_shuffle=True,
        train=True, query='group', pred_size=100, max_size=5000, min_size=10, min_freq=4):
        if dataset not in ['amazon', 'orkut', 'lj','friendster', 'youtube']:
            raise ValueError('Invalid dataset')
        self.query = query
        train_str = 'train_' if train else 'test_'
        filename = '{}_freq_{}_{}-{}_{}_cache.pkl'.format(self.query, min_freq, max_size, min_size, dataset )
        print(filename)
        cache_path = os.path.join('.cache', filename)
        if not os.path.exists(cache_path):
            community_filename = '{}/com-{}.all.cmty.txt'.format(dataset, dataset)
            with open(community_filename, 'r') as f:
                group2user = defaultdict(list)
                members = []
                for line in f.readlines():
                    members_ = line.strip().split('\t')
                    members_ = [  m.strip() for m in members_ ]
                    if len(members_) >= min_size:
                        group2user[len(group2user) ] = members_
                        members += members_
                
                member_frequency_ = Counter(members)
                valid_member = []
                member_map = defaultdict(int)
                for key, value in TOKENS.items():
                    member_map[key] = len(member_map)
                self.member_frequency = defaultdict(int)
                for m, frequency in member_frequency_.items():
                    # print(frequency)
                    if frequency >= min_freq:
                        self.member_frequency[m] = frequency
                        member_map[m] = len(member_map)
                        valid_member.append(m)

                self.group2user = defaultdict(list)
                for group_id, members_ in group2user.items():
                    for m in members_:
                        if m in member_map and member_map[m] != 0:
                            self.group2user[group_id].append(m)
                self.member_map = member_map
                self.keys = list(self.group2user.keys())
                random.shuffle(self.keys)

                cache_data = ( self.group2user, self.member_map, self.member_frequency, self.keys )
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(cache_path, 'rb') as f:
                ( self.group2user, self.member_map, self.member_frequency, self.keys ) = pickle.load(f)
        self.sample_rate = sample_ratio
        # print(self.data.head())
        pos = int(split_ratio*len(self.keys))
        if train:
            self.data = self.keys[:pos]
        else:
            self.data = self.keys[pos:]
        self.max_size = max_size
        self.order_shuffle = order_shuffle
        self.embedding = None
        if os.path.exists('{}/{}.rand.embeddings'.format(dataset, dataset)) and train:
            print('found embeddings')
            count = 0
            print(list(self.member_map)[:10])
            embeddings_dict, _, dimension = load_embeddings('{}/{}.rand.embeddings'.format(dataset, dataset))
            embedding = np.zeros((len(self.member_map), dimension))
            not_matched = []
            # pickle.load(open('temp.pkl', 'rb'))
            with open('temp.pkl', 'wb') as f:
                pickle.dump(embeddings_dict, f)
            print(len(embeddings_dict))

            count = 0
            for key, idx in self.member_map.items():
                if key not in TOKENS and key in embeddings_dict:
                    count += 1
                    embedding[idx, :] = embeddings_dict[key]
            self.embedding = embedding

            print('not matched: ',not_matched[:10])
            print("Added {}/{} embeddings".format(count, len(self.member_map)))

    def get_stats(self):
        # print(self.keys)
        stats = []
        for m, freq in self.member_frequency.items():
            stats.append(freq)
        print(np.mean(stats), np.max(stats), np.min(stats))

        return {
            'member': len(self.member_map),
        }
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # iterator to load data
        # row = self.data.iloc[idx, :]
        group_id = self.data[idx]#row['group_id']
        # event = self.data[idx]
        available_user = self.group2user[group_id]

        select_rate = self.sample_rate

        pred_users = random.choices(available_user, k=max(int((1-select_rate)*len(available_user)), 1 ) )
        # print(len(available_user), len(existing_users))
        interaction = []
        context_users = []

        existing_users = deepcopy(available_user)
        for u in pred_users:
            if u in existing_users:
                existing_users.remove(u)

        if self.order_shuffle is False:
            existing_users.sort(key=lambda x: self.member_frequency[x], reverse=True)
            pred_users.sort( key=lambda x: self.member_frequency[x], reverse=True)
        else:
            random.shuffle(existing_users)
            random.shuffle(pred_users)

        pred_users += ['EOS']
        # print(int(self.max_size*(self.sample_rate)),  int(self.max_size*(1-self.sample_rate)))
        pred_users_max_size = int(self.max_size*(1-self.sample_rate))
        existing_users_max_size = int(self.max_size*(self.sample_rate))
        if len(pred_users) > pred_users_max_size:
            pred_users = pred_users[:pred_users_max_size-1]
            pred_users += ['EOS'] # eos

        pred_users_cnt = len(pred_users)-1
        # pred_users += ['PAD']*(pred_users_max_size - len(pred_users))
        pred_users = [ self.member_map[e] for e in pred_users]
        existing_users += ['EOS']

        if len(existing_users) > existing_users_max_size:
            existing_users = existing_users[:existing_users_max_size-1]
            existing_users += ['EOS']
        # existing_users = ['PAD']*(existing_users_max_size - len(existing_users)) + existing_users
        existing_users = [  self.member_map[e] for e in existing_users]

        return existing_users, pred_users, pred_users_cnt, self.member_map['PAD']


class AMinerDataset(Dataset):
    def __init__(self, dataset='acm', split_ratio=0.8, sample_ratio=0.5, 
        order_shuffle=True,
        train=True, query='group', pred_size=100, max_size=5000, min_size=4, min_freq=4):
        if dataset not in ['acm', 'dblp']:
            raise ValueError('Invalid dataset')
        filename = '{}_{}-{}_{}_cache.pkl'.format( min_freq, max_size, min_size, dataset )
        cache_path = os.path.join('.cache', filename)
        if not os.path.exists(cache_path):
            citation_filename = 'aminer/{}.txt'.format(dataset)
            group2user = defaultdict(list)
            members = []
            with open(citation_filename, 'r') as f:
                for line in f.readlines():
                    if '#@' in line:
                        authors = line.strip().split('#@')[-1].split(',')
                        if len(authors) >= min_size:
                            group2user[len(group2user)] = authors
                            members += authors

            member_frequency_ = Counter(members)
            valid_member = []
            member_map = defaultdict(int)
            for key, value in TOKENS.items():
                member_map[key] = len(member_map)
            self.member_frequency = defaultdict(int)
            for m, frequency in member_frequency_.items():
                # print(frequency)
                if frequency >= min_freq:
                    self.member_frequency[m] = frequency
                    member_map[m] = len(member_map)
            self.group2user = defaultdict(list)
            for group_id, members_ in group2user.items():
                group_member = []
                for m in members_:
                    if m in member_map and member_map[m] != 0:
                        group_member.append(m)
                if len(group_member) > min_size:
                    self.group2user[group_id] = group_member
            #     group2user[group_id] = members_
            # self.group2user = group2user
            self.member_map = member_map
            self.keys = list(self.group2user.keys())
            random.shuffle(self.keys)

            cache_data = ( self.group2user, self.member_map, self.member_frequency, self.keys )
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(cache_path, 'rb') as f:
                ( self.group2user, self.member_map, self.member_frequency, self.keys ) = pickle.load(f)
        self.sample_rate = sample_ratio
        # print(self.data.head())
        pos = int(split_ratio*len(self.keys))
        if train:
            self.data = self.keys[:pos]
        else:
            self.data = self.keys[pos:]
        self.max_size = max_size
        self.order_shuffle = order_shuffle                                        
    def get_stats(self):
        # print(self.keys)
        stats = []
        for m, freq in self.member_frequency.items():
            stats.append(freq)
        print(np.mean(stats), np.max(stats), np.min(stats))

        return {
            'member': len(self.member_map),
        }
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # iterator to load data
        # row = self.data.iloc[idx, :]
        group_id = self.data[idx]#row['group_id']
        # event = self.data[idx]
        available_user = self.group2user[group_id]

        select_rate = self.sample_rate

        pred_users = random.choices(available_user, k=max(int((1-select_rate)*len(available_user)), 1 ) )
        # print(len(available_user), len(existing_users))
        interaction = []
        context_users = []

        existing_users = deepcopy(available_user)
        for u in pred_users:
            if u in existing_users:
                existing_users.remove(u)

        if self.order_shuffle is False:
            existing_users.sort(key=lambda x: self.member_frequency[x], reverse=True)
            pred_users.sort( key=lambda x: self.member_frequency[x], reverse=True)
        else:
            random.shuffle(existing_users)
            random.shuffle(pred_users)

        pred_users += ['EOS']
        # print(int(self.max_size*(self.sample_rate)),  int(self.max_size*(1-self.sample_rate)))
        pred_users_max_size = int(self.max_size*(1-self.sample_rate))
        existing_users_max_size = int(self.max_size*(self.sample_rate))
        if len(pred_users) > pred_users_max_size:
            pred_users = pred_users[:pred_users_max_size-1]
            pred_users += ['EOS'] # eos

        pred_users_cnt = len(pred_users)-1
        # pred_users += ['PAD']*(pred_users_max_size - len(pred_users))
        pred_users = [ self.member_map[e] for e in pred_users]
        existing_users += ['EOS']

        if len(existing_users) > existing_users_max_size:
            existing_users = existing_users[:existing_users_max_size-1]
            existing_users += ['EOS']
        # existing_users = ['PAD']*(existing_users_max_size - len(existing_users)) + existing_users
        existing_users = [  self.member_map[e] for e in existing_users]
        return existing_users, pred_users, pred_users_cnt, self.member_map['PAD']

if __name__ == "__main__":
    # from .models import Seq2SeqwTag
    import torch.nn as nn
    criterion = nn.NLLLoss(ignore_index=TOKENS['PAD'])


    # test = AMinerDataset(train=False, sample_ratio=0.8, query='group', max_size=500, dataset='acm', 
    #     min_freq=4)
    # train = AMinerDataset(train=True, sample_ratio=0.8, query='group', max_size=500, dataset='acm', 
    #     min_freq=4)

    # test = SocialDataset(train=False, sample_ratio=0.8, query='group', max_size=500, dataset='amazon', 
    #     min_freq=4)
    train = SocialDataset(train=True, sample_ratio=0.8, query='group', max_size=500, dataset='amazon', 
        min_freq=10)
    # print(train.get_stats())
    # train = SocialDataset(train=True, sample_ratio=0.8, query='group', max_size=500, dataset='lj', 
    #     min_freq=4)
    # print(train.get_stats())

    # test = Meetupv2(train=False, sample_ratio=0.8, query='group', max_size=500, city='nyc', min_freq=5)
    # train = Meetupv2(train=True, sample_ratio=0.8, query='group', max_size=500, city='nyc', min_freq=5)
    # print('total: ', len(test)+len(train))
    # print('Test size : {}, Train size : {}'.format(len(test), len(train)))
    # print(train.get_stats())
    # print(test.get_stats())

    # stats = train.get_stats()
    # model = Seq2SeqwTag(
    #     embed_size=32,
    #     vocab_size=stats['member']+3,
    #     hidden_size=64,
    #     enc_num_layers=2,
    #     dec_num_layers=2,dropout=0.1,
    #     st_mode=False,
    #     use_attn=True
    # )
    data = DataLoader(train, batch_size=16, num_workers=8, collate_fn=seq_collate)
    for batch in data:
        existing_users, pred_users, cnts = batch
        print(pred_users.cnts)
        train.sample_rate -= 0.1
    #     print(pred_users[:2])
    #     break
    #     print(existing_users.size(1),pred_users.size(1))
    #     decoder_outputs, d_h, hidden = model(existing_users, pred_users, tags)
    #     seq_length = decoder_outputs.shape[1]
    #     loss = 0
    #     print(seq_length)
    #     for t in range(seq_length):
    #         loss_ = criterion(torch.log(decoder_outputs[:, t, :]), pred_users[:,t+1], )
    #         loss += loss_
    #     print(loss)
    # print(len(dataset.group2user))

    # print('data size', len(dataset))
    # for batch in tqdm(data):
    #     existing_users, pred_users, _, tags = batch
    #     print(existing_users)
    #     print(pred_users.flatten().max())
    # y_onehot = torch.FloatTensor(8, 324)

    # label_idx = torch.randint(0, 324, (8, 32))
    # # label = torch.zeros(8*32, 324)
    # # label[:, label_idx.flatten()] = 1.0
    # y_onehot.zero_()
    # y_onehot.scatter_(1, label_idx, 1)

    # print(y_onehot.shape)

    # print(y_onehot[0])
    # print(label_idx[0])
