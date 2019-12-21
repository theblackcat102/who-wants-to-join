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

class Meetupv2(Dataset):

    def __init__(self, datapath='./meetup_v2', split_ratio=0.8, sample_ratio=0.5, 
        order_shuffle=True,
        train=True, query='group', pred_size=100, max_size=5000, min_freq=5, city='nyc'):
        if city not in city_map:
            raise ValueError("Invalid city name")
        self.query = query
        train_str = 'train_' if train else 'test_'
        filename = '{}_freq_{}_{}_{}_meetup_v2.1_data_cache.pkl'.format(self.query, min_freq,city, max_size )
        cache_path = os.path.join('.cache', filename)
        if not os.path.exists(cache_path):
            group = pd.read_csv(os.path.join(datapath, 'groups.csv'))
            group['created'] = pd.to_datetime(group['created'], format="%Y-%m-%d %H:%M:%S")
            group.sort_values(by='created')

            members = pd.read_csv(os.path.join(datapath, 'members.csv'), encoding='latin-1')
            member_frequency_ = Counter(members['member_id'].to_list())
            valid_member = []
            for m, frequency in member_frequency_.items():
                if frequency > min_freq:
                    valid_member.append(m)
            members = members[ members['member_id'].isin(valid_member) ]
            members = members[ members['city'].isin(city_map[city]) ]

            print('create group2user')
            self.group2user = create_relation(members, 'member_id', 'group_id')
            keys = list(self.group2user.keys())
            for group_id in keys:
                users = self.group2user[group_id]
                if len(users) < 10 or len(users) > max_size:
                    self.group2user.pop(group_id, None)

            group_topics = pd.read_csv(os.path.join(datapath, 'groups_topics.csv'), encoding='latin-1')
            print('create group2tag')
            self.group2tag = create_relation(group_topics, 'topic_id', 'group_id')
            print('create member mapping')
            self.member_map = create_user_map(self.group2user)

            self.member_frequency = member_frequency_
            print('create topic mapping')
            self.topic_map = create_mapping(group_topics, 'topic_id')
            print('create group mapping')
            self.group_map = create_dict_map(self.group2user)

            self.data = group
            self.keys = list(self.group2user.keys())
            random.shuffle(self.keys)
            cache_data = (self.data, self.group2tag, self.group2user, self.group_map, self.member_map, self.topic_map, self.keys, self.member_frequency)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(cache_path, 'rb') as f:
                (self.data, self.group2tag, self.group2user, self.group_map, self.member_map, self.topic_map, self.keys, self.member_frequency) = pickle.load(f)

        self.sample_rate = sample_ratio
        # print(self.data.head())
        pos = int(split_ratio*len(self.keys))
        if train:
            self.data = self.keys[:pos]
        else:
            self.data = self.keys[pos:]
        self.max_size = max_size
        self.order_shuffle = order_shuffle
        # self.data.index = np.arange(len(self.data))

    def get_stats(self):
        # print(self.keys)
        return {
            'group': len(self.group_map),
            'member': len(self.member_map),
            'topic': len(self.topic_map),
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
        pred_users_max_size = int(self.max_size*(1-self.sample_rate))
        existing_users_max_size = int(self.max_size*(self.sample_rate))

        if len(pred_users) > pred_users_max_size:
            pred_users = pred_users
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


        tags = self.group2tag[group_id]
        if len(tags) < 20:
            tags = ['PAD'] * (20 - len(tags)) + tags
        tags = [ self.topic_map[t] for t in tags[:20] ]

        # existing_users = np.array(existing_users)
        # pred_users = np.array(pred_users)

        return existing_users, pred_users, pred_users_cnt, tags, self.member_map['PAD']

class Meetupv1(Dataset):

    def __init__(self, datapath='./Meetup_data/NYC', split_ratio=0.8, sample_ratio=0.5, train=True, query='group', pred_size=100, max_size=5000):
        self.query = query
        
        train_str = 'train' if train else 'test'
        cache_name = train_str+ '_'+self.query+ '_query_nyc_meetup_v1_data_cache.pkl'

        if not os.path.exists(os.path.join('.cache',cache_name)):
            datapath = os.path.join(datapath, train_str)
            df = pd.read_csv(os.path.join(datapath, 'events.txt'), 
                names=['event', 'venue', 'date', 'group'], sep=' ')

            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')
            df['event'] = df['event'].apply(format_token)
            df['venue'] = df['venue'].apply(format_token)
            df['group'] = df['group'].apply(format_token)
            

            self.group2tag = extract_relation(os.path.join(datapath, 'group_tags.txt'))
            self.user2group = extract_relation(os.path.join(datapath, 'user_groups.txt'))
            self.user2tag = extract_relation(os.path.join(datapath, 'user_tags.txt'))
            self.event2user = extract_relation(os.path.join(datapath, 'event_users.txt'))
            self.group2user = create_inv_map(self.user2group)
            cleaned_df = []
            filter_relation = self.event2user

            if self.query == 'group':
                filter_relation = self.group2user

            for data in df.to_dict('records'):
                event_id = data[self.query]
                if len(filter_relation[event_id]) >= 10:
                    cleaned_df.append(data)
            self.data = pd.DataFrame(cleaned_df)

            self.positive_sample = self.group2user
            if query == 'event':
                self.positive_sample = self.event2user


            cache_data = (self.data, self.group2tag, self.user2group, self.event2user, self.user2tag, self.group2user)
            with open(os.path.join('.cache', cache_name), 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(os.path.join('.cache', cache_name), 'rb') as f:
                (self.data, self.group2tag, self.user2group, self.event2user, self.user2tag, self.group2user) = pickle.load(f)
        self.df = self.data
        if query == 'group':
            self.df = self.data.drop_duplicates(['group'])
        self.df.sort_values(by='date')


        self.sample_rate = sample_ratio
        self.max_size = max_size
        self.df.index = np.arange(len(self.df))
        print(self.df.head())
        self.positive_sample = self.group2user
        self.train = train
        if query == 'event':
            self.positive_sample = self.event2user
        self.data = []
        for key, _ in self.positive_sample.items():
            self.data.append(key)
        self.length = len(self.data)


    def get_stats(self):
        max_group = []
        for key, value in self.group2tag.items():
            max_group.append(len(value))
        max_group.sort(reverse=True)
        print('group size', max_group[:10])

        max_group = []
        for key, value in self.group2user.items():
            # max_group.append(len(value))
            max_group += value
        max_group.sort(reverse=True)
        print(max_group[:10])
        print(np.max(max_group))

        return {
            'user': max_group[0],
            'group': max(self.df['group']),
            'venue': max(self.df['venue']),
            'event': max(self.df['event']),
        }


    def __getitem__(self, idx): # iterator to load data
        # row = self.data.iloc[idx, :]

        # event = row[self.query]
        event = self.data[idx]

        available_user = self.positive_sample[event]

        # negative_users = self.negative_sample[event]
        # negative_users = random.choices(negative_users, k=int(self.sample_rate*len(negative_users)))
        # select_rate = random.random()*self.sample_rate
        # if not self.train:
        select_rate = self.sample_rate

        existing_users = random.choices(available_user, k=int(select_rate*len(available_user)))
        # print(len(available_user), len(existing_users))
        interaction = []
        context_users = []
        participate_size = len(existing_users)
        attn_mask = [1] * participate_size

        pred_users = deepcopy(available_user)
        for u in existing_users:
            if u in pred_users:
                pred_users.remove(u)

        if len(pred_users) > self.max_size:
            pred_users = pred_users
            pred_users = pred_users[:self.max_size-1]

        pred_users_cnt = len(pred_users)-1
        pred_users += [-2]*(self.max_size - len(pred_users))

        if len(existing_users) > self.max_size:
            existing_users = existing_users[:self.max_size]
        existing_users += [-2]*(self.max_size - len(existing_users))

        existing_users = np.array(existing_users)+3
        pred_users = np.array(pred_users)+3
        tags = None
        if self.query == 'group':
            tags = self.group2tag[event]
            if len(tags) < 20:
                tags += [0] * (20 - len(tags))
            tags = tags[:20]

        return existing_users, pred_users, pred_users_cnt, np.array(tags)

    def __len__(self):
        return len(self.data)


class SocialDataset(Dataset):
    def __init__(self, dataset='amazon', split_ratio=0.8, sample_ratio=0.5, 
        order_shuffle=True,
        train=True, query='group', pred_size=100, max_size=5000, min_size=10, min_freq=4):
        if dataset not in ['amazon', 'orkut', 'lj','friendster', 'youtube']:
            raise ValueError('Invalid dataset')
        self.query = query
        train_str = 'train_' if train else 'test_'
        filename = '{}_freq_{}_{}-{}_{}_cache.pkl'.format(self.query, min_freq, max_size, min_size, dataset )
        cache_path = os.path.join('.cache', filename)
        if not os.path.exists(cache_path):
            community_filename = '{}/com-{}.all.cmty.txt'.format(dataset, dataset)
            with open(community_filename, 'r') as f:
                group2user = defaultdict(list)
                members = []
                for line in f.readlines():
                    members_ = line.strip().split('\t')
                    members_ = [  m.strip() for m in members_ ]
                    if len(members_) >= min_size and len(members_) <= max_size:
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
                        if m in member_map and member_map[m] != 0 and member_frequency_[m] > min_freq:
                            self.group2user[group_id].append(m)
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

        pred_users = random.choices(available_user, k=max(int((1-select_rate)*len(available_user)), 2 ) )
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


    test = SocialDataset(train=False, sample_ratio=0.8, query='group', max_size=500, dataset='youtube', 
        min_freq=20)
    train = SocialDataset(train=True, sample_ratio=0.8, query='group', max_size=500, dataset='youtube', 
        min_freq=20)

    # test = Meetupv2(train=False, sample_ratio=0.8, query='group', max_size=500, city='nyc', min_freq=5)
    # train = Meetupv2(train=True, sample_ratio=0.8, query='group', max_size=500, city='nyc', min_freq=5)
    print('total: ', len(test)+len(train))
    print('Test size : {}, Train size : {}'.format(len(test), len(train)))
    print(train.get_stats())
    print(test.get_stats())

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
    # data = DataLoader(train, batch_size=16, num_workers=8, collate_fn=seq_collate)
    # for batch in data:
    #     existing_users, pred_users, cnts = batch
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
