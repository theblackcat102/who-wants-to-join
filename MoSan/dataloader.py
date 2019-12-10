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
from copy import deepcopy
os.makedirs('.cache', exist_ok=True)

def parse_date(date_str):
    return datetime.strptime(date_str, 'YYYYMMDDhhmmss')


class Node():
    def __init__(node_type, node_id):
        self.type = node_type
        self.id = node_id

def format_token(token_name):
    return int(token_name.split('_')[1])


def extract_relation(filename):
    event_user_pair = open(filename, 'r')
    relation_event = defaultdict(list)
    for line in event_user_pair.readlines():
        tokens = line.split(' ')
        event = format_token(tokens[0])
        users = tokens[1:]
        relation_event[event] += [ format_token(u) for u in users ]
    return relation_event


def create_inv_map(relationgraph):
    relation_event = defaultdict(list)
    for key, value in relationgraph.items():
        for v in value:
            if key not in relation_event[v]:
                relation_event[v].append(key)
    return relation_event


class Meetup(Dataset):

    def __init__(self, datapath='./meetup_v1', split_ratio=0.8, sample_k=10, train=True):
        if not os.path.exists(os.path.join('.cache', 'meetup_v1_data_cache.pkl')):
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

            for data in df.to_dict('records'):
                group_id = data['group']
                if len(self.group2user[group_id]) != 0:
                    cleaned_df.append(data)
            self.data = pd.DataFrame(cleaned_df)

            cache_data = (self.data, self.group2tag, self.user2group, self.event2user, self.user2tag, self.group2user)
            with open(os.path.join('.cache', 'meetup_v1_data_cache.pkl'), 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(os.path.join('.cache', 'meetup_v1_data_cache.pkl'), 'rb') as f:
                (self.data, self.group2tag, self.user2group, self.event2user, self.user2tag, self.group2user) = pickle.load(f)

        self.data.sort_values(by='date' )
        if train:
            self.data = self.data[:int(len(self.data)*split_ratio)]
        else:
            self.data = self.data[int(len(self.data)*split_ratio):]

        self.sample_k = sample_k
        self.data.index = np.arange(len(self.data))
        print(self.data.head())
        self.length = len(self.data)

    def get_stats(self):
        max_group = []
        for key, value in self.group2user.items():
            max_group += value
        max_group.sort(reverse=True)
        return {
            'user': max_group[0],
            'group': max(self.data['group']),
            'venue': max(self.data['venue']),
            'event': max(self.data['event']),
        }


    def __getitem__(self, idx): # iterator to load data
        row = self.data.iloc[idx, :]

        neg_idx = random.randint(0, self.length-1)
        while self.data.iloc[neg_idx, :]['venue'] == row['venue'] and neg_idx != idx:
            neg_idx = random.randint(0, self.length-1)

        neg_venue = self.data.iloc[neg_idx, :]['venue']
        venue = row['venue']
        event = row['event']
        group = row['group']
        available_user = self.group2user[row['group']]
        participate_users = random.choices(available_user, k=min(self.sample_k, len(available_user)))
        interaction = []
        context_users = []
        participate_size = len(participate_users)
        attn_mask = [1] * participate_size
        if (self.sample_k) > participate_size:
            attn_mask = [1] * participate_size + [0] * (self.sample_k - participate_size)
            participate_users += [-1]*(self.sample_k - participate_size)

        for u in participate_users[:participate_size]:
            if u == -1:
                continue
            _users = deepcopy(participate_users)
            _users.remove(u)
            context_users.append(u)
            interaction.append(_users)

        if (self.sample_k) > participate_size:
            context_users += [-1]*(self.sample_k - participate_size)
            interaction += [interaction[0]]*(self.sample_k - participate_size)

        context_users = np.array(context_users)+1
        interaction = np.array(interaction)+1

        return venue, event, group, context_users, interaction, neg_venue, np.array(attn_mask)

    def __len__(self):
        return len(self.data)

def seq_collate(batch):
    venues = []
    neg_venues = []
    events = []
    groups = []
    context_users_ = []
    interactions = []
    attn_masks = []
    for row in batch:
        venue, event, group, context_users, interaction, neg_venue, attn_mask = row
        venues.append(venue)
        neg_venues.append(neg_venue)
        events.append(event)
        groups.append(group)
        attn_masks.append(attn_mask)
        context_users_.append(context_users)
        interactions.append(interaction)

    interactions = torch.from_numpy(np.asarray(interactions)).long()
    context_users_ = torch.from_numpy(np.array(context_users_)).long()
    venues = torch.from_numpy(np.array(venues)).long()
    neg_venues = torch.from_numpy(np.array(neg_venues)).long()
    groups = torch.from_numpy(np.array(groups)).long()
    events = torch.from_numpy(np.array(events)).long()
    attn_masks = torch.from_numpy(np.array(attn_masks)).float()
    return {
        'event': events,
        'group': groups,
        'venue': venues,
        'neg_venue': neg_venues,
        'context_users': context_users_,
        'interactions': interactions,
        'attn_masks': attn_masks
    }


if __name__ == "__main__":
    # datapath=  './meetup_v1'
    # df = pd.read_csv(os.path.join(datapath, 'events.txt'), 
    #             names=['event', 'venue', 'date', 'group'], sep=' ')
    # df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')
    # df['event'] = df['event'].apply(format_token)
    # print(df.head())
    # df[0]
    dataset = Meetup(train=True, sample_k=10)
    data = DataLoader(dataset, batch_size=16, num_workers=8, collate_fn=seq_collate)
    print(len(data))
    for batch in tqdm(data):
        print(batch['attn_masks'].shape)
        print(batch['interactions'].shape)
        continue
