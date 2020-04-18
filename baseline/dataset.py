import torch
# torch.manual_seed(0)
from torch import nn
import pickle
import numpy as np
import pickle, os, glob
from tqdm import tqdm
from sklearn.metrics import f1_score
import random



class DatasetConvert(torch.utils.data.Dataset):
    def __init__(self, dataset, user_size, user2idx, group2id, max_seq = 10):
        self.dataset = dataset
        self.user_size = user_size
        self.user2idx = user2idx
        self.max_seq = max_seq
        self.group2id = group2id
        self.group_func = DatasetConvert.meetup_group_id if hasattr(dataset, 'group_size') else DatasetConvert.aminer_group_id

    @staticmethod
    def aminer_group_id(data):
        return int(data.titleid[0])

    @staticmethod
    def meetup_group_id(data):
        group_id = data.x[ data.x[:, -1] == 2, :][0][2]
        return int(group_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, user_node_id=0):
        data = self.dataset[idx]
        known_user_node_id = (data.x[:, 2] == user_node_id) & (data.x[:, 1] == 1)

        group_node_id  = self.group2id[self.group_func(data)]

        known_nodes = []
        for node in data.x[known_user_node_id, :]:
            idx = self.user2idx[-2]
            if int(node[0]) in self.user2idx:
                idx = self.user2idx[int(node[0])]
            known_nodes.append(idx)

        known_nodes = list(set(known_nodes))
        while len(known_nodes) < self.max_seq:
            known_nodes.append(self.user2idx[-1])

        known_nodes = np.array(known_nodes)[:self.max_seq]

        target_node_id = []
        for node in data.x[data.y == 1, :]:
            if int(node[0]) in self.user2idx:
                target_node_id.append(self.user2idx[int(node[0])])
            else:
                print(int(node[0]))

        y_target = np.zeros(self.user_size)
        y_target.fill(0)
        y_target[target_node_id] = 1.0


        mask_node_id = []
        # not known candidate group
        candidate_user_node_id = (data.x[:, 2] == user_node_id) & (data.x[:, 1] != 1)
        for node in data.x[candidate_user_node_id, :]:
            if int(node[0]) in self.user2idx:
                mask_node_id.append(self.user2idx[int(node[0])])

        # candidate_user_node_id = [ self.user2idx[int(node[0])] for node in data.x[candidate_user_node_id, :]]
        masked_target = np.zeros(self.user_size)
        masked_target.fill(0)
        # print(mask_node_id)
        # print(target_node_id)
        masked_target[mask_node_id] = 1.0

        return torch.from_numpy(known_nodes).long(), \
                torch.from_numpy(y_target), torch.from_numpy(masked_target).long(), group_node_id




class AgreeDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, user_size, user2idx, group2id, max_seq = 10, mode='group'):
        self.dataset = dataset
        self.user_size = user_size
        self.user2idx = user2idx
        self.max_seq = max_seq
        self.group2id = group2id
        self.mode = mode
        self.group_func = DatasetConvert.meetup_group_id if hasattr(dataset, 'group_size') else DatasetConvert.aminer_group_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, user_node_id=0):
        data = self.dataset[idx]
        known_user_node_id = (data.x[:, 2] == user_node_id) & (data.x[:, 1] == 1)

        group_node_id  = self.group2id[self.group_func(data)]
        if self.mode == 'group':
            known_nodes = []
            for node in data.x[known_user_node_id, :]:
                idx = self.user2idx[-2]
                if int(node[0]) in self.user2idx:
                    idx = self.user2idx[int(node[0])]
                known_nodes.append(idx)

            known_nodes = list(set(known_nodes))
            while len(known_nodes) < self.max_seq:
                known_nodes.append(self.user2idx[-1])

            known_nodes = np.array(known_nodes)[:self.max_seq]
        else:
            known_nodes = []
            for node in data.x[known_user_node_id, :]:
                if int(node[0]) in self.user2idx:
                    idx = self.user2idx[int(node[0])]
                    known_nodes.append(idx)
            select_index = np.random.randint(0, len(known_nodes))
            known_nodes = np.array([known_nodes[select_index]])

        target_index = (data.y == 1).nonzero()
        select_index = target_index[torch.randint(0, len(target_index), (1,))]

        positive_node = self.user2idx[int(data.x[select_index[0]][0][0])]

        # not known candidate group
        negative_node_id = ((data.x[:, 2] == user_node_id) & (data.x[:, 1] != 1) & (data.y != 1)).nonzero()
        if len(negative_node_id) == 0:
            known_users = data.x[data.y == 1,  0]
            select_index = np.random.randint(0, self.user_size)
            while select_index in known_users or select_index not in self.user2idx:
                select_index = np.random.randint(0, self.user_size)
            negative_node = self.user2idx[select_index]
        else:
            select_index = negative_node_id[torch.randint(0, len(negative_node_id), (1,))]
            negative_node = self.user2idx[int(data.x[select_index[0]][0][0])]

        return torch.from_numpy(known_nodes).long(), \
                torch.from_numpy(np.array([positive_node])).long(),\
                torch.from_numpy(np.array([negative_node])).long(), group_node_id