import os
import os.path as osp
from copy import deepcopy
import random
import pickle
from collections import defaultdict
from itertools import islice
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import networkx as nx
from sklearn.metrics import f1_score
import torch
from torch.autograd import Variable
from torch_geometric.data import Dataset, Data, DataLoader
import pandas as pd
import json

def parse_movie_json():
    


def build_movielens_graph(filename='movie_metadata_v2.jsonl'):
    movies = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            movie = json.loads(line)



class MovieLens(Dataset):
    def __init__(self, cutoff=2, ratio=0.8, min_size=5,
                 max_size=500):
        self.dataset = dataset
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size
        self.group_size = 0
        self.user2id = None
        user2id_filepath = osp.join(dataset_path, "user2id.pkl")
        group_size_filepath = osp.join(dataset_path, "group_size.pkl")
        if os.path.exists(user2id_filepath):
            with open(user2id_filepath, 'rb') as f:
                user2id = pickle.load(f)
            with open(group_size_filepath, 'rb') as f:
                self.group_size = pickle.load(f)
        else:
            user2id = defaultdict(int)
            with open(self.dataset_filepath, 'r') as f:
                for line in f.readlines():
                    if '#' not in line and len(line) > 0:
                        members = line.strip().split('\t')
                        if (len(members) >= self.min_size and
                                len(members) <= max_size):
                            self.group_size += 1

            ungraph_filepath = osp.join(
                dataset_path, "com-{}.ungraph.txt".format(self.dataset))
            with open(ungraph_filepath, 'r') as f:
                while True:
                    try:
                        line = f.readline()

                    except StopIteration:
                        break
                    if len(line) == 0:
                        break
                    if '#' in line or len(line) < 2:
                        continue
                    members = line.strip().split('\t')
                    for m in members:
                        if str(m) not in user2id:
                            user2id[str(m)] = len(user2id)

            with open(user2id_filepath, 'wb') as f:
                pickle.dump(user2id, f)
            with open(group_size_filepath, 'wb') as f:
                pickle.dump(self.group_size, f)
        self.user2id = user2id

        print('group size : ', self.group_size)
        print('total user : ', len(self.user2id))
        self.processed_file_idx = [idx for idx in range(self.group_size)]

        super(MovieLens, self).__init__(osp.join("processed", dataset),
                                            transform=None,
                                            pre_transform=None)


    @property
    def processed_file_names(self):
        return [
            '{}_{}_{}_{}_{}_v2.pt'.format(
                self.dataset, self.cutoff, self.ratio, self.min_size, idx)
            for idx in self.processed_file_idx
        ]

    def _download(self):
        pass

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    def process(self):