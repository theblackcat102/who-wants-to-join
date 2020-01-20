import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
from copy import deepcopy

from torch_geometric.data import InMemoryDataset, Dataset, Data, DataLoader
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import f1_score
import pickle
import os.path as osp
import os
import torch
from torch.autograd import Variable, Function
import random

def graph2data(G, embeddings):
    matrix = embeddings['embedding']
    name2id = embeddings['name2id']
    matrix_dim = len(matrix[0])
    graph_idx = {}

    # does the order of index matter?
    # seems to me it's relative to one and another?
    for n in G.nodes:
        graph_idx[n] = len(graph_idx)
    nodes = []
    edges = []
    labels = []
    for n in G.nodes:
        node_latent = None
        if str(n) in name2id:
            node_latent = Variable(torch.from_numpy(np.array([name2id[str(n)]])))
        else:
            print(str(n))
            continue

        edge_index = np.array(list( G.edges(n)))
        new_edges = []
        for idx in range(len(edge_index)):
            src, dst = edge_index[idx]
            edge_index[idx] = [graph_idx[src], graph_idx[dst]  ]
            new_edges.append([graph_idx[dst], graph_idx[src]  ])
        edges.append(new_edges)
        nodes.append(node_latent)
        labels.append(G.nodes[n]['predict'])
    if len(nodes) == 0:
        raise ValueError('Invalid graph node')
    x = torch.stack(nodes)
    y = torch.from_numpy(np.array(labels))
    edges = torch.from_numpy(np.transpose(np.concatenate(edges))).contiguous()
    return Data(x=x, edge_index=edges, y=y)

def create_sub_graph(G, group2member, exist_ratio=0.8, cutoff=2, min_size=2,max_size=1000):
    hit_rate = []
    idx = 0
    groups = []
    for group_id, members in tqdm(group2member.items(), dynamic_ncols=True):
        if len(members) < min_size or len(members) > max_size:
            continue

        random.shuffle(members)        
        ratio = int(len(members)*exist_ratio)
        predict_ratio = len(members) - ratio
        # make sure there's at least 2 group member to predict
        if predict_ratio < 2:
            predict_ratio = 2
            ratio = len(members) - predict_ratio

        exist_nodes = members[:ratio]
    
        sub_graph_nodes = []
        for node in exist_nodes:
            n_nodes = nx.single_source_shortest_path_length(G, node, cutoff=cutoff)
            sub_graph_nodes += [n for n in n_nodes]
        sub_graph_nodes = set(sub_graph_nodes)
        sub_G = nx.Graph()
        in_group_cnt = 0
        for nodes in sub_graph_nodes:
            in_group = 1 if nodes in members else 0
            predict = 0
            if nodes in exist_nodes:
                predict = 0
            elif nodes in members and nodes not in exist_nodes:
                predict = 1
            in_group_cnt += in_group
            sub_G.add_node(nodes, in_group=in_group, predict=predict)
        hit_rate.append(in_group_cnt/len(members))
        # print('total : ',in_group_cnt)
        # print(len(sub_G.nodes))
        for nodes in sub_graph_nodes:
            for n in G.neighbors(nodes):
                if sub_G.has_node(nodes) and sub_G.has_node(n):
                    sub_G.add_edge(nodes, int(n))
        yield sub_G
    #     groups.append(sub_G)
    #     idx += 1
    #     # if idx > 1000:
    #     #     break
    # return groups

class SNAPCommunity(Dataset):

    def __init__(self, dataset='amazon', cutoff=2, ratio=0.8, min_size=5, max_size=500):
        self.dataset = dataset
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size
        self.group_size = 0
        with open('{}/com-{}.all.cmty.txt'.format(self.dataset, self.dataset), 'r') as f:
            for line in f.readlines():
                if '#' not in line and len(line) > 0:
                    members = line.strip().split('\t')
                    if len(members) >= self.min_size and len(members) <= max_size:
                        self.group_size += 1
        print('group size : ', self.group_size)
        self.processed_file_idx = [ idx for idx in range(self.group_size)]

        self.user_map = None
        embedding_filename = 'graphv/{}-64-DeepWalk.pkl'.format(self.dataset)
        if os.path.exists(embedding_filename):
            with open(embedding_filename, 'rb') as f:
                embeddings = pickle.load(f)
                self.user_map = embeddings['name2id']

        super(SNAPCommunity, self).__init__('./', transform=None, pre_transform=None)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [
            '{}_{}_{}_{}_{}.pt'.format(self.dataset, self.cutoff, self.ratio, self.min_size, idx)
            for idx in self.processed_file_idx
        ]
    
    def _download(self):
        pass

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    def process(self):

        all_found = True
        length = 0
        for idx in range(self.group_size):
            filename = '{}_{}_{}_{}_{}.pt'.format(self.dataset, self.cutoff, self.ratio, self.min_size, idx)
            if not os.path.exists(osp.join(self.processed_dir, filename)):
                print(filename)
                all_found = False
                length = idx
                break
        if length != 0:
            self.group_size = length
            self.processed_file_idx = [ idx for idx in range(self.group_size)]
            return

        member2group = defaultdict(list)
        group2member = defaultdict(list)
        edges = defaultdict(int)
        embedding_filename = 'graphv/{}-64-DeepWalk.pkl'.format(self.dataset)
        if os.path.exists(embedding_filename):
            with open(embedding_filename, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            embeddings = {}
            embeddings['name2id'] = defaultdict(int)
        print('found pretrain embeddings...')
        # create member edge
        with open('{}/com-{}.ungraph.txt'.format(self.dataset, self.dataset), 'r') as f, tqdm() as pbar:
            while True:
                pbar.update(1)
                try:
                    line = f.readline()

                except StopIteration:
                    break
                if len(line) == 0:
                    break
                if '#' in line or len(line) < 2:
                    continue
                edge = [ str(int(m)) for m in line.strip().split('\t')]
                edge = '_'.join(edge)
                edges[edge] = 1
        print('load community graph')
        # load directed graph
        with open('{}/com-{}.all.cmty.txt'.format(self.dataset, self.dataset), 'r') as f,tqdm() as pbar: 
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
                members = [ int(m) for m in members]
                if len(members) < self.min_size or len(members) > self.max_size:
                    continue
                group_id = len(group2member)
                group2member[group_id] = members
                for m in members:
                    member2group[m].append(group_id)
                    if m not in embeddings['name2id']:
                        print(m)
                        embeddings['name2id'][str(m)] = len(embeddings['name2id'])
                pbar.update(1)
                if group_id > 500000:
                    break


        print('initialize networks...')
        G = nx.Graph()
        for edge in edges:
            src, dst = edge.split('_')
            src, dst = int(src), int(dst)
            
            if not G.has_node(src):
                G.add_node(src, group=member2group[src])

            if not G.has_node(dst):
                G.add_node(dst, group=member2group[dst])
            
            if G.has_node(src) and G.has_node(dst):
                G.add_edge(src, dst)
            if str(src) not in embeddings['name2id']:
                embeddings['name2id'][str(src)] = len(embeddings['name2id'])
            if str(dst) not in embeddings['name2id']:
                embeddings['name2id'][str(dst)] = len(embeddings['name2id'])
            

        print('populate sub graph')
        with open(embedding_filename, 'wb') as f:
            pickle.dump(embeddings, f)
        # sub_graphs = create_sub_graph(G, group2member, exist_ratio=self.ratio, 
        #     cutoff=self.cutoff, min_size=self.min_size)
        idx = 0
        for sub in create_sub_graph(G, group2member, exist_ratio=self.ratio, 
            cutoff=self.cutoff, min_size=self.min_size):
            if len(sub.nodes) == 0:
                continue
            data = graph2data(sub, embeddings)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            filename = '{}_{}_{}_{}_{}.pt'.format(self.dataset, self.cutoff, self.ratio, self.min_size, idx)
            torch.save(data, osp.join(self.processed_dir, filename))
            idx += 1
        print('Total {}'.format(idx))
        # self.data = [ graph2data(sub, embeddings) for sub in sub_graphs ]
        # data, slices = self.collate(self.data)
        # torch.save((data, slices), self.processed_paths[0])

    def __len__(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if isinstance(idx, list):
            self.processed_file_idx = idx
            return deepcopy(self)

        filename = '{}_{}_{}_{}_{}.pt'.format(self.dataset, self.cutoff, self.ratio, self.min_size, idx)
        data = torch.load(osp.join(self.processed_dir, filename))
        return data

if __name__ == "__main__":
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    import torch.nn as nn
    layer = GCNConv(64, 1)
    dataset = SNAPCommunity('friendster', cutoff=2, ratio=0.8)
    # dataset[:540]
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embeddings = nn.Embedding(334863, 16)
            self.conv1 = GCNConv(16, 16)
            self.conv2 = GCNConv(16, 16)
            self.conv3 = GCNConv(16, 16)
            self.conv4 = GCNConv(16, 1)

        def forward(self, x, edge_index):
            x = self.embeddings(x.squeeze(-1))
            # x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x,p=0.1, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x,p=0.1, training=self.training)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x,p=0.1, training=self.training)
            x = self.conv4(x, edge_index)

            return x
    model = Net()
    model = model.cuda()
    model.train()

    print(len(dataset))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    pos_weight = torch.ones([1])*10
    pos_weight = pos_weight.cuda()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, )

    for epoch in range(200):
        print(epoch)
        predictions = []
        targets = []
        with tqdm(total=len(loader)) as pbar:
            for data in loader:

                optimizer.zero_grad()
                # data = data.cuda()
                x, edge_index = data.x, data.edge_index
                x = x.cuda()
                edge_index = edge_index.cuda()
                label = data.y.unsqueeze(-1).cuda().float()
                output = model(x, edge_index)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                targets.append(data.y.cpu().detach().numpy())
                predictions.append( output.cpu().detach().numpy() )
                # break
                pbar.update(1)
                pbar.set_description("loss {:.4f}".format(loss.item()))

        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions) > 0.5 
        print(np.sum(predictions), np.sum(targets))
        score = f1_score(targets, predictions, average="micro")
        baselines = np.zeros(targets.shape)
        baseline_score = f1_score(targets, baselines, average='micro')
        print("\nF-1 score: {:.4f}, {:.4f}".format(score, baseline_score))
