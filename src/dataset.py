from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
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
        node_latent = torch.rand(matrix_dim, requires_grad=True)
        if str(n) in name2id:
            node_latent = Variable(torch.from_numpy(matrix[name2id[str(n)]]))
        edge_index = np.array(list( G.edges(n)))
        new_edges = []
        for idx in range(len(edge_index)):
            src, dst = edge_index[idx]
            edge_index[idx] = [graph_idx[src], graph_idx[dst]  ]
            new_edges.append([graph_idx[dst], graph_idx[src]  ])
        edges.append(new_edges)
        nodes.append(node_latent.unsqueeze(0))
        labels.append(G.nodes[n]['predict'])
    x = torch.stack(nodes)
    y = torch.from_numpy(np.array(labels))
    edges = torch.from_numpy(np.transpose(np.concatenate(edges)))
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
        groups.append(sub_G)
        idx += 1
        # if idx > 1000:
        #     break
    return groups

class SNAPCommunity(InMemoryDataset):

    def __init__(self, dataset='amazon', cutoff=2, ratio=0.8, min_size=2):
        self.dataset = dataset
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        super(SNAPCommunity, self).__init__('./', transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset+str(self.cutoff)+'_'+str(self.ratio)+str(self.min_size)+'.pt']
    
    def download(self):
        print('download')

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    def process(self):
        member2group = defaultdict(list)
        group2member = defaultdict(list)
        edges = defaultdict(int)
        filename = 'graphv/{}-64-DeepWalk.pkl'.format(self.dataset)
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
        print('found pretrain embeddings...')
        # create member edge
        with open('{}/com-{}.ungraph.txt'.format(self.dataset, self.dataset), 'r') as f:
            for line in tqdm(f.readlines()):
                edge = [ str(int(m)) for m in line.split('\t')]
                edge = '_'.join(edge)
                edges[edge] = 1

        # load directed graph
        with open('{}/com-{}.all.cmty.txt'.format(self.dataset, self.dataset), 'r') as f: 
            for line in tqdm(f.readlines()):
                members = line.split('\t')
                members = [ int(m) for m in members]
                group_id = len(group2member)
                group2member[group_id] = members
                for m in members:
                    member2group[m].append(group_id)

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

        print('populate sub graph')

        sub_graphs = create_sub_graph(G, group2member, exist_ratio=self.ratio, 
            cutoff=self.cutoff, min_size=self.min_size)
        self.data = [ graph2data(sub, embeddings) for sub in sub_graphs ]
        data, slices = self.collate(self.data)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    from torch_geometric.nn import GCNConv
    layer = GCNConv(64, 32)
    dataset = SNAPCommunity('amazon')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for data in loader:
        x, edge_index = data.x, data.edge_index
        output = layer(x, edge_index)
        print(output.shape)
        break