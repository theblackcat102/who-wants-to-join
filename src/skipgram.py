from torch_geometric.utils.convert import to_networkx
from src.meetup import Meetup
import networkx as nx
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp

def data_to_networkx_(data):
    G = nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))

    values = {key: data[key].squeeze().tolist() for key in data.keys}

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
    for x, attribute in enumerate(data.x):
        G.nodes[x]['id'] = attribute[0].data.item()
        G.nodes[x]['pred'] = attribute[1].data.item()
        G.nodes[x]['type'] = attribute[2].data.item()
    return G

def extract_node_type(G, type_):
    nodes = []
    for n in G.nodes:
        if G.nodes[n]['type'] == type_:
            nodes.append(n)
    return nodes


def sample_node_neighbour_hops(G, src, cutoff=2): # sample the same node type neighbour
    valid_node = []
    src_type = G.nodes[src]['type']
    neigbour = nx.single_source_shortest_path(G, src, cutoff=cutoff)
    for n in list(neigbour):
        if G.nodes[n]['type'] == src_type and n != src:
            valid_node.append(n)
    return valid_node

def create_sample_walk(G, src, total_walk=100):
    neighbours = sample_node_neighbour_hops(G, src)    
    labels, inputs = [], []
    random.shuffle(neighbours)
    neighbours = neighbours[:total_walk]
    src_id = G.nodes[src]['id']
    for n in neighbours:
        labels.append(src_id)
        inputs.append(G.nodes[n]['id'])
    return np.array(labels), np.array(inputs)

def sample_negative(G, total_size, inputs, num):
    negative = []
    population = list(range(total_size))
    for t in inputs:
        neg = np.random.randint(0, population, num)
        if t in neg:
            neg = np.random.randint(0, population, num)
        negative.append(neg)
    return np.array(negative)



def generate_batch(G, batch_size, node_type, embedding_size, neg_num=2):
    batches = []
    batch_cnt = 0
    batch_labels, batch_inputs, batch_negative_samples = [], [], []

    # pool = mp.Pool()
    results = []
    while batch_cnt < batch_size:
        similar_nodes = extract_node_type(G, node_type)
        for n in similar_nodes:
            # res = pool.apply_async(sample_pairs, (G, n, embedding_size, neg_num))
            # results.append(res)
            labels, inputs = create_sample_walk(G, n, total_walk=5)
            negative_samples = sample_negative(G, embedding_size, inputs, neg_num)
            batch_cnt += len(labels)

            batch_labels.append(labels)
            batch_inputs.append(inputs)

            if negative_samples.shape[-1] == neg_num:
                batch_negative_samples.append(negative_samples)
            if batch_cnt >= batch_size:
                break
    batch_labels = np.concatenate(batch_labels).flatten()
    batch_inputs = np.concatenate(batch_inputs).flatten()
    batch_negative_samples = np.concatenate(batch_negative_samples)

    shuffle_idx = list(range(batch_size))
    random.shuffle(shuffle_idx)

    return batch_inputs[shuffle_idx], batch_labels[shuffle_idx], batch_negative_samples[shuffle_idx]


def sample_pairs(dataset, neg_num, batch_size, node_type, embed_size):
    sub_G = data_to_networkx_(dataset)
    inputs, labels, negative = generate_batch(sub_G, batch_size, 
        node_type, embed_size, neg_num=neg_num)
    return labels, inputs, negative


def sample_walks(train_datasets, neg_num, batch_size, node_type, embed_size):
    pool = mp.Pool()
    results = []
    samples = []
    for dataset in train_datasets:
        res = pool.apply_async(sample_pairs ,(dataset, neg_num, batch_size, node_type, embed_size))
        results.append(res)
    for r in results:
        samples.append(r.get())
    pool.close()
    return samples

class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGramNeg, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.log_sigmoid = nn.LogSigmoid()

        initrange = (2.0 / (vocab_size + emb_dim)) ** 0.5  # Xavier init
        self.input_emb.weight.data.uniform_(-initrange, initrange)
        self.output_emb.weight.data.uniform_(-0, 0)


    def forward(self, target_input, context, neg):
        """
        :param target_input: [batch_size]
        :param context: [batch_size]
        :param neg: [batch_size, neg_size]
        :return:
        """
        # u,v: [batch_size, emb_dim]
        v = self.input_emb(target_input)
        u = self.output_emb(context)
        # positive_val: [batch_size]
        positive_val = self.log_sigmoid(torch.sum(u * v, dim=1)).squeeze()

        # u_hat: [batch_size, neg_size, emb_dim]
        u_hat = self.output_emb(neg)
        # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]
        # neg_vals: [batch_size, neg_size]
        neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze(2)
        # neg_val: [batch_size]
        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.input_emb(inputs)

