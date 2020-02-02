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
from tqdm import tqdm
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


def sample_node_neighbour_hops(G, src, cutoff=2): 
    # sample the same node type neighbour
    valid_node = []
    src_type = G.nodes[src]['type']
    neigbour = nx.single_source_shortest_path(G, src, cutoff=cutoff)
    for n in list(neigbour):
        if G.nodes[n]['type'] == src_type and n != src:
            valid_node.append(n)
    return valid_node

def create_sample_walk(G, src, total_walk=100):
    neighbours = sample_node_neighbour_hops(G, src)
    for cutoff in range(3, 6):
        # some node require more hops to reach same node type
        neighbours = sample_node_neighbour_hops(G, src, cutoff=cutoff)
    random.shuffle(neighbours)
    neighbours = neighbours[:total_walk]
    src_id = G.nodes[src]['id']
    target, context = [], []
    for n in neighbours:
        target.append(src_id)
        context.append(G.nodes[n]['id'])
    return np.array(target), np.array(context)

def sample_negative(total_size, inputs, num):
    negative = []
    for t in inputs:
        neg = np.random.randint(0, total_size, num)
        if t in neg:
            neg = np.random.randint(0, total_size, num)
        negative.append(neg)
    return np.array(negative)



def generate_batch(G, batch_size, node_type, embedding_size, neg_num=2):
    batches = []
    batch_cnt = 0
    batch_labels, batch_inputs, batch_negative_samples = [], [], []

    # pool = mp.Pool()
    results = []
    for cnt in range(100):
        similar_nodes = extract_node_type(G, node_type)
        for n in similar_nodes:
            target, context = create_sample_walk(G, n, total_walk=100)
            batch_cnt += len(context)
            if len(context) > 0:
                batch_labels.append(target)
                batch_inputs.append(context)
                negative_samples = sample_negative(embedding_size, target, neg_num)
                batch_negative_samples.append(negative_samples)

            if batch_cnt >= batch_size:
                break
        if batch_cnt >= batch_size:
            break
        # prevent infinite loop
    if len(batch_labels) != 0 and len(batch_inputs) != 0:
        batch_labels = np.concatenate(batch_labels)
        batch_inputs = np.concatenate(batch_inputs)
        batch_negative_samples = np.concatenate(batch_negative_samples)
        shuffle_size = np.min([batch_size, len(batch_inputs), len(batch_labels), len(batch_negative_samples)])
        shuffle_idx = list(range(shuffle_size))
        random.shuffle(shuffle_idx)
        return batch_inputs[shuffle_idx], batch_labels[shuffle_idx], batch_negative_samples[shuffle_idx]
    return [], [], []


def sample_pairs(dataset, neg_num, batch_size, node_type, embed_size):
    sub_G = data_to_networkx_(dataset)
    inputs, labels, negative = generate_batch(sub_G, batch_size, 
        node_type, embed_size, neg_num=neg_num)
    return labels, inputs, negative


def sample_walks(train_datasets, neg_num, batch_size, node_type, embed_size, cpu_count=None):
    pool = mp.Pool(processes=cpu_count)
    results = []
    samples = []
    for idx in range(len(train_datasets)):
        dataset = train_datasets[idx]
        res = pool.apply_async(sample_pairs ,(dataset, neg_num, batch_size, node_type, embed_size))
        results.append(res)

    for r in results:
        try:
            inputs, labels, neg_ = r.get(timeout=1800)
            if len(inputs)> 0:
                samples.append((inputs, labels, neg_))
        except mp.TimeoutError:
            continue
    # except KeyboardInterrupt:
    #     pool.close()
    #     raise ValueError('exit')
    # except:        
    #     pool.close()
    #     return samples

    pool.close()

    return samples

class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGramNeg, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        # self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.log_sigmoid = nn.LogSigmoid()

        self.input_emb.reset_parameters()
        # initrange = (2.0 / (vocab_size + emb_dim)) ** 0.5  # Xavier init
        # self.input_emb.weight.data.uniform_(-initrange, initrange)



    def forward(self, target_input, context, neg):
        """
        :param target_input: [batch_size]
        :param context: [batch_size]
        :param neg: [batch_size, neg_size]
        :return:
        """
        # u,v: [batch_size, emb_dim]
        v = self.input_emb(target_input)
        u = self.input_emb(context)
        # positive_val: [batch_size]
        positive_val = self.log_sigmoid(torch.sum(u * v, dim=1)).squeeze()

        # u_hat: [batch_size, neg_size, emb_dim]
        u_hat = self.input_emb(neg)
        # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]
        # neg_vals: [batch_size, neg_size]
        neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze(2)
        # neg_val: [batch_size]
        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.input_emb(inputs)

