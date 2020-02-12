import os.path as osp
from copy import deepcopy
import pickle
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
from torch.autograd import Variable
from torch_geometric.data import Dataset, Data
from src.dataset import split_group
import random
import os, glob
import gzip
import pandas as pd
from time import sleep

def init_dblp():
    author2id = defaultdict(int)
    paper2id = defaultdict(int)
    conf2id = defaultdict(int)
    member2freq = defaultdict(int)
    index2title = defaultdict(int)
    papers = []
    data = {}

    citation_graph = nx.Graph()
    valid_index = []
    cnt = 0
    with open('aminer/dblp.txt', 'r') as f:
        for line in tqdm(f.readlines()):
            if '#' in line:
                if '*' == line[1]:
                    if len(data) > 4 and 'authors' in data and len(data['authors']) >= 5 and len(data['authors']) <= 100:
                        index2title[data['index']] = data['title']
                        valid_index.append(data['index'])
                        papers.append(data)
                    data = {}
                    title = line.strip()[2:]
                    if title not in paper2id:
                        paper2id[title] = len(paper2id)
                    data['title'] = title
                elif '@' == line[1]:
                    author_line = line.strip()[2:]
                    authors = [a.strip() for a in author_line.split(',')]
                    if len(authors) < 5 or len(authors) > 100:
                        continue
                    for a in authors:
                        member2freq[a] += 1
                        if a not in author2id:
                            author2id[a] = len(author2id)
                    data['authors'] = authors
                elif 'c' == line[1]:
                    conf = line.strip()[2:]
                    if conf not in conf2id:
                        conf2id[conf] = len(conf2id)
                    data['conf'] = conf
                elif 't' == line[1]:
                    year = line.strip()[2:]
                    data['year'] = int(year)
                elif 'index' == line[1:6]:
                    data['index'] = line[7:].strip()
                elif '%' == line[1]:
                    index_num = line[2:].strip()
                    if 'index' in data:
                        if not citation_graph.has_node(data['index']):
                            citation_graph.add_node(data['index'])
                        if not citation_graph.has_node(index_num):
                            citation_graph.add_node(index_num)
                        citation_graph.add_edge(data['index'], index_num)

    if len(data) > 4 and 'authors' in data and len(data['authors']) >= 3:
        papers.append(data)


    print(len(papers), ' found')
    valid_index = set(valid_index)
    for n in tqdm(list(citation_graph.nodes)):
        if n not in valid_index:
            citation_graph.remove_node(n)

    paper2authors = defaultdict(list)
    for p in papers:
        paper2authors[paper2id[p['title']]] = [author2id[a] for a in p['authors'] ]


    with open('aminer/preprocess_dblp.pkl', 'wb') as f:
        pickle.dump({'papers': papers,
            'author2id': author2id,
            'paper2id': paper2id,
            'conf2id': conf2id,
            'citation_graph': citation_graph,
            'paper2authors': paper2authors,
            'index2title': index2title,
        }, f)


def init_graph(papers, paper2id, conf2id, author2id, citation_graph, index2title):
    G = nx.Graph()
    for paper in tqdm(papers):
        p = deepcopy(paper)

        p_id = 'p'+str(paper2id[p['title']])
        if not G.has_node(p_id):
            G.add_node(p_id)
        if citation_graph.has_node(p['index']):
            for n in citation_graph.neighbors(p['index']):
                neighbour_title = index2title[n]
                id_ = paper2id[neighbour_title]
                n_p_id = 'p'+str(id_)                
                if not G.has_node(n_p_id):
                    G.add_node(n_p_id)
                G.add_edge(p_id, n_p_id)


        for a in p['authors']:
            a_id = 'a'+str(author2id[a])

            if not G.has_node(a_id):
                G.add_node(a_id)
            G.add_edge(p_id, a_id)


        if 'conf' in p:
            c_id = 'c'+str(conf2id[p['conf']])

            if not G.has_node(c_id):
                G.add_node(c_id)

            G.add_edge(c_id, p_id)
    return G

def init_graph_baseline(papers, author2id):
    G = nx.Graph()
    for p in tqdm(papers):
        for a in p['authors']:
            a_id = 'a'+str(author2id[a])
            if not G.has_node(a_id):
                G.add_node(a_id)

            for c in p['authors'][1:]:
                c_id = 'a'+str(author2id[c])
                if not G.has_node(c_id):
                    G.add_node(c_id)

                if c_id != a_id:
                    G.add_edge(c_id, a_id)

    return G

def append_paper_graph(H, paper, paper2id, conf2id, author2id, citation_graph, index2title):
    # paper = deepcopy(p)
    p_id = 'p'+str(paper2id[paper['title']])
    c_id = 'c'+str(conf2id[paper['conf']])
    if not H.has_node(p_id):
        H.add_node(p_id)
    if not H.has_node(c_id):
        H.add_node(c_id)
    H.add_edge(c_id, p_id)

    for a in paper['authors']:
        a_id = 'a'+str(author2id[a])
        if not H.has_node(a_id):
            H.add_node(a_id)
            H.add_edge(p_id, a_id)

        if citation_graph.has_node(paper['index']):
            for n in citation_graph.neighbors(paper['index']):
                neighbour_title = index2title[n]
                id_ = paper2id[neighbour_title]
                n_p_id = 'p'+str(id_)
                if not H.has_node(n_p_id):
                    H.add_node(n_p_id)
                H.add_edge(p_id, n_p_id)

    return H

def append_paper_graph_baseline(H, paper, author2id):
    for a in paper['authors']:
        a_id = 'a'+str(author2id[a])
        for c in paper['authors'][1:]:
            c_id = 'a'+str(author2id[c])
            if not H.has_node(c_id):
                H.add_node(c_id)
            if c_id != a_id:
                H.add_edge(c_id, a_id)
    return H


def get_neighbour_nodes(G, node_id, cutoff=2, level=0):
    '''
    test = nx.Graph()
    for node_id in range(10):
        test.add_node(node_id)

    test.add_edge(1,2);test.add_edge(0,1)
    test.add_edge(1,3);test.add_edge(2,4)
    test.add_edge(4,6);test.add_edge(2,5)
    test.add_edge(5,7);test.add_edge(7,9)
    '''
    if cutoff == level:
        return []
    results = []
    for n in G.neighbors(node_id):
        results += get_neighbour_nodes(G, n, cutoff, level+1)
        results.append(n)
    return results

def create_subgraph(paper, G, exist_ratio=0.8, cutoff=3):

    sub_G = nx.Graph()

    authors_ = paper['authors']
    random.shuffle(authors_)
    ratio_ = int(len(authors_)*exist_ratio)
    predict_ratio = len(authors_) - ratio_
    # make sure there's at least 2 group member to predict
    if predict_ratio < 2:
        predict_ratio = 2
        ratio_ = len(authors_) - predict_ratio

    authors = [ 'a'+str(n) for n in authors_ ]
    sub_graph_nodes = []
    predict_nodes = []
    exist_nodes = []

    for a in authors:
        if G.has_node(a):
            if len(predict_nodes) < predict_ratio:
                predict_nodes.append(a)
            else:
                exist_nodes.append(a)

    if len(predict_nodes) == 0 or len(exist_nodes) == 0:
        return None, 0, predict_ratio

    # print(predict_nodes)
    # print(exist_nodes)
    for start_node in authors:
        if G.has_node(start_node):
            n_nodes = nx.single_source_shortest_path_length(G, start_node, cutoff=cutoff)
            sub_graph_nodes += list(n_nodes)
            sub_graph_nodes.append(start_node)

    sub_graph_nodes = set(sub_graph_nodes)
    # print(sub_graph_nodes)
    in_group_cnt = 0
    hit = 0
    exist_nodes = set(exist_nodes)
    authors = set(authors)
    predict_nodes = set(predict_nodes)

    for node in sub_graph_nodes:
        in_group = 1 if node in authors else 0
        known_member = 1 if node in exist_nodes else 0
        predict = 0
        if node in predict_nodes:
            predict = 1
            hit += 1
        in_group_cnt += in_group
        basic_attributes = {}
        basic_attributes['type'] = node[0]
        basic_attributes['id'] = int(node[1:])
        # within known group
        basic_attributes['in_group'] = known_member
        basic_attributes['predict'] = predict
        basic_attributes['known'] = known_member
        sub_G.add_node(node, **basic_attributes)

    basic_attributes = {}
    basic_attributes['type'] = 'p'
    basic_attributes['id'] = paper['title']
    basic_attributes['in_group'] = 1
    basic_attributes['predict'] = 0
    basic_attributes['known'] = 0
    sub_G.add_node('p'+str(paper['title']), **basic_attributes)

    basic_attributes = {}
    basic_attributes['type'] = 'c'
    basic_attributes['id'] = paper['conf']
    basic_attributes['in_group'] = 1
    basic_attributes['predict'] = 0
    basic_attributes['known'] = 0
    sub_G.add_node('c'+str(paper['conf']), **basic_attributes)

    for node in exist_nodes:
        sub_G.add_edge(node, 'c'+str(paper['conf']))
        sub_G.add_edge(node, 'p'+str(paper['title']))

    for node in sub_graph_nodes:
        for n in G.neighbors(node):
            if node == paper['title'] and n in predict_nodes:
                continue
            if n == paper['title'] and node in predict_nodes:
                continue

            if sub_G.has_node(node) and sub_G.has_node(n):
                sub_G.add_edge(node, n)

    return sub_G, hit, predict_ratio

def graph2data(G, titleid):
    '''
        [node_id, known/to predict : 0/1,
         node_type : [member: 0, topic: 1, category : 2, event: 3, group: 4]]
    '''
    graph_idx = {}

    # does the order of sub-graph index matter?
    # seems to me it's relative to one and another?


    nodes = []
    edges = []
    labels = []
    inputs = []
    loss_mask = []

    for n in G.nodes:
        node_type = 0
        node_attributes = G.nodes[n]
        if 'in_group' in node_attributes:
            inputs.append(node_attributes['in_group'])
        else:
            inputs.append(0)
        if node_attributes['type'] == 'a':
            node_type = 0
        elif node_attributes['type'] == 'p':
            node_type = 1
        elif node_attributes['type'] == 'c':
            node_type = 2

        node_latent = Variable(
            torch.from_numpy(
                np.array([node_attributes['id'], node_attributes['known'], node_type])))


        nodes.append(node_latent)
        if node_type == 0:
            loss_mask.append(1)
        else:
            loss_mask.append(0)

        labels.append(node_attributes['predict'])

    nodes_id = np.array([n for n in G.nodes])
    inputs = np.array(inputs)
    resort_idx = np.argsort(-inputs)
    nodes_id = nodes_id[ resort_idx ]
    x = torch.stack(nodes)[resort_idx]
    y = torch.from_numpy(np.array(labels))[resort_idx]
    input_mask = torch.from_numpy(inputs)[resort_idx]
    loss_mask = torch.from_numpy(np.array(loss_mask))[resort_idx]

    for n in nodes_id:
        graph_idx[n] = len(graph_idx)

    for n in nodes_id:
        edge_index = list(G.edges(n))
        new_edges = []
        for idx in range(len(edge_index)):
            src, dst = edge_index[idx]
            new_edges.append([graph_idx[src], graph_idx[dst]])

            # bidirected with authors
            if G.nodes[src]['type'] == 'a' and G.nodes[dst]['type'] == 'a':
                new_edges.append([graph_idx[dst], graph_idx[src]])
        edges.append(new_edges)

    x = torch.stack(nodes)
    y = torch.from_numpy(np.array(labels))
    input_mask = torch.from_numpy(np.array(inputs))
    loss_mask = torch.from_numpy(np.array(loss_mask))
    edges = torch.from_numpy(np.transpose(np.concatenate(edges))).contiguous()

    data = Data(x=x, edge_index=edges, y=y, label_mask=loss_mask, input_mask=input_mask,
        titleid=torch.from_numpy(np.array([titleid])))
    # add output mask to mask additional nodes : category, venue, topic node
    return data


def async_postprocessing(paper, H, idx, processed_dir, cache_file_prefix, cutoff=2):
    sub_G, hit, pred_cnt = create_subgraph(paper, H, cutoff=cutoff)
    if sub_G is None:
        return None
    if idx < 100: # debug purpose make sure sub_G nodes number differ each iteration
        print(len(sub_G.nodes), hit, pred_cnt)
    data = graph2data(sub_G, paper['title'])
    filename = cache_file_prefix+'_{}_v2.pt'.format(idx)
    torch.save(data, osp.join(processed_dir, filename))
    return None


def async_postprocessing_job(papers, Hs, idx, processed_dir, cache_file_prefix, pbar_queue, cutoff=2):
    idx_ = idx
    for p, H in zip(papers, Hs):
        sub_G, hit, pred_cnt = create_subgraph(p, H, cutoff=cutoff)
        if sub_G is None:
            return None
        if idx_ < 100: # debug purpose make sure sub_G nodes number differ each iteration
            print(len(sub_G.nodes), hit, pred_cnt)
        data = graph2data(sub_G, p['title'])
        filename = cache_file_prefix+'_{}_v2.pt'.format(idx_)
        torch.save(data, osp.join(processed_dir, filename))
        idx_ += 1
    pbar_queue.put(len(papers))
    return None


class Aminer(Dataset):

    def __init__(self, cutoff=2, ratio=0.8, min_size=5, max_size=100,
                 city_id=10001, baseline=False, transform=None):
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size
        self.baseline = baseline
        if not os.path.exists('aminer/preprocess_dblp.pkl'):
            init_dblp()
        self.data_folder = 'dblp_v1'
        if baseline:
            self.data_folder = 'dblp_hete_base'
        self.cache_file_prefix = '{}_{}_{}_{}_3'.format(
            'dblp', self.cutoff, self.ratio, self.min_size)
        temp = osp.join(osp.join("processed", self.data_folder), 'processed')
        match_filename = self.cache_file_prefix+'_*_v2.pt'
        self.processed_file_idx = list(glob.glob(osp.join(temp, match_filename)))

        os.makedirs(osp.join(osp.join("processed", self.data_folder), 'processed'), exist_ok=True)
        super(Aminer, self).__init__(osp.join("processed", self.data_folder),
                                     transform=transform,
                                     pre_transform=None)
        self.process()

    @property
    def processed_file_names(self):
        return self.processed_file_idx

    def _download(self):
        pass

    @property
    def raw_file_names(self):
        return ['some_file_1']

    def process(self):
        length = 0
        print(self.processed_dir)
        match_filename = self.cache_file_prefix+'_*_v2.pt'
        length = len(list(glob.glob(osp.join(self.processed_dir, match_filename))))
        print('length: {}'.format(length))
        if length != 0:
            print('update')
            self.group_size = length
            self.processed_file_idx = list(glob.glob(osp.join(self.processed_dir, match_filename)))
            return
        self.init_preprocessing()
        self.list_of_data = list(glob.glob(osp.join(self.processed_dir, match_filename)))


    def __len__(self):
        return len(self.processed_file_idx)

    def init_preprocessing(self):
        with open('aminer/preprocess_dblp.pkl', 'rb') as f:
            dblp = pickle.load(f)

        papers = dblp['papers']
        author2id = dblp['author2id']
        paper2id = dblp['paper2id']
        conf2id = dblp['conf2id']
        paper2authors = dblp['paper2authors']
        index2title = dblp['index2title']
        citation_graph = dblp['citation_graph']
        first_half_papers = papers[len(papers)//2:]
        second_half_papers = papers[:len(papers)//2]

        if self.baseline:
            G = init_graph_baseline(first_half_papers, author2id)
        else:
            G = init_graph(first_half_papers, paper2id, conf2id, author2id, citation_graph, index2title)

        H = G.copy()
        for idx, p in tqdm(enumerate(second_half_papers), total=len(second_half_papers), dynamic_ncols=True):
            paper = p
            paper['title'] = paper2id[paper['title']]
            paper['authors'] = [author2id[a] for a in paper['authors']]
            if 'conf' in paper:
                paper['conf'] = conf2id[paper['conf']]
            async_postprocessing(paper, H.copy(), idx, self.processed_dir, self.cache_file_prefix,
                cutoff=self.cutoff)
            if self.baseline:
                H = append_paper_graph_baseline(H, deepcopy(p), author2id)
            else:
                H = append_paper_graph(H, deepcopy(p), paper2id, conf2id,
                                       author2id, citation_graph, index2title)

        match_filename = self.cache_file_prefix+'_*_v2.pt'
        self.processed_file_idx = list(glob.glob(osp.join(self.processed_dir, match_filename)))

    def get(self, idx):
        if isinstance(idx, list):
            new_self = deepcopy(self)
            new_self.processed_file_idx = np.array(self.processed_file_idx)[idx]
            return new_self
        filename = self.processed_file_idx[idx]
        data = torch.load(filename)
        data.size = torch.from_numpy(np.array([len(data.x)]))
        data.known = torch.from_numpy(np.array([ (data.input_mask == 1).sum()]))
        return data

class BatchPadData(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Pad features into a fixed size feature
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchPadData, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, pad_idx=874608):
        batch = BatchPadData()
        keys = ['x', 'y', 'label_mask', 'input_mask', 'edge_index', 'titleid', 'size', 'batch', 'size', 'known']
        for key in keys:
            #print(key)
            batch[key] = []

        cumsum_edge = 0
        max_size = 0
        for data in data_list:
            data_len = len(data.x)
            max_size = data_len if max_size < data_len else max_size

        for idx, data in enumerate(data_list):

            for key in ['titleid', 'size', 'known']:
                batch[key].append(data[key])

            for key in ['edge_index']:
                item = data[key]
                item = item + cumsum_edge if batch.cumsum(key, item) else item
                batch[key].append(item)

            seq_len = len(data['y'])
            for key in ['y', 'label_mask', 'input_mask']:
                item = data[key]
                attribute = item
                if (max_size - seq_len) > 0 :
                    pad_ = torch.from_numpy(np.array([0]*(max_size-seq_len)))
                    attribute =  torch.cat((attribute, pad_))
                batch[key].append(attribute)

            attribute = data['x']
            if (max_size - seq_len) > 0 :
                pad_ = torch.from_numpy(np.array([[pad_idx, 0, 0]]*(max_size-seq_len)))
                attribute =  torch.cat((attribute, pad_))
            batch['x'].append(attribute)
            batch['batch'].append( torch.from_numpy(np.array([idx]*max_size)) )

        for key in ['x', 'y', 'label_mask', 'input_mask', 'edge_index', 'titleid', 'size', 'batch', 'known']:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))

        batch['length'] = max_size
        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ['edge_index'] else 0

    def cumsum(self, key, item):
        return key in ['edge_index']
    
    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1



class PaddedDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(PaddedDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchPadData.from_data_list(data_list),
            **kwargs)


if __name__ == "__main__":
    # init_dblp()
    dataset = Aminer()
    dataloader = PaddedDataLoader(dataset, batch_size=8)
    for batch in dataloader:
        print(batch)
        print(len(batch.size))
        print(batch.x.reshape(-1, batch.length, 3).shape)
        break