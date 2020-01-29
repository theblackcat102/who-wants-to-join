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


def init_graph(papers, paper2id, conf2id, author2id):
    with open('aminer/preprocess_dblp.pkl', 'rb') as f:
        dblp = pickle.load(f)

    G = nx.Graph()

    for p in papers[:len(papers)//2]:
        p_id = 'p'+str(paper2id[p['title']])
        c_id = 'c'+str(conf2id[p['conf']])

        if not G.has_node(p_id):
            G.add_node(p_id)

        for a in p['authors']:
            a_id = 'a'+str(author2id[a])

            if not G.has_node(a_id):
                G.add_node(a_id)
            G.add_edge(p_id, a_id)    

        if not G.has_node(c_id):
            G.add_node(c_id)
        G.add_edge(c_id, p_id)

    return G

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
        if G.has_node(a) and len(predict_nodes) < predict_ratio:
            predict_nodes.append(a)
        else:
            exist_nodes.append(a)

    if len(predict_nodes) == 0:
        return None, 0, predict_ratio

    # print(predict_nodes)
    # print(exist_nodes)
    for start_node in authors:
        if G.has_node(start_node):
            n_nodes = get_neighbour_nodes(G, start_node, cutoff=cutoff)
            sub_graph_nodes += [n for n in n_nodes]
            sub_graph_nodes.append(start_node)

    sub_graph_nodes = set(sub_graph_nodes)
    # print(sub_graph_nodes)
    in_group_cnt = 0
    hit = 0
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
        basic_attributes['in_group'] = in_group
        basic_attributes['predict'] = predict
        basic_attributes['known'] = known_member
        sub_G.add_node(node, **basic_attributes)
    
    for node in sub_graph_nodes:
        for n in G.neighbors(node):
            if sub_G.has_node(node) and sub_G.has_node(n):
                sub_G.add_edge(node, n)
    return sub_G, hit, predict_ratio

def graph2data(G):
    '''
        [node_id, known/to predict : 0/1,
         node_type : [member: 0, topic: 1, category : 2, event: 3, group: 4]]
    '''
    graph_idx = {}

    # does the order of sub-graph index matter?
    # seems to me it's relative to one and another?

    for n in G.nodes:
        graph_idx[n] = len(graph_idx)

    nodes = []
    edges = []
    labels = []
    loss_mask = []

    for n in G.nodes:
        node_type = 0
        if G.nodes[n]['type'] == 'a':
            node_type = 0
        elif G.nodes[n]['type'] == 'p':
            node_type = 1
        elif G.nodes[n]['type'] == 'c':
            node_type = 2

        node_latent = Variable(
            torch.from_numpy(
                np.array([G.nodes[n]['id'], G.nodes[n]['known'], node_type])))

        edge_index = list(G.edges(n))
        new_edges = []
        for idx in range(len(edge_index)):
            src, dst = edge_index[idx]
            new_edges.append([graph_idx[dst], graph_idx[src]])
            new_edges.append([graph_idx[src], graph_idx[dst]])

        edges.append(new_edges)
        nodes.append(node_latent)
        if node_type == 0:
            loss_mask.append(1)
        else:
            loss_mask.append(0)

        labels.append(G.nodes[n]['predict'])

    x = torch.stack(nodes)
    y = torch.from_numpy(np.array(labels))
    loss_mask = torch.from_numpy(np.array(loss_mask))
    edges = torch.from_numpy(np.transpose(np.concatenate(edges))).contiguous()

    data = Data(x=x, edge_index=edges, y=y, label_mask=loss_mask)
    # add output mask to mask additional nodes : category, venue, topic node
    return data



class Aminer(Dataset):

    def __init__(self, cutoff=2, ratio=0.8, min_size=5, max_size=100,
                 city_id=10001):
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size
        if not os.path.exists('aminer/preprocess_dblp.pkl'):
            init_dblp()
        
        # with open('aminer/preprocess_dblp.pkl', 'rb') as f:
        #     cache_data = pickle.load(f)
        self.cache_file_prefix = '{}_{}_{}_{}'.format(
            'dblp', self.cutoff, self.ratio, self.min_size)
        self.processed_dir = osp.join(osp.join("processed", 'dblp_hete'), 'processed')
        match_filename = self.cache_file_prefix+'_*_v2.pt'
        self.list_of_data = list(glob.glob(osp.join(self.processed_dir, match_filename)))

        os.makedirs(osp.join(osp.join("processed", 'dblp_hete'), 'processed'), exist_ok=True)
        super(Aminer, self).__init__(osp.join("processed", 'dblp_hete'),
                                     transform=None,
                                     pre_transform=None)
        self.process()

    def init_graph():
        G = nx.Graph()
        with open('aminer/preprocess_dblp.pkl', 'rb') as f:
            cache_data = pickle.load(f)
        papers = cache_data['papers']
        for p in papers[:len(papers)//2]:
            p_id = 'p'+str(paper2id[p['title']])
            c_id = 'c'+str(conf2id[p['conf']])

            for a in p['authors']:
                a_id = 'a'+str(author2id[a])
                if not G.has_node(p_id):
                    G.add_node(p_id)
                if not G.has_node(a_id):
                    G.add_node(a_id)
                G.add_edge(p_id, a_id)    
            if not G.has_node(c_id):
                G.add_node(c_id)
            G.add_edge(c_id, p_id)
        return G

    @property
    def processed_file_names(self):
        return self.list_of_data

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
        return len(self.processed_file_names)

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

        G = nx.Graph()
        second_half_papers = papers[len(papers)//2:]
        for paper in tqdm(papers[:len(papers)//2]):

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

        H = G.copy()
        for idx, p in tqdm(enumerate(second_half_papers), 
            dynamic_ncols=True, total=len(second_half_papers)):
            papers = deepcopy(p)
            papers['title'] = paper2id[papers['title']]
            papers['authors'] = [ author2id[a] for a in papers['authors']]
            sub_G, hit, pred_cnt = create_subgraph(papers, H, cutoff=2)
            if sub_G is None:
                continue

            if idx < 1000:
                print(len(sub_G.nodes) , hit, pred_cnt)
            data = graph2data(sub_G)

            filename = self.cache_file_prefix+'_{}_v2.pt'.format(idx)
            torch.save(data, osp.join(self.processed_dir, filename))

            paper = deepcopy(p)
            p_id = 'p'+str(paper2id[paper['title']])
            c_id = 'c'+str(conf2id[paper['conf']])
            for a in paper['authors']:
                a_id = 'a'+str(author2id[a])
                if not H.has_node(p_id):
                    H.add_node(p_id)
                if citation_graph.has_node(paper['index']):
                    for n in citation_graph.neighbors(paper['index']):
                        neighbour_title = index2title[n]
                        id_ = paper2id[neighbour_title]
                        n_p_id = 'p'+str(id_)
                        if not H.has_node(n_p_id):
                            H.add_node(n_p_id)
                        H.add_edge(p_id, n_p_id)
                if not H.has_node(a_id):
                    H.add_node(a_id)
                H.add_edge(p_id, a_id)
            if not H.has_node(c_id):
                H.add_node(c_id)
            H.add_edge(c_id, p_id)
        self.list_of_data = list(glob.glob(osp.join(self.processed_dir, match_filename)))

    def get(self, idx):
        if isinstance(idx, list):
            self.processed_file_idx = idx
            return deepcopy(self)
        filename = self.processed_file_names[idx]
        data = torch.load(filename)
        return data


if __name__ == "__main__":
    # init_dblp()
    Aminer()
