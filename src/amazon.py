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


def split_group(group_id, members, G, exist_ratio, cutoff):
    random.shuffle(members)
    ratio_ = int(len(members)*exist_ratio)
    predict_ratio = len(members) - ratio_
    # make sure there's at least 2 group member to predict
    if predict_ratio < 2:
        predict_ratio = 2
        ratio_ = len(members) - predict_ratio

    exist_nodes = members[:ratio_]

    # find nodes reachable from start_node within cutoff distance
    sub_graph_nodes = []
    for start_node in exist_nodes:

        n_nodes = nx.single_source_shortest_path_length(G, start_node,
                                                        cutoff=cutoff)
        sub_graph_nodes += [n for n in n_nodes]
        sub_graph_nodes.append(start_node)

    # Build subgraph
    sub_graph_nodes = set(sub_graph_nodes)
    sub_G = nx.Graph()
    in_group_cnt = 0
    for node in sub_graph_nodes:
        in_group = 1 if node in members else 0
        known_member = 1 if node in exist_nodes else 0
        predict = 0
        if node in exist_nodes:
            predict = 0
        elif node in members and node not in exist_nodes:
            predict = 1
        in_group_cnt += in_group
        sub_G.add_node(node, in_group=in_group, predict=predict,
                        known_member=known_member)

    for node in sub_graph_nodes:
        for n in G.neighbors(node):
            if sub_G.has_node(node) and sub_G.has_node(n):
                sub_G.add_edge(node, int(n))
    return sub_G

def chunks(data, size=10000):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}

def graph2data(G, name2id, node_attr, cat2id):
    graph_idx = {}
    # does the order of sub-graph index matter?
    # seems to me it's relative to one and another?
    for n in G.nodes:
        graph_idx[n] = len(graph_idx)
    nodes = []
    edges = []
    labels = []
    label_mask = []
    for n in G.nodes:
        node_latent = None
        if str(n) in name2id:
            node_latent = Variable(
                torch.from_numpy(
                    np.array([name2id[str(n)], G.nodes[n]['known_member'], 0])))
        else:
            print(str(n))
            continue

        edge_index = np.array(list(G.edges(n)))
        new_edges = []
        for idx in range(len(edge_index)):
            src, dst = edge_index[idx]
            edge_index[idx] = [graph_idx[src], graph_idx[dst]]
            new_edges.append([graph_idx[dst], graph_idx[src]])
        edges.append(new_edges)
        nodes.append(node_latent)
        labels.append(G.nodes[n]['predict'])
        label_mask.append(1)
    
    for n in G.nodes:
        if n in node_attr:
            attributes = node_attr[n]
            for cat in attributes['categories'][:100]:
                if cat in [1000]:
                    continue
                cat_name = 'cat'+str(cat)
                if cat_name not in graph_idx:
                    graph_idx[cat_name] = len(graph_idx)
                    node_latent = Variable(torch.from_numpy(np.array([cat2id[cat], -1, 1])))
                    nodes.append(node_latent)
                    labels.append(0)
                    label_mask.append(0)
                edges.append([[ graph_idx[cat_name], graph_idx[n] ]])

    if len(nodes) == 0:
        raise ValueError('Invalid graph node')
    x = torch.stack(nodes)
    label_mask = torch.from_numpy(np.array(label_mask))
    y = torch.from_numpy(np.array(labels))
    edges = torch.from_numpy(np.transpose(np.concatenate(edges))).contiguous()
    return Data(x=x, edge_index=edges, y=y, label_mask=label_mask)


def create_sub_graph(G, group2member, user2id, pbar_queue,
                     node_attr, cat2id,
                     processed_dir='./processed', dataset='amazon', startidx=0,
                     exist_ratio=0.8, cutoff=2, min_size=2, max_size=1000,
                     pre_filter=None, pre_transform=None):
    idx = startidx
    filename_prefix = '{}_{}_{}_{}_hete'.format(dataset, cutoff, exist_ratio,
                                           min_size)
    for group_id, members in group2member.items():
        random.shuffle(members)
        ratio_ = int(len(members)*exist_ratio)
        predict_ratio = len(members) - ratio_
        # make sure there's at least 2 group member to predict
        if predict_ratio < 2:
            predict_ratio = 2
            ratio_ = len(members) - predict_ratio

        exist_nodes = members[:ratio_]

        # find nodes reachable from start_node within cutoff distance
        sub_graph_nodes = []
        for start_node in exist_nodes:
            n_nodes = nx.single_source_shortest_path_length(G, start_node,
                                                            cutoff=cutoff)
            sub_graph_nodes += [n for n in n_nodes]
            sub_graph_nodes.append(start_node)

        # build subgraph
        sub_graph_nodes = set(sub_graph_nodes)
        sub_G = nx.Graph()
        in_group_cnt = 0
        for node in sub_graph_nodes:
            in_group = 1 if node in members else 0
            known_member = 1 if node in exist_nodes else 0
            predict = 0
            if node in exist_nodes:
                predict = 0
            elif node in members and node not in exist_nodes:
                predict = 1
            in_group_cnt += in_group
            sub_G.add_node(node, in_group=in_group, predict=predict,
                           known_member=known_member)

        for node in sub_graph_nodes:
            for n in G.neighbors(node):
                if sub_G.has_node(node) and sub_G.has_node(n):
                    sub_G.add_edge(node, int(n))
        if len(sub_G.nodes) == 0:
            continue
        # graph to data
        data = graph2data(sub_G, user2id, node_attr, cat2id)
        if pre_filter is not None and not pre_filter(data):
            continue
        if pre_transform is not None:
            data = pre_transform(data)
        # save data
        filename = filename_prefix+'_{}_v2.pt'.format(idx)
        torch.save(data, osp.join(processed_dir, filename))
        idx += 1
        # clear memory
        sub_G.clear()
        del sub_G, sub_graph_nodes
    pbar_queue.put(len(group2member))


def load_amazon_meta():
    amazon_node = []
    with open('data/amazon/amazon-meta.txt', 'r') as f:
        lines = f.readlines()
        idx = 0
        print(len(lines))
        object_attr = {}

        while idx < len(lines):
            line = lines[idx]
            if ('Id:' in line and 'title' not in line )or 'discontinued' in line:
                if len(object_attr) > 2:
                    amazon_node.append(object_attr)
                    object_attr = {}
                if 'Id:' in line:
                    try:
                        object_attr = {'id': int(line.strip().split(':')[-1].strip()) }
                    except:
                        print(line)

            if 'group' in line:
                object_attr['group'] = line.strip().split(':')[-1].strip()
            if 'title' in line:
                object_attr['title'] = line.strip().split(':')[-1].strip()
            if 'categories' in line:
                offset = int(line.strip().split(':')[-1].strip())
                categories = []
                for jdx in range(offset):
                    cat_line = lines[idx + jdx + 1]
                    for cat in cat_line.split('|'):
                        try:
                            if '[' in cat:
                                cat_num = cat.split('[')[1].strip()[:-1]
                                if len(cat_num) == 0:
                                    print(cat)
                                else:
                                    categories.append(int(cat_num))
                        except KeyboardInterrupt:
                            break
                        except:
                            continue
                if len(categories) > 0:
                    object_attr['categories'] = list(set(categories))
                idx += offset
            idx += 1

    amazon_meta_data = {}
    cat2id = defaultdict(int)
    categories = []
    for node in amazon_node:
        if 'categories' in node:
            categories += node['categories']
        else:
            node['categories'] = []
        amazon_meta_data[node['id']] = node

    for cat in list(set(categories)):
        cat2id[cat] = len(cat2id)

    print(len(cat2id))

    with open('data/amazon/amazon_meta.pkl', 'wb') as f:
        pickle.dump(amazon_meta_data, f)
    with open('data/amazon/cat2id.pkl', 'wb') as f:
        pickle.dump(cat2id, f)
    print('saved meta data')



class AmazonCommunity(Dataset):
    def __init__(self, cutoff=2, ratio=0.8, min_size=5,
                 max_size=500):
        self.dataset = 'amazon'
        dataset = 'amazon'
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size
        self.group_size = 0


        dataset_path = osp.join("data", dataset)
        postfix = ""
        if self.dataset == "amazon":
            postfix = ".dedup"
        dataset_filename = "com-{}.all{}.cmty.txt".format(dataset, postfix)
        self.dataset_filepath = osp.join(dataset_path, dataset_filename)
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

        super(AmazonCommunity, self).__init__(osp.join("processed", dataset+'_hete'),
                                            transform=None,
                                            pre_transform=None)
        # self.process()
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [
            '{}_{}_{}_{}_hete_{}_v2.pt'.format(
                self.dataset, self.cutoff, self.ratio, self.min_size, idx)
            for idx in self.processed_file_idx
        ]

    def _download(self):
        pass

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    def process(self):
        user2id = self.user2id
        length = 0
        print(self.processed_dir)
        for idx in range(self.group_size):
            filename = '{}_{}_{}_{}_hete_{}_v2.pt'.format(
                self.dataset, self.cutoff, self.ratio, self.min_size, idx)
            length = idx
            if not os.path.exists(osp.join(self.processed_dir, filename)):
                print(filename)
                # all_found = False
                length = idx
                break
        if length != 0:
            self.group_size = length
            self.processed_file_idx = [idx for idx in range(self.group_size)]
            return
        with open('data/amazon/amazon_meta.pkl', 'rb') as f:
            node_attr = pickle.load(f)
        with open('data/amazon/cat2id.pkl', 'rb') as f:
            cat2id = pickle.load(f)

        member2group = defaultdict(list)
        group2member = defaultdict(list)
        edges = defaultdict(int)
        print('found pretrain embeddings...')
        edges_filename = "com-{}.ungraph.txt".format(self.dataset)
        edges_filepath = osp.join("data", self.dataset, edges_filename)
        # create member edge
        with open(edges_filepath, 'r') as f, tqdm() as pbar:
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
                edge = [str(int(m)) for m in line.strip().split('\t')]
                edge = '_'.join(edge)
                edges[edge] = 1
        print('load community graph')
        # load directed graph
        with open(self.dataset_filepath, 'r') as f, tqdm() as pbar:
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
                members = [int(m) for m in members]
                if (len(members) < self.min_size or
                        len(members) > self.max_size):
                    continue
                group_id = len(group2member)
                group2member[group_id] = members
                for m in members:
                    member2group[m].append(group_id)
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

        print('populate sub graph')
        # sub_graphs = create_sub_graph(G, group2member,
        #                               exist_ratio=self.ratio,
        #     cutoff=self.cutoff, min_size=self.min_size)
        idx = 0
        manager = mp.Manager()

        # one progressbar for multiprocessing
        def pbar_listener(q, total_size):
            pbar = tqdm(total=total_size)
            while True:
                item = q.get()
                if item is None:
                    break
                pbar.update(item)

        pbar_queue = manager.Queue()
        pbar_proc = mp.Process(target=pbar_listener,
                               args=[pbar_queue, len(group2member), ])
        pbar_proc.start()
        # chunkize to cpu_count()*5 for better load balance
        total_cpu = 4 #mp.cpu_count()
        chunk_size = len(group2member)//total_cpu//5
        pool = mp.Pool(processes=total_cpu)
        results = []
        for sub_group2member in chunks(group2member, chunk_size):
            args = [G, sub_group2member, user2id, pbar_queue, node_attr, cat2id ]
            kwds = {
                'processed_dir': self.processed_dir, 'dataset': self.dataset,
                'startidx': idx, 'exist_ratio': self.ratio,
                'cutoff': self.cutoff, 'min_size': self.min_size,
                'max_size': 1000, 'pre_filter': self.pre_filter,
                'pre_transform': self.pre_transform}
            res = pool.apply_async(create_sub_graph, args=args, kwds=kwds)
            # results.append(res)
            idx += len(sub_group2member)
        # for res in results:
        #     res.get()
        pool.close()
        pool.join()
        pbar_queue.put(None)
        pbar_proc.join()
        print('Total {}'.format(idx))

    def __len__(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if isinstance(idx, list):
            self.processed_file_idx = idx
            return deepcopy(self)

        filename = '{}_{}_{}_{}_hete_{}_v2.pt'.format(
            self.dataset, self.cutoff, self.ratio, self.min_size, idx)
        data = torch.load(osp.join(self.processed_dir, filename))
        return data


if __name__ == "__main__":
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    import torch.nn as nn
    from src.layers import StackedGCNAmazon
    load_amazon_meta()
    # layer = GCNConv(64, 1)
    dataset = AmazonCommunity()
    # dataset[:540]

    # with open('data/amazon/cat2id.pkl', 'rb') as f:
    #     cat2id = pickle.load(f)
    # model = StackedGCNAmazon(user_size=len(dataset.user2id), category_size=len(cat2id))
    # model = model.cuda()
    # model.train()

    # print(len(dataset))

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=0.0005, weight_decay=5e-4)
    # pos_weight = torch.ones([1])*30
    # pos_weight = pos_weight.cuda()
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # for epoch in range(200):
    #     print(epoch)
    #     predictions = []
    #     targets = []
    #     with tqdm(total=len(loader)) as pbar:
    #         for data in loader:
    #             optimizer.zero_grad()
    #             # data = data.cuda()
    #             x, edge_index = data.x, data.edge_index
    #             x = x.cuda()
    #             edge_index = edge_index.cuda()
    #             pred_mask = data.label_mask.cuda()
    #             label = data.y.unsqueeze(-1).cuda().float()
    #             output = model(edge_index, x)

    #             loss = criterion(output[pred_mask], label[pred_mask])
    #             loss.backward()
    #             optimizer.step()

    #             targets.append(data.y.cpu().detach().numpy())
    #             predictions.append(output[pred_mask].cpu().detach().numpy())
    #             # break
    #             pbar.update(1)
    #             pbar.set_description("loss {:.4f}".format(loss.item()))

    #     targets = np.concatenate(targets)
    #     predictions = np.concatenate(predictions) > 0.5
    #     print(np.sum(predictions), np.sum(targets))
    #     score = f1_score(targets, predictions, average="micro")
    #     baselines = np.zeros(targets.shape)
    #     baseline_score = f1_score(targets, baselines, average='micro')
    #     print("\nF-1 score: {:.4f}, {:.4f}".format(score, baseline_score))