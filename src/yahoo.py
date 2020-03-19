import os, glob
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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from src.utils import dict2table, pbar_listener, confusion
from src.dataset import split_group, chunks
from dataset.aminer import get_neighbour_nodes
from src.layers import StackedGCNYahoo
import multiprocessing as mp


USER_OFFSET = 638124
USER_SIZE = 999745 # include offset 1

def get_node_name(node_id):
    if node_id <= USER_OFFSET:
        node_type = 'g'
    else: # > 638124
        node_type = 'u'
        node_id -= USER_OFFSET
    node_name = node_type+str(node_id)
    return node_name

def init_graph_with_group(size='normal'):
    '''
        1,637,868 total nodes
            638,124 nodes on left  (representing groups)
            999,744 nodes on right (represent users)
        15,205,016 edges          (representing group membership)

        For convenience, left side nodes are numbered from 1 to N_left,
        and right side nodes are numbered from N_left+1 to N_left+N_right.
        line separated by spaces
    '''
    G = nx.Graph()
    group_mappings = {}
    limit = 1000000
    if size == 'small':
        limit = 10000+1
    print(limit)
    with open('data/yahoo-group/ydata-ygroups-user-group-membership-graph-v1_0.txt', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0: # ignore header line
                continue
            node_id = idx
            node_name = get_node_name(node_id)
            if node_name[0] == 'g' and node_id >= limit:
                continue

            edges_node = [ int(node_id_) for node_id_ in line.strip().split(' ') ]
            if edges_node < 5:
                continue
            
            if not G.has_node(node_name):
                G.add_node(node_name)
            cnt = 0
            valid_nodes = []
            for n in edges_node:
                if node_name[0] == 'u' and n >= limit:
                    continue

                neighbour_name = get_node_name(n)
                if not G.has_node(neighbour_name):
                    G.add_node(neighbour_name)
                G.add_edge(node_name, neighbour_name)
                valid_nodes.append(n)
                cnt += 1

            if cnt == 0:
                G.remove_node(node_name)
            elif node_name[0] == 'g':
                group_mappings[node_id] = np.array(valid_nodes) - USER_OFFSET

    return G, group_mappings


def graph2data(G):
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
        attributes = G.nodes[n]
        node_latent = None
        node_latent = Variable(
            torch.from_numpy(
                np.array([attributes['id'], attributes['known_member'], attributes['type']]
            )))

        edge_index = np.array(list(G.edges(n)))
        new_edges = []

        for idx in range(len(edge_index)):
            src, dst = edge_index[idx]
            edge_index[idx] = [graph_idx[src], graph_idx[dst]]
            new_edges.append([graph_idx[dst], graph_idx[src]])
            new_edges.append([graph_idx[src], graph_idx[dst]])

        if attributes['type'] == 0:
            loss_mask.append(1)
        else:
            loss_mask.append(0)

        edges.append(new_edges)
        nodes.append(node_latent)
        labels.append(attributes['predict'])


    if len(nodes) == 0:
        raise ValueError('Invalid graph node')
    x = torch.stack(nodes)
    y = torch.from_numpy(np.array(labels))
    loss_mask = torch.from_numpy(np.array(loss_mask))

    edges = torch.from_numpy(np.transpose(np.concatenate(edges))).contiguous()
    return Data(x=x, edge_index=edges, y=y, label_mask=loss_mask)


def preprocess_groups(G, group_mappings, pbar_queue, max_size, min_size, ratio, cutoff,
    cache_file_prefix,processed_dir):
    in_group_cnt = 0
    for group_id, members in group_mappings.items():
        if len(members) < min_size or len(members) > max_size:
            continue
        random.shuffle(members)
        ratio_ = int(len(members)*ratio)
        predict_ratio = len(members) - ratio_
        # make sure there's at least 2 group member to predict
        if predict_ratio < 2:
            predict_ratio = 2
            ratio_ = len(members) - predict_ratio
        exist_nodes = members[:ratio_]
        pred_nodes = members[ratio_:]
        sub_graph_nodes = []
        for start_node in exist_nodes:
            node_name = get_node_name(start_node)
            n_nodes = get_neighbour_nodes(G, node_name, cutoff=cutoff)
            sub_graph_nodes += [n for n in n_nodes]
            sub_graph_nodes.append(node_name)

        sub_graph_nodes = set(sub_graph_nodes)
        sub_G = nx.Graph()
        for node_name in sub_graph_nodes:
            node_id = int(node_name[1:])
            node_type = 1 if node_name[0] == 'g' else 0
            in_group = 1 if node_id in members else 0
            known_member = 1 if node_id in exist_nodes else 0
            predict = 0

            if node_id in pred_nodes and node_type == 0:
                predict = 1
            in_group_cnt += in_group
            sub_G.add_node(node_name, in_group=in_group, predict=predict,
                        known_member=known_member, id=node_id, type=node_type)

        for node_name in sub_graph_nodes:
            for n in G.neighbors(node_name):
                if sub_G.has_node(node_name) and sub_G.has_node(n):
                    sub_G.add_edge(node_name, n)

        data = graph2data(sub_G)
        filename = cache_file_prefix+'_{}_v2.pt'.format(group_id)
        torch.save(data, osp.join(processed_dir, filename))
        pbar_queue.update(1)
    return in_group_cnt

class Yahoo(Dataset):
    def __init__(self, cutoff=2, ratio=0.8, min_size=5,
                max_size=500, include_group=True):
        '''
        1,637,868 total nodes
            638,124 nodes on left  (representing groups)
            999,744 nodes on right (represent users)
        15,205,016 edges          (representing group membership)

        For convenience, left side nodes are numbered from 1 to N_left,
        and right side nodes are numbered from N_left+1 to N_left+N_right.

        '''
        self.data_folder = 'yahoo_hete'
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size
        self.group_size = 0

        self.cache_file_prefix = '{}_{}_{}_{}'.format(
            self.data_folder, self.cutoff, self.ratio, self.min_size)

        self.processed_dir = osp.join(osp.join("processed", self.data_folder), 'processed')
        match_filename = self.cache_file_prefix+'_*_v2.pt'
        self.list_of_data = list(glob.glob(osp.join(self.processed_dir, match_filename)))

        os.makedirs(osp.join(osp.join("processed", self.data_folder), 'processed'), exist_ok=True)
        super(Yahoo, self).__init__(osp.join("processed", self.data_folder),
                                     transform=None,
                                     pre_transform=None)
        if len(self.list_of_data) == 0:
            self.process()

    def process(self):
        if len(self.list_of_data) != 0:
            return
        print('initialze graph mappings')
        if not os.path.exists('data/yahoo-group/cache.pkl'):
            G, group_mappings = init_graph_with_group()
            with open('data/yahoo-group/cache.pkl', 'wb') as f:
                pickle.dump({
                    'G': G, 'mappings': group_mappings
                }, f)
        else:
            with open('data/yahoo-group/cache.pkl', 'rb') as f:
                cache = pickle.load(f)
            G, group_mappings = cache['G'], cache['mappings']

        in_group_cnt = 0
        # with tqdm(total=len(group_mappings), dynamic_ncols=True) as pbar:
        #     preprocess_groups(G, group_mappings, pbar, 
        #      self.max_size, self.min_size, self.ratio, self.cutoff,
        #      self.cache_file_prefix, self.processed_dir)

        manager = mp.Manager()
        pbar_queue = manager.Queue()
        pbar_proc = mp.Process(target=pbar_listener,
                               args=[pbar_queue, len(group_mappings), ])
        pbar_proc.start()
        results = []
        idx = 0
        # chunkize to cpu_count()*5 for better load balance
        chunk_size = len(group_mappings)//3
        pool = mp.Pool(processes=3)
        for sub_group2member in chunks(group_mappings, chunk_size):
            args = [G, sub_group2member, pbar_queue, 
                self.max_size, self.min_size, self.ratio, self.cutoff,
                 self.cache_file_prefix, self.processed_dir]
            res = pool.apply_async(preprocess_groups, args=args)
            results.append(res)
            idx += len(sub_group2member)

        for res in results:
            in_group_cnt += res.get()
        pool.close()
        pool.join()
        pbar_queue.put(None)
        pbar_proc.join()
        print('Total {}/{}'.format(idx, len(group_mappings)))



    def __len__(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if isinstance(idx, list):
            new_copy = deepcopy(self)
            new_copy.list_of_data = np.array(self.list_of_data)[idx]
            return new_copy
        filename = self.processed_file_names[idx]
        data = torch.load(filename)
        return data

    @property
    def processed_file_names(self):
        return self.list_of_data

    def _download(self):
        pass

    @property
    def raw_file_names(self):
        return ['some_file_1']



class GroupGCN():
    def __init__(self, args):
        dataset_name = 'yahoo'
        dataset = Yahoo(cutoff=args.maxhop,
            min_size=args.min_size, max_size=args.max_size)

        # make sure each runs share the same results
        if osp.exists('yahoo_g3_shuffle_idx.pkl'):
            with open('yahoo_g3_shuffle_idx.pkl', 'rb') as f:
                shuffle_idx = pickle.load(f)
        else:
            shuffle_idx = [idx for idx in range(len(dataset))]
            random.shuffle(shuffle_idx)
            with open('yahoo_g3_shuffle_idx.pkl', 'wb') as f:
                pickle.dump(shuffle_idx, f)

        dataset = dataset[shuffle_idx]

        split_pos = int(len(dataset)*0.7)
        train_idx = shuffle_idx[:split_pos]
        valid_idx_ = shuffle_idx[split_pos:]
        test_pos = int(len(valid_idx_)*0.333)
        test_idx = valid_idx_[:test_pos]
        valid_idx = valid_idx_[test_pos:]

        self.train_dataset = dataset[train_idx]
        self.test_dataset = dataset[test_idx]
        self.valid_dataset = dataset[valid_idx]

        self.args = args

        self.log_path = osp.join(
            "logs", "yahoo",
            'multi_yahoo_hete_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.save_path = osp.join(self.log_path, "models")
        os.makedirs(self.save_path, exist_ok=True)
        print('finish init')

    def save_checkpoint(self, checkpoint, save_path, save_name):
        save_path = osp.join(save_path, 'model_{}.pth'.format(save_name))
        torch.save(checkpoint, save_path)

    def evaluate(self, dataloader, model):
        model.eval()
        recalls = []
        precisions = []
        B = args.batch_size
        user_size = USER_SIZE
        print('Validation')
        with torch.no_grad():
            y_pred = torch.FloatTensor(B, user_size)
            y_target = torch.FloatTensor(B, user_size)
            for val_data in tqdm(dataloader, dynamic_ncols=True):
                x, edge_index = val_data.x, val_data.edge_index
                y = val_data.y
                pred_mask = val_data.label_mask
                pred = model(edge_index.cuda(), x.cuda())
                pred = torch.sigmoid(pred).cpu()
                # y = y[pred_mask]
                y_pred.zero_()
                y_target.zero_()

                for idx, batch_idx in enumerate(val_data.batch):
                    if pred_mask[idx] == 1:
                        if y[idx] == 1:
                            y_target[batch_idx.data, x[idx][0]] = 1
                        if pred[idx] > 0.5:
                            y_pred[batch_idx.data, x[idx][0]] = 1

                TP, FP, TN, FN = confusion(y_pred, y_target)

                recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
                precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
                precisions.append(precision)
                recalls.append(recall)

        avg_recalls = np.mean(recalls)
        avg_precisions = np.mean(precisions)
        f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)
        model.train()
        return f1, avg_recalls, avg_precisions

    def train(self, epochs=200):
        args = self.args
        train_size = len(self.train_dataset)
        val_size = len(self.valid_dataset)
        assert (len(set(self.valid_dataset+self.train_dataset))) == (train_size+val_size)

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=args.val_batch_size,
                                  shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=args.val_batch_size,
                                 shuffle=False)

        model = StackedGCNYahoo(
                           user_dim=args.user_dim,
                           group_dim=args.group_dim,
                           input_channels=args.input_dim,
                           layers=args.layers,
                           dropout=args.dropout)
        model = model.cuda()

        if args.pos_weight <= 0:
            weight = 100  # default
            args.pos_weight = weight
        else:
            weight = args.pos_weight
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=5e-4)
        print('weight : ', weight)
        pos_weight = torch.ones([1])*weight
        pos_weight = pos_weight.cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        crossentropy = torch.nn.CrossEntropyLoss()
        n_iter = 0
        best_f1 = 0

        self.writer.add_text('Text', dict2table(vars(args)), 0)

        with tqdm(total=len(train_loader)*epochs, dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                for data in train_loader:
                    optimizer.zero_grad()
                    x, edge_index = data.x, data.edge_index
                    x = x.cuda()
                    edge_index = edge_index.cuda()
                    pred_mask = data.label_mask.cuda()
                    label = data.y.unsqueeze(-1).cuda().float()
                    output = model(edge_index, x)
                    # entropy_loss = crossentropy(node_pred, x[:, 2])
                    binary_loss = criterion(output[ pred_mask == 1 ], label[ pred_mask == 1 ])
                    loss = binary_loss#+entropy_loss
                    loss.backward()

                    optimizer.step()
                    self.writer.add_scalar(
                        "Train/BCEWithLogitsLoss", binary_loss.item(), n_iter)
                    # self.writer.add_scalar(
                    #     "Train/CrossEntropyLoss", entropy_loss.item(), n_iter
                    # )
                    pbar.update(1)
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(loss.item(), epoch))
                    n_iter += 1

                if epoch % args.eval == 0:
                    print('Epoch: ', epoch)
                    f1, recalls, precisions = self.evaluate(valid_loader,
                                                            model)
                    self.writer.add_scalar("Valid/F1", f1, n_iter)
                    self.writer.add_scalar("Valid/Recalls", recalls, n_iter)
                    self.writer.add_scalar("Valid/Precisions", precisions,
                                           n_iter)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_checkpoint = {
                            'epoch': epoch+1,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'f1': f1
                        }
                    print(f1)

                if epoch % args.save == 0:
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f1': f1
                        },
                        self.save_path,
                        "{}".format(epoch+1)
                    )

        print("Testing")
        self.save_checkpoint(best_checkpoint,
                             self.save_path,
                             "best")
        model.load_state_dict(best_checkpoint["model"])
        f1, recalls, precisions = self.evaluate(test_loader, model)
        self.writer.add_scalar("Test/F1", f1, n_iter)
        self.writer.add_scalar("Test/Recalls", recalls, n_iter)
        self.writer.add_scalar("Test/Precisions", precisions, n_iter)
        self.writer.flush()

if __name__ == "__main__":
    # Yahoo()
    import argparse
    parser = argparse.ArgumentParser(
        description='Deepset Recommendation Model on Amazon with categories')
    # dataset parameters
    parser.add_argument('--min-size', type=int, default=5)
    parser.add_argument('--max-size', type=int, default=100)
    parser.add_argument('--pred-ratio', type=float, default=0.8)
    parser.add_argument('--maxhop', type=int, default=2)
    # training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pos-weight', type=float, default=-1)
    parser.add_argument('--eval', type=int, default=10)
    parser.add_argument('--save', type=int, default=50)
    # model parameters
    parser.add_argument('--user-dim', type=int, default=8)
    parser.add_argument('--group-dim', type=int, default=4)
    parser.add_argument('--input-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=[16, 16, 16])

    args = parser.parse_args()

    trainer = GroupGCN(args)
    trainer.train(epochs=args.epochs)
