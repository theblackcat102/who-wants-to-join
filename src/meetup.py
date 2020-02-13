import os.path as osp
import os
import glob
from copy import deepcopy
import pickle
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import networkx as nx
import torch
from torch.autograd import Variable
from torch_geometric.data import Dataset, Data
from src.dataset import split_group
from src.utils import pbar_listener
import gzip
import pandas as pd
import time


MEETUP_FOLDER = 'meetup_v2/'
MEETUP_GROUP = 'meetup_v2/groups.csv'
MEETUP_MEMBER = 'meetup_v2/members.csv'
MEETUP_TOPIC = 'meetup_v2/topics.csv'
MEETUP_CATEGORY = 'meetup_v2/categories.csv'
MEETUP_MEMBER2TOPIC = 'meetup_v2/members_topics.csv'
MEETUP_GROUP2TOPIC = 'meetup_v2/groups_topics.csv'

locations_id = {
    'NY': 10001,
    'Chicago': 60601,
    'SF': 94101
}


def build_initial_graph(city_id=10001, min_size=5, max_size=500, cutoff=2,
                        exist_ratio=0.8, min_freq=3):
    print('Load group and member csv (might take some time)')
    df = pd.read_csv(MEETUP_GROUP)
    df = df[df['city_id'] == city_id]

    df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values('created')

    groups = df.T.to_dict()
    groups = [m for _, m in groups.items()]
    groups.sort(key=lambda x: x['created'])

    valid_group_id = [int(g['group_id']) for g in groups]

    df = pd.read_csv(MEETUP_MEMBER, encoding='ISO-8859-1')
    df = df[df['group_id'].isin(valid_group_id)]
    print('Build member mapping', end=": ")
    t_build_member_mapping = time.perf_counter()
    group_map_name = '%d_%d_%d_%d_group_mapping.pkl' % (city_id, min_size,
                                                        max_size, min_freq)

    if not osp.exists(osp.join(MEETUP_FOLDER, group_map_name)):
        # group_mappings_ = defaultdict(list)
        # member_frequecies_ = defaultdict(int)
        # for idx, row in tqdm(df.iterrows()):
        #     if row['group_id'] in valid_group_id:
        #         group_mappings_[row['group_id']].append(row['member_id'])
        #         member_frequecies_[row['member_id']] += 1

        # valid_members_ = []
        # for m, freq in member_frequecies_.items():
        #     if freq >= min_freq:
        #         valid_members_.append(m)
        # valid_members_ = set(valid_members_)
        # new_group_ = defaultdict(list)
        # valid_members_ = set(valid_members_)
        # for group_id, members in group_mappings_.items():
        #     for m in members:
        #         if m in valid_members_:
        #             new_group_[group_id].append(m)
        # group_mappings_ = new_group_
        # all_group_id_ = list(group_mappings_.keys())
        # for group_id in all_group_id_:
        #     members_len = len(group_mappings_[int(group_id)])
        #     if members_len < min_size or members_len > max_size:
        #         group_mappings_.pop(group_id, None)

        group_mappings = df.groupby('group_id').agg({'member_id': list})
        group_mappings = group_mappings[
            group_mappings.index.isin(valid_group_id)]
        member_frequecies = (
            df['member_id'].value_counts()
            .rename_axis('member_id')
            .reset_index(name='counts')
            )
        print("filter member < %d" % min_freq)
        member_frequecies = member_frequecies[
            member_frequecies['counts'] >= min_freq]
        valid_members = set(member_frequecies.member_id.values)

        group_mappings = (
            group_mappings["member_id"]
            .apply(lambda x: [item for item in x if item in valid_members])
            .reset_index(name='member_id')
            .set_index('group_id'))
        all_group_id = list(group_mappings.index.values)
        group_mappings = group_mappings[
            (group_mappings['member_id'].map(len) >= min_size) &
            (group_mappings['member_id'].map(len) <= max_size)]
        group_mappings = group_mappings.T.to_dict()
        group_mappings = {k: v['member_id'] for k, v in group_mappings.items()}

        print("Init {} Group found".format(len(group_mappings)))
        with open(osp.join(MEETUP_FOLDER, group_map_name), 'wb') as f:
            pickle.dump(group_mappings, f)
    else:
        with open(osp.join(MEETUP_FOLDER, group_map_name), 'rb') as f:
            group_mappings = pickle.load(f)
    user2id_name = '%d_%d_%d_%d_user2id.pkl' % (city_id, min_size, max_size,
                                                min_freq)
    if not osp.exists(osp.join(MEETUP_FOLDER, user2id_name)):
        user2id = defaultdict(int)
        for _, members in group_mappings.items():
            for m in members:
                if m not in user2id:
                    user2id[m] = len(user2id)
        with open(osp.join(MEETUP_FOLDER, user2id_name), 'wb') as f:
            pickle.dump(user2id, f)
    else:
        with open(osp.join(MEETUP_FOLDER, user2id_name), 'rb') as f:
            user2id = pickle.load(f)
    print("{}ms".format(round(time.perf_counter()-t_build_member_mapping, 5)))
    print("{} Group found".format(len(group_mappings)))
    # build social network based on first half group
    new_groups = []
    all_group_id = list(group_mappings.keys())
    for g in groups:
        if g['group_id'] in all_group_id:
            new_groups.append(g)
    groups = new_groups
    flag = len(groups)//2
    first_half_group = groups[:flag]
    second_half_group = groups[flag:]

    G = nx.Graph()

    for group in first_half_group:
        group_id = group['group_id']
        members = group_mappings[int(group_id)]
        for idx, m in enumerate(members):
            if len(members) < (idx+1):
                for jdx, n in enumerate(members[idx+1:]):
                    if not G.has_node(n):
                        G.add_node(n, group=group_id)
                    if not G.has_node(m):
                        G.add_node(m, group=group_id)
                    if G.has_node(n) and G.has_node(m):
                        G.add_edge(n, m)

    return G, first_half_group, second_half_group


def save_topic_id():
    df = pd.read_csv(MEETUP_TOPIC, encoding='latin-1')
    topic2id = defaultdict(int)
    for idx, row in df.iterrows():
        topic2id[row['topic_id']] = len(topic2id)
    with open(osp.join(MEETUP_FOLDER, 'topic2id.pkl'), 'wb') as f:
        pickle.dump(topic2id, f)
    return topic2id


def save_category_id():
    df = pd.read_csv(MEETUP_CATEGORY, encoding='latin-1')
    cat2id = defaultdict(int)
    for idx, row in df.iterrows():
        cat2id[row['category_id']] = len(cat2id)
    with open(osp.join(MEETUP_FOLDER, 'cat2id.pkl'), 'wb') as f:
        pickle.dump(cat2id, f)
    return cat2id


def save_member2topic():
    df = pd.read_csv(MEETUP_MEMBER2TOPIC, encoding='latin-1')
    member2topic = defaultdict(list)
    with open(osp.join(MEETUP_FOLDER, 'topic2id.pkl'), 'rb') as f:
        topic2id = pickle.load(f)
    for idx, row in df.iterrows():
        member2topic[int(row['member_id'])].append(topic2id[row['topic_id']])
    with open(osp.join(MEETUP_FOLDER, 'member2topic.pkl'), 'wb') as f:
        pickle.dump(member2topic, f)
    return member2topic


def save_group2topic():
    df = pd.read_csv(MEETUP_GROUP2TOPIC, encoding='latin-1')
    group2topic = defaultdict(list)
    # with open(osp.join(MEETUP_FOLDER, 'topic2id.pkl'), 'rb') as f:
    #     topic2id = pickle.load(f)
    for idx, row in df.iterrows():
        group2topic[int(row['group_id'])].append(row['topic_id'])
    with open(osp.join(MEETUP_FOLDER, 'group2topic.pkl'), 'wb') as f:
        pickle.dump(group2topic, f)
    return group2topic


def graph2data(G, name2id, member2topic, group2topic, category2id, group2id,
               topic2id):
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
    # member: 0
    for n in G.nodes:
        node_latent = None
        if n in name2id:
            node_latent = Variable(
                torch.from_numpy(
                    np.array([name2id[n], G.nodes[n]['known_member'], 0])))
        else:
            print(str(n))
            continue

        edge_index = np.array(list(G.edges(n)))
        new_edges = []
        for idx, (src, dst) in enumerate(edge_index):
            # src, dst = edge_index[idx]
            # edge_index[idx] = [graph_idx[src], graph_idx[dst]]
            new_edges.append([graph_idx[dst], graph_idx[src]])
        edges.append(new_edges)
        nodes.append(node_latent)
        loss_mask.append(1)
        labels.append(G.nodes[n]['predict'])
    # topic: 1
    for n in G.nodes:
        if n in member2topic:
            for t in member2topic[int(n)]:
                t_id = 't'+str(t)
                if t_id not in graph_idx:
                    graph_idx[t_id] = len(graph_idx)
                    # no need to change topic2id
                    nodes.append(torch.from_numpy(np.array([t, -1, 1])))
                    loss_mask.append(0)
                    labels.append(0)
                edges.append([[graph_idx[t_id], graph_idx[n]]])
    # group: 4
    nodes.append(torch.from_numpy(
                    np.array([group2id[G.graph['group_id']], -1, 4])))
    loss_mask.append(0)
    labels.append(0)
    group_name = 'g'+str(G.graph['group_id'])
    graph_idx[group_name] = len(graph_idx)

    for n in G.nodes:
        edges.append([[graph_idx[n], graph_idx[group_name]]])
    # topic: 1
    if G.graph['group_id'] in group2topic:
        new_edges = []
        for t in group2topic[int(G.graph['group_id'])]:
            t_id = 't'+str(t)
            if t_id not in graph_idx:
                graph_idx[t_id] = len(graph_idx)
            nodes.append(torch.from_numpy(np.array([topic2id[t], -1, 1])))
            loss_mask.append(0)
            labels.append(0)
            new_edges.append([graph_idx[t_id], graph_idx[group_name]])
        edges.append(new_edges)
    # category: 2
    if G.graph['category_id'] in category2id:
        cat_name = 'c'+str(category2id[G.graph['category_id']])
        graph_idx[cat_name] = len(graph_idx)
        # no need category2id
        nodes.append(
            torch.from_numpy(
                np.array([G.graph['category_id'], -1, 2])))
        loss_mask.append(0)
        labels.append(0)
        new_edges.append([graph_idx[cat_name], graph_idx[group_name]])

    if len(nodes) == 0:
        raise ValueError('Invalid graph node')

    x = torch.stack(nodes)
    y = torch.from_numpy(np.array(labels))
    loss_mask = torch.from_numpy(np.array(loss_mask))
    edges = torch.from_numpy(np.transpose(np.concatenate(edges))).contiguous()

    data = Data(x=x, edge_index=edges, y=y, label_mask=loss_mask)
    # add output mask to mask additional nodes : category, venue, topic node
    return data


def async_graph_save(group, group_mappings, ratio, cutoff, G, user2id,
                     member2topic, group2topic, category2id, group2id,
                     topic2id, filename_prefix, processed_dir, file_idx,
                     pbar_queue, transform=None,pre_filter=None, pre_transform=None):
    group_id = group['group_id']
    members = group_mappings[int(group_id)]

    sub_G = split_group(group_id, members, G, ratio, cutoff)
    for key, value in group.items():
        sub_G.graph[key] = value

    data = graph2data(sub_G, user2id, member2topic, group2topic, category2id,
                      group2id, topic2id)

    # if pre_filter is not None:
    #     pre_filter(data)
    if pre_transform is not None:
        data = pre_transform(data)
        if data is None:
            return
    filename = filename_prefix+'_{}_v2.pt'.format(file_idx)
    torch.save(data, osp.join(processed_dir, filename))
    del G
    pbar_queue.put(1)


def convertmemberattributes(city_id, min_size, max_size, node_min_freq=3):
    df = pd.read_csv(MEETUP_MEMBER, encoding='latin-1')
    user2id_name = '%d_%d_%d_%d_user2id.pkl' % (city_id, min_size, max_size,
                                                node_min_freq)
    with open(os.path.join(MEETUP_FOLDER, user2id_name), 'rb') as f:
        user2id = pickle.load(f)

    df['member_id'] = df['member_id'].map(user2id)
    print(df.columns)
    df = df.drop(['group_id', 'member_name', 'bio', 'link', 'joined',
                  'country'], axis=1)
    print(df.head(100))


class Meetup(Dataset):
    def __init__(self, cutoff=2, ratio=0.8, min_size=5, max_size=100,
                 city_id=10001, min_freq=3, transform=None, data_type=None):
        self.cutoff = cutoff
        self.ratio = ratio
        self.min_size = min_size
        self.max_size = max_size
        self.group_size = 5000
        self.city_id = city_id
        self.min_freq = min_freq
        self.cache_file_prefix = '{}_{}_{}_{}_{}_{}'.format(
            'meetups', self.city_id, self.cutoff, self.ratio, self.min_size,
            self.min_freq)
        if data_type:
            self.cache_file_prefix = data_type+'_'+self.cache_file_prefix

        user2id_name = '%d_%d_%d_%d_user2id.pkl' % (
            self.city_id, self.min_size, self.max_size, self.min_freq)
        if osp.exists(osp.join(MEETUP_FOLDER, user2id_name)):
            with open(osp.join(MEETUP_FOLDER, user2id_name), 'rb') as f:
                self.user2id = pickle.load(f)
        group2id_name = '%d_%d_%d_%d_group2id.pkl' % (
            self.city_id, self.min_size, self.max_size, self.min_freq)

        if osp.exists(osp.join(MEETUP_FOLDER, group2id_name)):
            with open(osp.join(MEETUP_FOLDER, group2id_name), 'rb') as f:
                self.group2id = pickle.load(f)
        match_filename = self.cache_file_prefix + '_*_v2.pt'
        temp = osp.join("processed", str(self.city_id),
                                      'processed')
        self.processed_file_idx = sorted(list(glob.glob(osp.join(temp,
                                                          match_filename))))

        super(Meetup, self).__init__(osp.join("processed", str(city_id)),
                                     transform=transform,
                                     pre_transform=None)
        if len(self.processed_file_idx) == 0:
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
        if len(self.processed_file_idx) != 0:
            return

        cache_data_f = '{}_{}_{}_{}.pklz'.format(
            self.city_id, self.min_size, self.max_size, self.cutoff)
        if not osp.exists(osp.join(MEETUP_FOLDER, cache_data_f)):
            G, first_half_group, second_half_group = build_initial_graph(
                city_id=self.city_id,
                min_size=self.min_size, max_size=self.max_size,
                cutoff=self.cutoff, exist_ratio=self.ratio)
            with gzip.open(osp.join(MEETUP_FOLDER, cache_data_f), 'wb') as f:
                pickle.dump({
                    'G': G,
                    'first': first_half_group,
                    'second': second_half_group
                }, f)
        else:
            with gzip.open(osp.join(MEETUP_FOLDER, cache_data_f), 'rb') as f:
                cache_data = pickle.load(f)
            G, first_half_group, second_half_group = (
                cache_data['G'], cache_data['first'], cache_data['second'])
            del cache_data

        group2id_name = '%d_%d_%d_%d_group2id.pkl' % (
            self.city_id, self.min_size, self.max_size, self.min_freq)

        if not osp.exists(osp.join(MEETUP_FOLDER, group2id_name)):
            group2id = defaultdict(int)
            for group in second_half_group:
                group2id[group['group_id']] = len(group2id)
            with open(osp.join(MEETUP_FOLDER, group2id_name), 'wb') as f:
                pickle.dump(group2id, f)
        else:
            with open(osp.join(MEETUP_FOLDER, group2id_name), 'rb') as f:
                group2id = pickle.load(f)
        group_map_name = '%d_%d_%d_%d_group_mapping.pkl' % (
            self.city_id, self.min_size, self.max_size, self.min_freq)
        # user2id_name = '%d_%d_%d_user2id.pkl' % (
        #     self.city_id, self.min_size, self.max_size)
        group2id_name = '%d_%d_%d_%d_group2id.pkl' % (
            self.city_id, self.min_size, self.max_size, self.min_freq)

        with open(osp.join(MEETUP_FOLDER, group_map_name), 'rb') as f:
            group_mappings = pickle.load(f)
        with open(osp.join(MEETUP_FOLDER, group2id_name), 'rb') as f:
            group2id = pickle.load(f)

        user2id_name = '%d_%d_%d_%d_user2id.pkl' % (
            self.city_id, self.min_size, self.max_size, self.min_freq)
        if not osp.exists(osp.join(MEETUP_FOLDER, user2id_name)):
            user2id = defaultdict(int)
            for _, members in group_mappings.items():
                for m in members:
                    if m not in user2id:
                        user2id[m] = len(user2id)
            with open(osp.join(MEETUP_FOLDER, user2id_name), 'wb') as f:
                pickle.dump(user2id, f)
        else:
            with open(osp.join(MEETUP_FOLDER, user2id_name), 'rb') as f:
                user2id = pickle.load(f)
        self.user2id = user2id
        print('Build subgraph')
        # build social graph of each group
        # sub_groups = []

        # check cat2id, topic2id, group2topic, member2topic
        for feature in ['cat2id', 'topic2id', 'group2topic', 'member2topic']:
            if not osp.exists(osp.join(MEETUP_FOLDER, feature+'.pkl')):
                if feature == 'cat2id':
                    save_category_id()
                if feature == 'topic2id':
                    save_topic_id()
                if feature == 'group2topic':
                    save_group2topic()
                if feature == 'member2topic':
                    save_member2topic()
        with open(osp.join(MEETUP_FOLDER, 'cat2id.pkl'), 'rb') as f:
            cat2id = pickle.load(f)
        with open(osp.join(MEETUP_FOLDER, 'topic2id.pkl'), 'rb') as f:
            topic2id = pickle.load(f)
        with open(osp.join(MEETUP_FOLDER, 'group2topic.pkl'), 'rb') as f:
            group2topic = pickle.load(f)
        with open(osp.join(MEETUP_FOLDER, 'member2topic.pkl'), 'rb') as f:
            member2topic = pickle.load(f)

        user2id = self.user2id
        print('total group ', len(second_half_group))
        print('total users ', len(user2id))

        filename_prefix = self.cache_file_prefix

        # make dir for torch.save
        os.makedirs(self.processed_dir, exist_ok=True)

        # one progressbar for multiprocessing
        manager = mp.Manager()
        pbar_queue = manager.Queue()
        pbar_proc = mp.Process(target=pbar_listener,
                               args=[pbar_queue, len(second_half_group), ])
        pbar_proc.start()
        pool = mp.Pool(processes=mp.cpu_count()-1)
        results = []
        file_idx = 0
        for group in second_half_group:
            group_id = group['group_id']
            members = group_mappings[int(group_id)]

            if len(members) == 0:
                print('no member ', group_id)
                continue
            # add current group into Graph
            for idx, m in enumerate(members):
                if not G.has_node(m):
                    G.add_node(m, group=group_id)
                if len(members) > (idx+1):
                    for jdx, n in enumerate(members[idx+1:]):
                        if not G.has_node(n):
                            G.add_node(n, group=group_id)
                        if G.has_node(n) and G.has_node(m):
                            G.add_edge(n, m)

            # split current group into sub graph
            args = [group, group_mappings, self.ratio, self.cutoff, G.copy(),
                    user2id, member2topic, group2topic, cat2id, group2id,
                    topic2id, filename_prefix, self.processed_dir, file_idx,
                    pbar_queue]

            kwds = {
                'pre_filter': self.pre_filter,
                'pre_transform': self.pre_transform}
            res = pool.apply_async(async_graph_save, args=args, kwds=kwds)
            results.append(res)
            file_idx += 1
            
            # if group_idx % 200 == 0 and group_idx != 0:
            #     sleep(10)
        # for res in results:
        #     res.get()
        pool.close()
        pool.join()
        pbar_queue.put(None)
        pbar_proc.join()
        match_filename = self.cache_file_prefix + '_*_v2.pt'
        self.processed_dir = osp.join("processed", str(self.city_id),
                                      'processed')
        self.processed_file_idx = sorted(list(glob.glob(osp.join(self.processed_dir,
                                                          match_filename))))

    def __len__(self):
        return len(self.processed_file_idx)

    def get(self, idx):
        if isinstance(idx, list):
            new_ = deepcopy(self)
            new_.processed_file_idx = list(
                np.array(self.processed_file_idx)[idx])
            return new_

        filename = self.processed_file_idx[idx]
        data = torch.load(filename)
        return data


if __name__ == "__main__":
    # from src.layers import StackedGCNMeetup
    # save_category_id()
    # save_topic_id()
    # save_member2topic()
    # save_group2topic()
    # build_initial_graph()
    import argparse
    parser = argparse.ArgumentParser(
        description='Deepset Recommendation Model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='SF',
                        choices=['NY', 'SF'])
    args = parser.parse_args()
    dataset = Meetup(city_id=locations_id[args.dataset])
    print(len(dataset))
    subset = dataset[[1, 2, 3, 4]]
    print(dataset[5])

    # with open('cache.pkl', 'rb') as f:
    #     sub_G = pickle.load(f)
    # user2id_name = '%d_%d_%d_user2id.pkl' % (locations_id['SF'], 5, 500)
    # print(osp.join(MEETUP_FOLDER, user2id_name))
    # with open(osp.join(MEETUP_FOLDER, user2id_name), 'rb') as f:
    #     user2id = pickle.load(f)
    # with open(osp.join(MEETUP_FOLDER, 'cat2id.pkl'), 'rb') as f:
    #     cat2id = pickle.load(f)
    # with open(osp.join(MEETUP_FOLDER, 'group2topic.pkl'), 'rb') as f:
    #     group2topic = pickle.load(f)
    # with open(osp.join(MEETUP_FOLDER, 'member2topic.pkl'), 'rb') as f:
    #     member2topic = pickle.load(f)
    # print('finish loading')
    # # member2topic, group2topic, category2id
    # data = graph2data(sub_G, user2id, member2topic, group2topic, cat2id)
    # print(data)
