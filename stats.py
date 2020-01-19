import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm


def compute_dataset(dataset):
    member2group = defaultdict(list)
    group2member = defaultdict(list)
    edges = defaultdict(int)

    # create member edge
    with open('{}/com-{}.ungraph.txt'.format(dataset, dataset), 'r') as f:
        for line in tqdm(f.readlines()):
            edge = [ str(int(m)) for m in line.split('\t')]
            edge = '_'.join(edge)
            edges[edge] = 1

    # load directed graph
    with open('{}/com-{}.all.cmty.txt'.format(dataset, dataset), 'r') as f: 
        for line in tqdm(f.readlines()):
            members = line.split('\t')
            members = [ int(m) for m in members]
            group_id = len(group2member)
            group2member[group_id] = members
            for m in members:
                member2group[m].append(group_id)

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

    stats = defaultdict(int)
    group_size = defaultdict(int)
    group_num = 0
    for group_id, members in tqdm(group2member.items()):
        for idx, src in enumerate(members):
            for jdx, dst in enumerate(members[idx+1:]):
                distance = nx.shortest_path_length(G, source=src, target=dst)
                stats[distance] += 1
        group_size[len(members)] += 1
        group_num += 1
        if group_num > 1000:
            break
    fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(15,5)) # two axes on figure
    fig2.suptitle(dataset)
    data = {
        'x': [ key for key, _ in stats.items() ],
        'y': [ size for _, size in stats.items() ],
    }
    ax1.set_title('group members shortest distance')
    sns.lineplot(x='x', y='y', data=data, ax=ax1)
    data = {
        'x': [ key for key, _ in group_size.items() ],
        'y': [ size for _, size in group_size.items() ],
    }
    ax2.set_title('group size')
    sns.lineplot(x='x', y='y', data=data, ax=ax2)
    fig2.savefig('{}.png'.format(dataset))



if __name__ == "__main__":
    for dataset in ['lj', 'orkut']:
        compute_dataset(dataset)
