import numpy as np
import networkx as nx
from node2vec import Node2Vec


def pretrain_model(graph):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepset Recommendation Model')
    parser.add_argument('--dataset', type=str, default='amazon', choices=['amazon', 'orkut', 'lj','friendster', 'youtube'])
    args = parser.parse_args()

    community_filename = '{}/com-{}.all.cmty.txt'.format(args.dataset, args.dataset)
    edge_filename = '{}/com-{}.ungraph.txt'.format(args.dataset, args.dataset)
    if not os.path.exists(edge_filename):
        raise ValueError('Edge relation file not found')
    if not os.path.exists(edge_filename):
        raise ValueError('Edge relation file not found')
    FG = nx.Graph()