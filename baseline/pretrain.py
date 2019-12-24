import numpy as np
import networkx as nx
from node2vec import Node2Vec
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepset Recommendation Model')
    parser.add_argument('--dataset', type=str, default='amazon', choices=['amazon', 'orkut', 'lj','friendster', 'youtube'])
    parser.add_argument('--output', type=str, default='amazon', choices=['amazon', 'orkut', 'lj','friendster', 'youtube'])
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--walk-length', type=int, default=30)
    parser.add_argument('--num-walk', type=int, default=200)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--batch-words', type=int, default=4)

    args = parser.parse_args()

    community_filename = '{}/com-{}.all.cmty.txt'.format(args.dataset, args.dataset)
    edge_filename = '{}/com-{}.ungraph.txt'.format(args.dataset, args.dataset)
    if not os.path.exists(edge_filename):
        raise ValueError('Edge relation file not found')
    if not os.path.exists(edge_filename):
        raise ValueError('Edge relation file not found')
    graph = nx.Graph()
    print('create graph connection')
    with open(edge_filename, 'r') as f:
        for line in f.readlines():
            if '#' not in line:
                edge = [ int(e) for e in line.strip().split('\t')][:2]
                graph.add_edge(*edge)
    print('start node2vec training')    
    node2vec = Node2Vec(graph, 
        dimensions=args.dimension, 
        walk_length=args.walk_length, 
        num_walks=args.num_walk, 
        workers=args.workers,
        temp_folder='./tmp')  # Use temp_folder for big graphs

    model = node2vec.fit(window=args.window, min_count=args.min_count, batch_words=args.batch_words)
    model.save('{}/embed.emb'.format(args.output))
