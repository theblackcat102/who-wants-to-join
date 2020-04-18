from dataset.aminer import Aminer
from dataset.meetup import Meetup,locations_id
import numpy as np
import pickle
from sklearn.metrics import f1_score
from src.utils import dict2table, np_confusion, str2bool

def similarity_evaluation(dataset, embeddings, user_node_id, top_k=20, user_size = 399212):
    name2id = embeddings['name2id']
    embedding = embeddings['embedding']
    precisions, recalls = [], []
    max_node_idx=  0

    for data in dataset:

        # extract user node with known before
        known_user_node_id = (data.x[:, 2] == user_node_id) & (data.x[:, 1] == 1)
        known_nodes = [ '{}_{}'.format(int(node[-1]), int(node[0])) for node in data.x[known_user_node_id, :] ]

        # known_paper_node_id = (data.x[:, 2] == 1)
        # known_nodes += [ '{}_{}'.format(int(node[-1]), int(node[0])) for node in data.x[known_user_node_id, :] ]

        # print(known_nodes)
        user_embeddings = np.array([  embedding[name2id[nameid]] for nameid in known_nodes if nameid in name2id ])

        candidate_user_node_id = (data.x[:, 2] == user_node_id) & (data.x[:, 1] != 1)
        target_node_id = [ int(node[0]) for node in data.x[data.y == 1, :]]
        max_node_idx = max([ max_node_idx ] + target_node_id)

        candidate_node_id = [ int(node[0]) for node in data.x[candidate_user_node_id, :]]
        max_node_idx = max([ max_node_idx ] + candidate_node_id)
        candidate_nodes = [ '{}_{}'.format(int(node[-1]), int(node[0])) for node in data.x[candidate_user_node_id, :] ]
        candidate_nodes = [ '{}_{}'.format(int(node[-1]), int(node[0])) for node in data.x[candidate_user_node_id, :] ]
        candidate_user_embeddings = np.array([  embedding[name2id[nameid]] for nameid in candidate_nodes if nameid in name2id ])
        if len(user_embeddings) > 0 and len(candidate_user_embeddings) > 0:
            norm_embeddings = user_embeddings.sum(0)/len(user_embeddings)
            norm_embeddings = np.linalg.norm(norm_embeddings)
            candidate_user_embeddings = np.linalg.norm(candidate_user_embeddings, axis=1)
            dot_prod = candidate_user_embeddings.dot(norm_embeddings)
            rank = [ (candidate_node_id[idx], weight)  for idx, weight in enumerate(dot_prod) ]
            rank.sort(key=lambda x: x[1])
            pred_nodes = [ ]
            if len(rank) < top_k:
                pred_nodes = [ pair[0] for pair in rank ]
            else:
                pred_nodes = [ rank[i][0] for i in range(top_k)]

            # print(pred_nodes, target_node_id)
            y_pred, y_target = np.zeros(user_size), np.zeros(user_size)
            y_pred[pred_nodes] = 1.0
            y_target[target_node_id] = 1.0

            TP, FP, TN, FN = np_confusion(y_pred, y_target)

            recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
            precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
            precisions.append(precision)
            recalls.append(recall)

    avg_recalls = np.mean(recalls)
    avg_precisions = np.mean(precisions)
    f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)
    print(max_node_idx)
    print(f1, avg_recalls, avg_precisions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='CF rank method for group expansion')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--city', type=str, default='SF',
                        choices=['NY', 'SF'])
    parser.add_argument('--dataset', type=str, default='aminer',
                        choices=['meetup', 'aminer'])
    parser.add_argument('--user-node', type=int, default=0, 
                        help='integer which user node id is represented in')
    parser.add_argument('--user-size', type=int, default=399212, 
                        help='maximum user node id')

    parser.add_argument('--embeddings', type=str,
                        help='graphvite embedding pickle')
    args = parser.parse_args()

    if args.dataset == 'aminer':
        dataset = Aminer()
    else:
        dataset = Meetup(city_id=locations_id[args.city])
    data_size = len(dataset)
    train_split, val, test = int(data_size*0.7), int(data_size*0.1), int(data_size*0.2)
    indexes = np.array(list(range(data_size)), dtype=np.long)[train_split+val:]
    print(indexes[:10])

    val_dataset = dataset[list(indexes)]

    with open(args.embeddings, 'rb') as f:
        embeddings = pickle.load(f)

    similarity_evaluation(val_dataset, embeddings,
        user_node_id=args.user_node, user_size=args.user_size,
        top_k=args.top_k)
