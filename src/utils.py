from itertools import islice
from tqdm import tqdm
import torch
import argparse
from networkx import nx
import numpy as np
from torch_geometric.data import Dataset, Data

TMP_WRITER_PATH = "tmp_writer"


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and
        `truth` tensors, i.e. the amount of positions where the values of
        `prediction` and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    return true_positives, false_positives, true_negatives, false_negatives


def dict2table(params):
    if not isinstance(params, dict):
        params = vars(params)
    text = '\n\n'
    text = '|  Attribute  |     Value    |\n'+'|'+'-'*13+'|'+'-'*14+'|'
    for key, value in params.items():
        text += '\n|{:13}|{:14}|'.format(str(key), str(value))
    return text


def pbar_listener(q, total_size):
    pbar = tqdm(total=total_size, dynamic_ncols=True)
    while True:
        item = q.get()
        if item is None:
            break
        pbar.update(item)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def chunks(data, size=10000):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}



def nx_to_graph_data_obj(g, center_id, allowable_features_downstream=None,
                         allowable_features_pretrain=None,
                         node_id_to_go_labels=None):
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    # nodes
    nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
    # in a particular ordering. Will be used as a mapping to convert
    # between nx node ids and data obj node indices
    x = torch.from_numpy(np.array([ g.nodes[n_i]['value'] for n_i in g.nodes() ]))
    center_node_idx = nx_node_ids.index(center_id)
    center_node_idx = torch.tensor([center_node_idx], dtype=torch.long)    
    edges_list = []
    for node_1, node_2 in g.edges():
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edges_list.append((j, i))

    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)    
    data = Data(x=x, edge_index=edge_index, **g.graph)

    data.center_node_idx = center_node_idx
    if node_id_to_go_labels:  # supervised case with go node labels
        # Construct a dim n_pretrain_go_classes tensor and a
        # n_downstream_go_classes tensor for the center node. 0 is no data
        # or negative, 1 is positive.
        downstream_go_node_feature = [0] * len(allowable_features_downstream)
        pretrain_go_node_feature = [0] * len(allowable_features_pretrain)
        if center_id in node_id_to_go_labels:
            go_labels = node_id_to_go_labels[center_id]
            # get indices of allowable_features_downstream that match with elements
            # in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_downstream, go_labels, return_indices=True)
            for idx in node_feature_indices:
                downstream_go_node_feature[idx] = 1
            # get indices of allowable_features_pretrain that match with
            # elements in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_pretrain, go_labels, return_indices=True)
            for idx in node_feature_indices:
                pretrain_go_node_feature[idx] = 1
        data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature),
                                        dtype=torch.long)
        data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature),
                                        dtype=torch.long)

    return data

def graph_data_obj_to_nx(data):
    """
    Converts pytorch geometric Data obj to network x data object.
    :param data: pytorch geometric Data object
    :return: nx graph object
    """
    G = nx.Graph()
    # edges
    edge_index = data.edge_index.cpu().numpy()
    n_edges = edge_index.shape[1]
    x = data.x.cpu().numpy()


    for j in range(0, n_edges):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_node(begin_idx):
            G.add_node(begin_idx, value=x[begin_idx])

        if not G.has_node(end_idx):
            G.add_node(end_idx, value=x[end_idx])

        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)

    # # add center node id information in final nx graph object
    # nx.set_node_attributes(G, {data.center_node_idx.item(): True}, 'is_centre')

    G.graph['label_mask'] = data.label_mask
    if hasattr(data, 'titleid'):
        G.graph['titleid'] = data.titleid
    G.graph['y'] = data.y
    return G

def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


def calculate_f_score(i, j):
    i = [int(x) for x in i]
    j = [int(x) for x in j]
    inter=set(i).intersection(set(j))
    precision=len(inter)/float(len(j))
    recall=len(inter)/float(len(i))
    if recall==0 and precision==0:
        fscore=0
    else:
        fscore=2*(precision*recall)/(precision+recall)
    return fscore, precision, recall
