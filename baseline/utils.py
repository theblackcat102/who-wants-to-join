import random
import torch
import numpy as np
import networkx as nx
from src.utils import nx_to_graph_data_obj, graph_data_obj_to_nx, reset_idxes


class ExtractSubstructureContextPair:
    def __init__(self, l1, center=True):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds the whole graph, and the
        context substructures that corresponds to
        the subgraph that is between l1 and the edge of the graph. If
        center=True, then will select the center node as the root node.
        :param center: True, will select a center node as root node, otherwise
        randomly selects a node
        :param l1:
        """
        self.center = center
        self.l1 = l1

        if self.l1 == 0:
            self.l1 = -1

    def __call__(self, data, root_idx=None):
        """
        :param data: pytorch geometric data object
        :param root_idx: Usually None. Otherwise directly sets node idx of
        root (
        for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """

        G = graph_data_obj_to_nx(data)
        num_atoms = len(G.nodes)
        if root_idx == None:
            if self.center == True:
                root_idx = data.center_node_idx.item()
            else:
                root_idx = random.sample(range(num_atoms), 1)[0]

        # in the PPI case, the subgraph is the entire PPI graph
        data.x_substruct = data.x
        data.edge_attr_substruct = data.edge_attr
        data.edge_index_substruct = data.edge_index
        data.center_substruct_idx = 0


        # Get context that is between l1 and the max diameter of the PPI graph
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        # l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
        #                                                       self.l2).keys()
        l2_node_idxes = range(num_atoms)
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, other data obj does not
            # make sense
            context_data = nx_to_graph_data_obj(context_G, 0)   # use a dummy
            # center node idx
            data.x_context = context_data.x
            data.edge_index_context = context_data.edge_index
        else:
            return None

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)
        else:
            return None

        return data

    def __repr__(self):
        return '{}(l1={}, center={})'.format(self.__class__.__name__,
                                              self.l1, self.center)

if __name__ == "__main__":
    from src.meetup import Meetup
    extract = ExtractSubstructureContextPair(2, center=False)
    meetup = Meetup(transform=extract)
    for data in meetup:
        print(data)