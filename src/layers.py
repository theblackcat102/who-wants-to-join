import torch
from torch_geometric.nn import GCNConv, SAGEConv,GATConv
import torch.nn as nn


class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(
            self, user_size, input_channels, output_channels, layers=[32, 32],
            dropout=0.1):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features.
        """
        super(StackedGCN, self).__init__()
        self.embeddings = nn.Embedding(user_size, input_channels)

        self.known_embeddings = nn.Embedding(2, input_channels)

        self.layers_dim = layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.layers = []
        self.layers_dim = ([self.input_channels] +
                           self.layers_dim +
                           [self.output_channels])
        for i, _ in enumerate(self.layers_dim[:-2]):
            self.layers.append(
                GCNConv(self.layers_dim[i], self.layers_dim[i+1]))
        self.layers.append(GCNConv(self.layers_dim[-2], self.layers_dim[-1]))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        features_ = self.embeddings(features[:, 0])
        known_feat = self.known_embeddings(features[:, 1])
        features = known_feat + features_

        for i, _ in enumerate(self.layers[:-2]):
            features = nn.functional.relu(self.layers[i](features, edges))
            if i > 1:
                features = nn.functional.dropout(
                    features, p=self.dropout, training=self.training)
        features = self.layers[-1](features, edges)
        # predictions = torch.nn.functional.log_sigmoid(features, dim=1)
        return features

class StackedGCNYahoo(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(
            self, user_size=999744+1, group_size=638124+1,
            user_dim=8, group_dim=4,
            input_channels=8, output_channels=1, layers=[16, 16],
            dropout=0.1):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features.
        """
        super(StackedGCNYahoo, self).__init__()
        self.embeddings = nn.Embedding(user_size, user_dim)
        self.known_embeddings = nn.Embedding(2, user_dim)
        self.user_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(user_dim, input_channels)
        )
        self.group_embeddings = nn.Embedding(group_size, group_dim)
        self.group_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(group_dim, input_channels)
        )

        self.layers_dim = layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.layers = []
        self.layers_dim = ([self.input_channels] +
                           self.layers_dim +
                           [self.output_channels])
        for i, _ in enumerate(self.layers_dim[:-2]):
            self.layers.append(
                GCNConv(self.layers_dim[i], self.layers_dim[i+1]))
        self.layers.append(GCNConv(self.layers_dim[-2], self.layers_dim[-1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """

        user_feature_idx = features[ features[:, -1] == 0, 0 ]
        known_user_idx = features[ features[:, -1] == 0, 1 ]

        category_idx = features[ features[:, -1] == 1, 0]
        user_feature = self.embeddings(user_feature_idx )
        known_feat = self.known_embeddings(known_user_idx)
        user_feature = self.user_proj(user_feature + known_feat)

        group_feature = self.group_proj(self.group_embeddings(category_idx))

        new_features = torch.zeros((len(features), self.input_channels)).cuda()
        new_features[ features[:, -1] == 0 ] = user_feature
        new_features[ features[:, -1] == 1 ] = group_feature
        features = new_features

        for i, _ in enumerate(self.layers[:-2]):
            features = nn.functional.relu(self.layers[i](features, edges))
            if i > 1:
                features = nn.functional.dropout(
                    features, p=self.dropout, training=self.training)
        features = self.layers[-1](features, edges)
        return features

class StackedGCNAmazon(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(
            self, user_size, category_size,
            user_dim=8, category_dim=4,
            input_channels=8, output_channels=16, layers=[16, 16],
            dropout=0.1):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features.
        """
        super(StackedGCNAmazon, self).__init__()

        # print(user_size, category_size, topic_size, group_size)
        # print(user_dim, category_dim, topic_dim, group_dim)

        self.embeddings = nn.Embedding(user_size, user_dim)
        self.known_embeddings = nn.Embedding(2, user_dim)
        self.user_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(user_dim, input_channels)
        )
        self.category_embeddings = nn.Embedding(category_size, category_dim)
        self.category_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(category_dim, input_channels)
        )

        self.layers_dim = layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.layers = []
        self.layers_dim = ([self.input_channels] +
                           self.layers_dim +
                           [self.output_channels])
        for i, _ in enumerate(self.layers_dim[:-2]):
            self.layers.append(
                GCNConv(self.layers_dim[i], self.layers_dim[i+1]))
        self.layers.append(GCNConv(self.layers_dim[-2], self.layers_dim[-1]))
        self.layers = nn.ModuleList(self.layers)
        self.predict_member = nn.Linear(self.output_channels, 1)

        self.predict_node = nn.Linear(self.output_channels, 2)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """

        user_feature_idx = features[ features[:, -1] == 0, 0 ]
        known_user_idx = features[ features[:, -1] == 0, 1 ]

        category_idx = features[ features[:, -1] == 1, 0]

        user_feature = self.embeddings(user_feature_idx )
        known_feat = self.known_embeddings(known_user_idx)
        user_feature = self.user_proj(user_feature + known_feat)

        category_feature = self.category_proj(self.category_embeddings(category_idx))

        new_features = torch.zeros((len(features), self.input_channels)).cuda()
        new_features[ features[:, -1] == 0 ] = user_feature
        new_features[ features[:, -1] == 1 ] = category_feature
        features = new_features

        for i, _ in enumerate(self.layers[:-2]):
            features = nn.functional.relu(self.layers[i](features, edges))
            if i > 1:
                features = nn.functional.dropout(
                    features, p=self.dropout, training=self.training)
        features = nn.functional.relu(self.layers[-1](features, edges))

        # predictions = torch.nn.functional.log_sigmoid(features, dim=1)
        node_pred = self.predict_node(features)
        member_pred = self.predict_member(features)
        return member_pred, node_pred

class StackedGCNDBLP(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(
            self, author_size, paper_size, conf_size,
            user_dim=8, paper_dim=4, conf_dim=4,
            input_channels=8, output_channels=1, layers=[16, 16],
            dropout=0.1):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features.
        """
        super(StackedGCNDBLP, self).__init__()

        # print(user_size, category_size, topic_size, group_size)
        # print(user_dim, category_dim, topic_dim, group_dim)

        self.embeddings = nn.Embedding(author_size, user_dim)
        self.known_embeddings = nn.Embedding(2, user_dim)
        self.user_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(user_dim, input_channels)
        )
        self.conf_embeddings = nn.Embedding(conf_size, conf_dim)
        self.conf_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(conf_dim, input_channels)
        )
        self.paper_embeddings = nn.Embedding(paper_size, paper_dim)
        self.paper_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(paper_dim, input_channels)
        )

        self.layers_dim = layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.layers = []
        self.layers_dim = ([self.input_channels] +
                           self.layers_dim +
                           [self.output_channels])
        for i, _ in enumerate(self.layers_dim[:-2]):
            self.layers.append(
                GCNConv(self.layers_dim[i], self.layers_dim[i+1]))
        self.layers.append(GCNConv(self.layers_dim[-2], self.layers_dim[-1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """

        author_node_idx = features[:, -1] == 0
        paper_node_idx = features[:, -1] == 1
        conf_node_idx = features[:, -1] == 2

        authors_idx = features[ author_node_idx, 0 ]
        known_user_idx = features[ author_node_idx, 1 ]
        paper_idx = features[ paper_node_idx, 0]
        conf_idx = features[ conf_node_idx, 0]
        # 576192        1956554         6620
        # 874608        3605603        12770
        # print(authors_idx.max(), paper_idx.max(), conf_idx.max())
        user_feature = self.embeddings(authors_idx )
        known_feat = self.known_embeddings(known_user_idx)
        author_feature = self.user_proj(user_feature + known_feat)

        paper_feature = self.paper_proj(self.paper_embeddings(paper_idx))
        conf_feature = self.conf_proj(self.conf_embeddings(conf_idx))

        new_features = torch.zeros((len(features), self.input_channels)).cuda()
        new_features[ author_node_idx ] = author_feature
        new_features[ paper_node_idx ] = paper_feature
        new_features[ conf_node_idx ] = conf_feature
        features = new_features

        for i, _ in enumerate(self.layers[:-2]):
            features = nn.functional.relu(self.layers[i](features, edges))
            if i > 1:
                features = nn.functional.dropout(
                    features, p=self.dropout, training=self.training)
        features = self.layers[-1](features, edges)
        # predictions = torch.nn.functional.log_sigmoid(features, dim=1)
        return features
class StackedGCNMeetup(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(
            self, user_size, 
            category_size, 
            topic_size, 
            group_size, 
            user_dim=8, category_dim=2, topic_dim=8, group_dim=8,
            input_channels=8, output_channels=1, layers=[16, 16],
            dropout=0.1):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features.
        """
        super(StackedGCNMeetup, self).__init__()

        # print(user_size, category_size, topic_size, group_size)
        # print(user_dim, category_dim, topic_dim, group_dim)

        self.embeddings = nn.Embedding(user_size, user_dim)
        self.known_embeddings = nn.Embedding(2, user_dim)
        self.user_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(user_dim, input_channels)
        )
        self.category_embeddings = nn.Embedding(category_size, category_dim)
        self.category_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(category_dim, input_channels)
        )
        self.topic_embeddings = nn.Embedding(topic_size, topic_dim)
        self.topic_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(topic_dim, input_channels)
        )

        self.group_embeddings = nn.Embedding(group_size, group_dim)
        self.group_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(group_dim, input_channels)
        )

        self.layers_dim = layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.layers = []
        self.layers_dim = ([self.input_channels] +
                           self.layers_dim +
                           [self.output_channels])
        for i, _ in enumerate(self.layers_dim[:-2]):
            self.layers.append(
                GCNConv(self.layers_dim[i], self.layers_dim[i+1]))
        self.layers.append(GCNConv(self.layers_dim[-2], self.layers_dim[-1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """

        user_feature_idx = features[ features[:, -1] == 0, 0 ]
        known_user_idx = features[ features[:, -1] == 0, 1 ]
        topic_idx = features[ features[:, -1] == 1, 0 ]

        group_idx = features[ features[:, -1] == 4, 0]

        category_idx = features[ features[:, -1] == 2, 0]

        user_feature = self.embeddings(user_feature_idx )
        known_feat = self.known_embeddings(known_user_idx)
        user_feature = self.user_proj(user_feature + known_feat)

        topic_feature = self.topic_proj(self.topic_embeddings(topic_idx))
        group_feature = self.group_proj(self.group_embeddings(group_idx))
        category_feature = self.category_proj(self.category_embeddings(category_idx))

        new_features = torch.zeros((len(features), self.input_channels)).cuda()
        new_features[ features[:, -1] == 0 ] = user_feature
        new_features[ features[:, -1] == 1 ] = topic_feature
        new_features[ features[:, -1] == 2 ] = category_feature
        new_features[ features[:, -1] == 4 ] = group_feature
        features = new_features

        for i, _ in enumerate(self.layers[:-2]):
            features = nn.functional.relu(self.layers[i](features, edges))
            if i > 1:
                features = nn.functional.dropout(
                    features, p=self.dropout, training=self.training)
        features = self.layers[-1](features, edges)
        # predictions = torch.nn.functional.log_sigmoid(features, dim=1)
        return features


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


if __name__ == "__main__":
    model = StackedGCN(39687, 16, 1, [16, 16, 16], 0.1)
