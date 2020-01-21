import torch
from torch_geometric.nn import GCNConv
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
        features = self.embeddings(features.squeeze(-1))
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
