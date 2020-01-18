import torch
from torch_geometric.nn import GCNConv

class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(self, args, input_channels, output_channels):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features. 
        """
        super(StackedGCN, self).__init__()
        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers = []
        self.args.layers = [self.input_channels] + self.args.layers + [self.output_channels]
        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers.append(GCNConv(self.args.layers[i],self.args.layers[i+1]))
        self.layers = ListModule(*self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        for i, _ in enumerate(self.args.layers[:-2]):
            features = torch.nn.functional.relu(self.layers[i](features, edges))
            if i>1:
                features = torch.nn.functional.dropout(features, p = self.args.dropout, training = self.training)
        features = self.layers[i+1](features, edges)
        predictions = torch.nn.functional.log_softmax(features, dim=1)
        return predictions



if __name__ == "__main__":
    model = StackedGCN(32, 32)
    
