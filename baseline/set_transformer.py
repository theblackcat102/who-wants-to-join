import torch.nn as nn
from .blocks import SetAttentionBlock
from .blocks import InducedSetAttentionBlock
from .blocks import PoolingMultiheadAttention
from .models import FactorizedEmbeddings


class SetTransformer(nn.Module):

    def __init__(self, user_size, heads=4, hidden=128, layers=2):
        """
        Arguments:
            in_dimension: an integer.
            out_dimension: an integer.
        """
        super(SetTransformer, self).__init__()
        d = hidden
        m = 16  # number of inducing points
        h = heads  # number of heads
        k = 4  # number of seed vectors
        self.embeddings = nn.Embedding(user_size, hidden)
        if user_size > 1e6:
            self.embeddings = FactorizedEmbeddings(user_size, hidden, hidden//3)

        layer = []
        for _ in range(layers):
            layer.append(
                InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
            )
        self.encoder = nn.Sequential(*layer)
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )
        self.predictor = nn.Linear(k * d, user_size)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.dropout = nn.Dropout(0.5)
        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dimension].
        Returns:
            a float tensor with shape [b, out_dimension].
        """

        x = self.embeddings(x)  # shape [b, n, d]
        x = self.encoder(x)  # shape [b, n, d]
        X = self.dropout(x)
        x = self.decoder(x)  # shape [b, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)
        return self.predictor(x)


class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)