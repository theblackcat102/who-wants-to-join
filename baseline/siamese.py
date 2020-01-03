import torch
import random
import numpy as np
import torch.nn as nn
from .blocks import SetAttentionBlock
from .blocks import InducedSetAttentionBlock
from .blocks import PoolingMultiheadAttention
from .models import FactorizedEmbeddings
from .set_transformer import RFF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def seq_collate(batches):
    max_exists_user = max([len(u[0]) for u in batches ])
    existing_users, pred_users, labels, cnts = [], [], [], []
    for batch in batches:
        if len(batch) == 4:
            tag = None
            existing_user, pred_user, label, pad_idx = batch
        else:
            existing_user, pred_user, cnt, tag, pad_idx = batch
        existing_users.append( np.array([pad_idx]*(max_exists_user - len(existing_user)) + existing_user))
        pred_users.append(pred_user)
        labels.append(label)
    pred_users = np.array(pred_users)
    pred_users = torch.from_numpy(np.array(pred_users)).long()
    existing_users = torch.from_numpy(np.array(existing_users)).long()
    labels = torch.from_numpy(np.array(labels)).long()

    return existing_users, pred_users, labels

class SiameseDataset(Dataset):

    def __init__(self, dataset, neg_sample=0.5):
        self.dataset = dataset
        self.neg_sample = neg_sample

    def __getitem__(self, idx):
        existing_users, pred_users, cnt, pad_idx = self.dataset[idx]
        label = 1.0
        if random.random() > self.neg_sample:
            label = 0
            # negative sampling
            rand_batch = random.choice(self.dataset)
            pred_users = rand_batch[1]
            return existing_users, random.choice(pred_users), label, pad_idx
        return existing_users, random.choice(pred_users), label, pad_idx

    def __len__(self):
        return len(self.dataset)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, negative_weight=1.0, positive_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight

    def forward(self, output1, output2, target, size_average=True):
        dist = (output2 - output1).pow(2).sum(1)  # squared dist

        losses = 0.5 * (target.float() * dist * self.positive_weight +
        (1 + -1 * target).float() * F.relu(self.margin - (dist + self.eps).sqrt() * self.negative_weight
            ).pow(2))
        return losses.mean() if size_average else losses.sum()

class SiameseSetTransformer(nn.Module):

    def __init__(self, user_size, heads=4, hidden=128, layers=2, pred_dim=32):
        """
        Arguments:
            in_dimension: an integer.
            out_dimension: an integer.
        """
        super(SiameseSetTransformer, self).__init__()
        d = hidden
        m = 16  # number of inducing points
        h = heads  # number of heads
        k = 4  # number of seed vectors
        self.embeddings = nn.Embedding(user_size, hidden)
        # if user_size > 1e6:
        #     self.embeddings = FactorizedEmbeddings(user_size, hidden, hidden//3)

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
        self.proj = nn.Linear(k * d, pred_dim)
        # self.proj2 = nn.Sequential(
        #     nn.Linear(hidden, 128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, pred_dim),
        # )
        self.proj2 = nn.Linear(hidden, pred_dim)
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.dropout = nn.Dropout(0.5)
        self.apply(weights_init)
    
    def group_latent(self, x):
        x = self.embeddings(x)  # shape [b, n, d]
        x = self.encoder(x)  # shape [b, n, d]
        X = self.dropout(x)
        x = self.decoder(x)  # shape [b, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)
        group_latent = self.proj(x)

        return group_latent

    def forward(self, x, y):
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

        y = self.embeddings(y)
        y = self.proj2(y)
        y = y.view(b, -1)

        group_latent = self.proj(x)

        return group_latent, y

if __name__ == "__main__":
    from .dataset import SocialDataset, AMinerDataset, TOKENS

    train_dataset = SiameseDataset(SocialDataset(train=True, 
            dataset='youtube',
            order_shuffle=False))
    model = SiameseSetTransformer(10000, )
    criterion = ContrastiveLoss(margin=1)
    
    dataloader = DataLoader(train_dataset,batch_size=16, collate_fn=seq_collate)

    for batch in dataloader:
        existing_users, pred_users, labels = batch
        output1, output2 = model(existing_users, pred_users)
        loss = criterion(output1, output2, labels)
        print(loss)
