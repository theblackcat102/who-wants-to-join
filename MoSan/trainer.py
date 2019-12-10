import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .models import MoSAN, bpr_loss
from .dataloader import Meetup, seq_collate
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class MeetupMoSAN(pl.LightningModule):

    def __init__(self, args):
        super(MeetupMoSAN, self).__init__()
        # not the best model...
        self.hparams = args
        self.max_group = args.max_group

        self.model = MoSAN(group=args.max_group, 
            context_dim=args.embed_dim,
            hidden_dim=args.embed_dim,
            user_dim=args.user_dim)

    def training_step(self, batch, batch_idx):
        # REQUIRED

        neg_venue = batch['neg_venue']
        pos_venue = batch['venue']
        context_users = batch['context_users']
        interactions = batch['interactions']

        pos, neg = self.model(context_users, interactions,
            pos_venue, neg_venue )
        loss = bpr_loss(pos, neg)
        tensorboard_logs = {'train_loss': loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        neg_venue = batch['neg_venue']
        pos_venue = batch['venue']
        context_users = batch['context_users']
        interactions = batch['interactions']
        # print(context_users.shape, interactions.shape)
        pos, neg = self.model(context_users, interactions,
            pos_venue, neg_venue )
        loss = bpr_loss(pos, neg)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss.item()}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.regularizer)

    @pl.data_loader
    def train_dataloader(self):
        self.dataset = Meetup(train=True, sample_k=self.max_group)
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            collate_fn=seq_collate, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        self.dataset = Meetup(train=False, sample_k=self.max_group)
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            collate_fn=seq_collate, batch_size=8, num_workers=8)

    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MoSAN Group Recommendation Model')
    parser.add_argument('--dataset', type=str, default='meetup')
    parser.add_argument('--embed-dim', type=int, default=50)
    parser.add_argument('--user-dim', type=int, default=50)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--min-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--regularizer', type=float, default=0.01)
    parser.add_argument('--max-group', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    model = MeetupMoSAN(args)
    trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=1.0, 
        gpus=[args.gpu], )
    trainer.fit(model)