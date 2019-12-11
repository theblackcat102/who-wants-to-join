import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import random
from .models import FactorizedEmbeddings
from .dataset import Meetupv1, Meetupv2, TOKENS
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# 46899
USER_MAX_SIZE = 1087928
GROUP_SIZE = 10000

class Deepset(nn.Module):

    def __init__(self, user_size, hidden_size, set_features=20):
        super(Deepset, self).__init__()
        self.embedding = FactorizedEmbeddings(user_size, hidden_size, hidden_size//2)
        # self.tag_embedding = nn.Embedding(tag_size, 30)

        self.extractor = nn.Sequential(
            nn.Linear(hidden_size, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, user_size),
        )

    def forward(self, users):
        output_users = self.embedding(users)
        latent = self.extractor(output_users)
        latent = latent.sum(1)
        return self.regressor(latent)

class Model(pl.LightningModule):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hparams = args
        self.user_size=1087928
        hidden_size=32
        set_features=512
        tag_size=966
        self.model = Deepset(self.user_size, hidden_size, set_features)
        self.l2 = nn.MSELoss()


    def training_step(self, batch, batch_idx):
        # REQUIRED
        existing_users, pred_users, pred_users_cnt, tags = batch        
        B = existing_users.shape[0]
        y_onehot = torch.FloatTensor(B, self.user_size)
        output = self.model(existing_users)
        y_onehot.zero_()
        y_onehot = y_onehot.to(pred_users.device)
        y_onehot.scatter_(1, pred_users, 1)
        loss = self.l2(output, y_onehot)
        tensorboard_logs = {'loss/train': loss.item(), 'norm_loss/train': loss.item()/B}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        existing_users, pred_users, pred_users_cnt, tags = batch        
        B = existing_users.shape[0]
        y_onehot = torch.FloatTensor(B, self.user_size)
        output = self.model(existing_users)
        y_onehot.zero_()
        y_onehot = y_onehot.to(pred_users.device)
        y_onehot.scatter_(1, pred_users, 1)
        loss = self.l2(output, y_onehot)
        # pred_user = torch.argmax(result.unsqueeze(-1), dim=2)
        # pred_users 
        # print(decoder_outputs.shape)

        return {'val_loss': loss }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_n_loss = np.array([x['norm_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        tensorboard_logs = {
            'loss/val': avg_loss.item(), 
            # 'avg_acc/val': avg_acc, 
            'val_loss': avg_loss.item(), 
            # 'norm_loss/train': avg_n_loss.item()
        }

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        self.dataset = Meetupv2(train=True, 
            sample_ratio=self.hparams.sample_ratio, max_size=GROUP_SIZE, query=self.hparams.query)
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            batch_size=self.hparams.batch_size, num_workers=4, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        self.dataset = Meetupv2(train=False, 
            sample_ratio=self.hparams.sample_ratio, max_size=GROUP_SIZE, query=self.hparams.query)
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            batch_size=8, num_workers=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MoSAN Group Recommendation Model')
    parser.add_argument('--dataset', type=str, default='meetup')
    parser.add_argument('--query', type=str, default='group')
    parser.add_argument('--model', type=str, default='deepset')
    parser.add_argument('--user-dim', type=int, default=50)
    parser.add_argument('--max-epochs', type=int, default=40)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sample-ratio', type=float, default=0.8)
    parser.add_argument('--max-group', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    model = Model(args)

    trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=1.0, 
        gpus=[args.gpu])
    trainer.fit(model)