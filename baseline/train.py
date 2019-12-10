import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import Meetupv1, Meetupv2, TOKENS
from .models import Seq2Seq
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils import clip_grad_norm


class Skipgram(nn.Module):
    def __init__(self, user_size=46895, user_dim=32):
        super(Skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(user_size, user_dim)   
        self.v_embeddings = nn.Embedding(user_size+3, user_dim) 
        self.user_dim = user_dim
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.user_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, u_pos, v_pos, batch_size=32):
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)

        score  = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

        # neg_embed_v = self.v_embeddings(v_neg)

        # neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        # neg_score = torch.sum(neg_score, dim=1)
        # sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target# + sum_log_sampled

        return -1*loss.sum()/batch_size

class BoW(nn.Module):
    def __init__(self, user_dim=32, user_size=46895):
        super(BoW, self).__init__()
        self.user_embed = nn.Embedding(user_size+1, user_dim, padding_idx=0, sparse=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pred = nn.Sequential(
            nn.BatchNorm1d(user_dim),
            nn.ReLU(),
            nn.Linear(user_dim, 256),
            nn.ReLU(),
            nn.Linear(256, user_size+1),
        )

    def forward(self, exist_user):
        users_latent = self.user_embed(exist_user)
        # print(users_latent.shape)
        h = self.pool(users_latent.transpose(1, 2))
        return self.pred(h.squeeze(2))


class Baseline(pl.LightningModule):

    def __init__(self, args):
        super(Baseline, self).__init__()
        # not the best model...
        self.hparams = args
        self.max_group = args.max_group
        self.criterion = nn.CrossEntropyLoss(ignore_index=TOKENS['PAD'])
        self.model = Seq2Seq(
            user_size=1087928,
            hidden_size=args.user_dim,
        )
        # self.skip_gram = Skipgram()

    def training_step(self, batch, batch_idx):
        # REQUIRED
        existing_users, pred_users, pred_users_cnt, tags = batch
        existing_users = existing_users.transpose(0, 1)
        pred_users = pred_users.transpose(0, 1)
        tags = tags.transpose(0, 1)
        
        loss, norm_loss, _ = self.model(existing_users, pred_users,tags, self.criterion, 
            teacher_forcing_ratio=0.8, device='cuda')
        
        tensorboard_logs = {'loss/train': loss.item(), 'norm_loss/train': norm_loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        existing_users, pred_users, pred_users_cnt, tags = batch
        existing_users = existing_users.transpose(0, 1)
        pred_users = pred_users.transpose(0, 1)
        tags = tags.transpose(0, 1)

        
        loss, norm_loss, decoder_outputs = self.model(existing_users, pred_users,tags, self.criterion, 
            teacher_forcing_ratio=1.0, device='cuda')

        argmax = torch.argmax(decoder_outputs, dim=-1)
        invalid_targets = pred_users.eq(TOKENS['PAD'])
        accuracy = argmax.eq(pred_users).masked_fill_(invalid_targets, 0)\
            .float().sum()/pred_users_cnt.sum()

        # print(decoder_outputs.shape)

        return {'val_loss': loss, 'accuracy': accuracy, 'norm_loss': norm_loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_n_loss = np.array([x['norm_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        tensorboard_logs = {
            'loss/val': avg_loss.item(), 
            'avg_acc/val': avg_acc, 
            'val_loss': avg_loss.item(), 
            'norm_loss/train': avg_n_loss.item()
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
            sample_ratio=self.hparams.sample_ratio, max_size=self.max_group, query=self.hparams.query)
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            batch_size=self.hparams.batch_size, num_workers=10, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        self.dataset = Meetupv2(train=False, 
            sample_ratio=self.hparams.sample_ratio, max_size=self.max_group, query=self.hparams.query)
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            batch_size=8, num_workers=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MoSAN Group Recommendation Model')
    parser.add_argument('--dataset', type=str, default='meetup')
    parser.add_argument('--query', type=str, default='group')
    parser.add_argument('--user-dim', type=int, default=50)
    parser.add_argument('--max-epochs', type=int, default=40)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--sample_ratio', type=float, default=0.8)
    parser.add_argument('--max-group', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    model = Baseline(args)
    trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=1.0, 
        gpus=[args.gpu], gradient_clip_val=args.clip_grad)
    trainer.fit(model)