import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import argparse
import math
import random
from .test import load_params, extract_checkpoint_files
from .models import FactorizedEmbeddings
from .dataset import SocialDataset, AMinerDataset, TOKENS, seq_collate
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from .set_classifier import PermEqui1_max, PermEqui2_max, PermEqui2_mean, PermEqui1_mean
from .utils import confusion, str2bool
# from .sparse_binary_loss import SparseBinaryWithLogits
from pytorch_lightning.callbacks import ModelCheckpoint
# from .random_sampler import ImbalancedDatasetSampler


class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()
        self.kl_loss = nn.KLDivLoss()
        self.log_sigmoid = nn.LogSigmoid()
    def forward(self, y_pred, y_true):
        return self.kl_loss(self.log_sigmoid(y_pred), self.log_sigmoid(y_true))


class Deepset(nn.Module):

    def __init__(self, vocab_size, hidden_size, set_features=1024, pool='max1'):
        super(Deepset, self).__init__()
        # self.embedding = FactorizedEmbeddings(vocab_size, hidden_size, hidden_size//2)
        pool_functions = {
            'max': PermEqui2_max,
            'max1': PermEqui1_max,
            'mean': PermEqui2_mean,
            'mean1': PermEqui1_mean
        }
        proj_fn = pool_functions[pool]
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        self.extractor = nn.Sequential(
            nn.Dropout(0.1),
            proj_fn(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            proj_fn(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(512),
            proj_fn(hidden_size, set_features),
            nn.ELU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(set_features, hidden_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, vocab_size),
            # nn.Sigmoid(),
        )

    def forward(self, users):
        output_users = self.embeddings(users)
        latent = self.extractor(output_users)
        latent, _ = latent.max(1)
        return self.regressor(latent)#, self.reconstruct(latent)

class Model(pl.LightningModule):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hparams = args
        self.max_group = args.max_group
        self.dataset_class = AMinerDataset
        if args.task == 'socnet':
            self.dataset_class = SocialDataset

        self.train_dataset = self.dataset_class(train=True, 
            order_shuffle=args.order_shuffle,
            sample_ratio=self.hparams.sample_ratio, max_size=args.max_group, dataset=args.dataset,
            query=self.hparams.query, min_freq=args.freq)


        stats = self.train_dataset.get_stats()
        self.user_size=stats['member']+3
        set_features=args.feature
        if self.train_dataset.embedding is not None:
            args.hidden = self.train_dataset.embedding.shape[-1]

        self.model = Deepset(self.user_size, args.hidden, set_features, pool=args.pool)
        if self.train_dataset.embedding is not None:
            print('Use pretrained graph embedding')
            embedding_weight = torch.from_numpy(self.train_dataset.embedding)
            self.model.embeddings.from_pretrained(embedding_weight)
            self.model.embeddings.weight.requires_grad=False

        pos_weight = torch.ones([self.user_size])*((self.user_size//args.freq)//700)
        self.l2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.l2 = nn.MultiLabelSoftMarginLoss(weight=pos_weight)#nn.MSELoss()


    def training_step(self, batch, batch_idx):

        existing_users, pred_users, pred_users_cnt = batch        
        B = existing_users.shape[0]
        output = self.model(existing_users)
        # y_onehot = torch.FloatTensor(B, self.user_size)
        y_onehot = torch.randn((B, self.user_size))*0.05
        y_onehot = y_onehot.to(pred_users.device)
        y_onehot.scatter_(1, pred_users, 1)
        y_onehot[:, :4] = 0.0
        y_onehot_input = torch.FloatTensor(B, self.user_size)
        y_onehot_input.zero_()
        y_onehot_input = y_onehot_input.to(pred_users.device)
        y_onehot_input.scatter_(1, existing_users, 1)
        y_onehot_input[:, :4] = 0.0

        loss = self.l2(output, y_onehot)# + self.l2(exists_output, y_onehot_input)
        tensorboard_logs = {'Loss/train': loss.item(), 'norm_loss/train': loss.item()/B}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        existing_users, pred_users, pred_users_cnt = batch        

        B = existing_users.shape[0]
        y_onehot = torch.FloatTensor(B, self.user_size)
        output = self.model(existing_users)
        y_onehot.zero_()

        y_onehot = y_onehot.to(pred_users.device)
        y_onehot.scatter_(1, pred_users, 1)
        y_onehot[:, :4] = 0.0
        pred_labels = ( torch.sigmoid(output) > 0.5 ).long()

        TP, FP, TN, FN = confusion(pred_labels, y_onehot)

        recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
        precision =  0 if (TP+FP) < 1e-5 else TP/(TP+FP)

        if (recall +precision) < 1e-5:
            f1 = 0
        else:
            f1 = 2*(recall*precision)/(recall+precision)

        if np.isnan([f1, recall, precision]).any() or np.isnan(f1):
            recall, precision, f1 = 0, 0, 0

        loss = self.l2(output, y_onehot)

        return {'val_loss': loss, 'f1': f1, 'recall': recall, 'precision': precision }


    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_n_loss = np.array([x['norm_loss'] for x in outputs]).mean()
        avg_acc = np.mean([x['f1'] for x in outputs if x['f1'] > 0])
        avg_per = np.mean([x['precision'] for x in outputs if x['precision'] > 0])
        avg_recall = np.mean([x['recall'] for x in outputs if x['recall'] > 0])
        tensorboard_logs = {
            'Loss/val': avg_loss.item(), 
            'Val/F1': avg_acc, 
            'Val/Recall': avg_recall,
            'Val/Precision': avg_per,
            'val_loss': avg_loss.item(), 
            # 'norm_loss/train': avg_n_loss.item()
        }
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.hparams.lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer]#, [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            # sampler=self.dist_sampler, 
            batch_size=self.hparams.batch_size, num_workers=10, shuffle=True, collate_fn=seq_collate)

    @pl.data_loader
    def val_dataloader(self):
        self.dataset = self.dataset_class(train=False,
            order_shuffle=self.hparams.order_shuffle,
            sample_ratio=self.hparams.sample_ratio, min_freq=self.hparams.freq,
            dataset=self.hparams.dataset, max_size=self.max_group, query=self.hparams.query)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            collate_fn=seq_collate,
            batch_size=self.hparams.batch_size, num_workers=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepset Recommendation Model')
    parser.add_argument('--query', type=str, default='group')
    parser.add_argument('--task', type=str, default='academic', choices=['socnet', 'academic'])
    parser.add_argument('--model', type=str, default='deepset')
    parser.add_argument('--freq', type=int, default=4, help='user exists minimal frequency')
    parser.add_argument('--order-shuffle', type=str2bool, default=True)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--feature', type=int, default=32)
    parser.add_argument('--pool', type=str, default='max1', choices=['max1', 'max', 'mean', 'mean1'])
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--min-epochs', type=int, default=100)
    parser.add_argument('--bce-weight', type=float, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sample-ratio', type=float, default=0.8)
    parser.add_argument('--dataset', type=str, default='acm', 
        choices=['dblp', 'acm', 'amazon', 'lj', 'friendster', 'orkut'])
    parser.add_argument('--max-group', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    model = Model(args)


    trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=0.2, 
        gpus=[args.gpu])
    trainer.fit(model)
