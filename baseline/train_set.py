import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import math
from .test import load_params, extract_checkpoint_files
from .models import FactorizedEmbeddings
from .dataset import SocialDataset, AMinerDataset, TOKENS, seq_collate
from .utils import str2bool, confusion
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from .set_transformer import SetTransformer
from pytorch_lightning.callbacks import ModelCheckpoint


class Model(pl.LightningModule):
    def __init__(self, args):
        super(Model, self).__init__()

        self.max_group = args.max_group
        self.dataset_class = AMinerDataset
        if args.task == 'socnet':
            self.dataset_class = SocialDataset
        self.train_dataset = self.dataset_class(train=True, 
            order_shuffle=args.order_shuffle,
            sample_ratio=args.sample_ratio, max_size=args.max_group, dataset=args.dataset,
            query=args.query, min_freq=args.freq)
        stats = self.train_dataset.get_stats()
        self.user_size=stats['member']
        self.sample_ratio = args.sample_ratio
        self.ratio_raise = args.raising_ratio
        if self.train_dataset.embedding is not None:
            args.hidden = self.train_dataset.embedding.shape[-1]
        self.model = SetTransformer(user_size=self.user_size, 
            hidden=args.hidden, heads=args.heads, layers=args.enc_layer)
        if self.train_dataset.embedding is not None:
            print('Use pretrained graph embedding')
            embedding_weight = torch.from_numpy(self.train_dataset.embedding)
            self.model.embeddings.from_pretrained(embedding_weight)
            self.model.embeddings.weight.requires_grad=False

        pos_weight = torch.ones([self.user_size])*((self.user_size//args.freq)//200)
        self.l2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.hparams = args

    def training_step(self, batch, batch_idx):
        # if self.ratio_raise > 0:
        #     self.train_dataset.sample_rate = max(1.0 - self.trainer.current_epoch*self.ratio_raise, self.sample_ratio)
        sample_rate = self.train_dataset.sample_rate
        existing_users, pred_users, pred_users_cnt = batch
        B = existing_users.shape[0]
        y_onehot = torch.randn((B, self.user_size))*0.05

        output = self.model(existing_users)
        # y_onehot.zero_()
        y_onehot = y_onehot.to(pred_users.device)
        y_onehot.scatter_(1, pred_users, 1)
        y_onehot[:, :4] = 0.0
        loss_1 = self.l2(output, y_onehot)# + self.l2(exists_output, y_onehot_input)

        output = self.model(pred_users)
        # y_onehot = torch.FloatTensor(B, self.user_size)
        y_onehot = torch.randn((B, self.user_size))*0.05
        y_onehot = y_onehot.to(existing_users.device)
        y_onehot.scatter_(1, existing_users, 1)
        y_onehot[:, :4] = 0.0

        y_onehot_input = torch.FloatTensor(B, self.user_size)
        y_onehot_input.zero_()
        y_onehot_input = y_onehot_input.to(existing_users.device)
        y_onehot_input.scatter_(1, existing_users, 1)
        y_onehot_input[:, :4] = 0.0

        loss_2 = self.l2(output, y_onehot)# + self.l2(exists_output, y_onehot_input)
        loss = loss_1+loss_2*0.5
        tensorboard_logs = {'Loss/train': loss.item(), 'norm_loss/train': loss.item()/B,'sample_ratio': sample_rate }
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
            f1 = -1
        else:
            f1 = 2*(recall*precision)/(recall+precision)

        if np.isnan([f1, recall, precision]).any() or np.isnan(f1):
            recall, precision, f1 = 0, 0, 0

        loss = self.l2(output, y_onehot)

        return {'val_loss': loss, 'f1': f1, 'recall': recall, 'precision': precision }

    def validation_end(self, outputs):
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                self.logger.experiment.add_histogram(name, param, self.trainer.current_epoch)

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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

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
    parser = argparse.ArgumentParser(description='Set Transformer Recommendation Model')
    parser.add_argument('--query', type=str, default='group')
    parser.add_argument('--task', type=str, default='academic', choices=['socnet', 'academic'])
    parser.add_argument('--model', type=str, default='set_transformer')
    parser.add_argument('--freq', type=int, default=4, help='user exists minimal frequency')
    parser.add_argument('--order-shuffle', type=str2bool, default=True)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--enc-layer', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--bce-weight', type=float, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sample-ratio', type=float, default=0.8)
    parser.add_argument('--raising-ratio', type=float, default=0.01)
    parser.add_argument('--dataset', type=str, default='acm', 
        choices=['dblp', 'acm', 'amazon', 'lj', 'friendster', 'orkut'])
    parser.add_argument('--max-group', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    model = Model(args)
    trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=0.2, 
        gpus=[args.gpu])
    trainer.fit(model)
