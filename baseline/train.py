import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import SocialDataset, AMinerDataset, TOKENS, seq_collate
from .models import Seq2Seq
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils import clip_grad_norm
from .utils import IgnoreLogger, str2bool, confusion


class Baseline(pl.LightningModule):

    def __init__(self, args):
        super(Baseline, self).__init__()
        # not the best model...
        self.hparams = args
        self.max_group = args.max_group
        self.criterion = nn.CrossEntropyLoss(ignore_index=TOKENS['PAD'])
        self.dataset_class = AMinerDataset
        if args.task == 'socnet':
            self.dataset_class = SocialDataset

        self.train_dataset = self.dataset_class(train=True, 
            order_shuffle=args.order_shuffle,
            sample_ratio=self.hparams.sample_ratio, max_size=args.max_group, dataset=args.dataset,
            query=self.hparams.query, min_freq=args.freq)

        stats = self.train_dataset.get_stats()
        if self.train_dataset.embedding is not None:
            args.hidden = self.train_dataset.embedding.shape[-1]

        self.model = Seq2Seq(
            user_size=stats['member'],
            hidden_size=args.hidden,
        )
        if self.train_dataset.embedding is not None:
            print('Use pretrained graph embedding')
            embedding_weight = torch.from_numpy(self.train_dataset.embedding)
            self.model.encoder.embedding.from_pretrained(embedding_weight)
            self.model.encoder.embedding.weight.requires_grad=False

        # self.skip_gram = Skipgram()

    def training_step(self, batch, batch_idx):
        # REQUIRED
        existing_users, pred_users, pred_users_cnt = batch
        existing_users = existing_users.transpose(0, 1)
        pred_users = pred_users.transpose(0, 1)
        
        loss, norm_loss, _ = self.model(existing_users, pred_users, self.criterion, 
            teacher_forcing_ratio=0.3, device='cuda')
        
        tensorboard_logs = {'Loss/train': loss.item(), 'norm_loss/train': norm_loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        existing_users, pred_users, pred_users_cnt = batch
        existing_users = existing_users.transpose(0, 1)
        pred_users = pred_users.transpose(0, 1)
        
        loss, norm_loss, decoder_outputs = self.model(existing_users, pred_users, self.criterion, 
            teacher_forcing_ratio=self.hparams.teach_ratio, device='cuda')

        argmax = torch.argmax(decoder_outputs, dim=-1)

        invalid_targets = pred_users.eq(TOKENS['PAD'])
        accuracy = argmax.eq(pred_users).masked_fill_(invalid_targets, 0)\
            .float().sum()/pred_users_cnt.sum()


        return {'val_loss': loss, 'accuracy': accuracy, 'norm_loss': norm_loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = np.mean( [x['val_loss'] for x in outputs] )
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        norm_loss = np.mean( [x['norm_loss'] for x in outputs] )

        tensorboard_logs = {
            'Loss/val': avg_loss, 
            'avg_acc/val': avg_acc, 
            'Loss/norm_val': norm_loss,
            'val_loss': avg_loss, 
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
    parser = argparse.ArgumentParser(description='MoSAN Group Recommendation Model')
    parser.add_argument('--task', type=str, default='academic', choices=['socnet', 'academic'])
    parser.add_argument('--query', type=str, default='group')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--min-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--freq', type=int, default=4, help='user exists minimal frequency')
    parser.add_argument('--order-shuffle', type=str2bool, default=True)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--sample-ratio', type=float, default=0.8, 
        help='number of users in group selected for input and the rest for prediction')
    parser.add_argument('--max-group', type=int, default=120)
    parser.add_argument('--city', type=str, default='nyc', choices=['nyc', 'sf', 'chicago'])
    parser.add_argument('--teach-ratio', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='acm', 
        choices=['dblp', 'acm', 'amazon', 'lj', 'friendster', 'orkut'])
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    model = Baseline(args)
    trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=1.0, 
        gpus=[args.gpu], gradient_clip_val=args.clip_grad, 
        # logger=IgnoreLogger,
        )
    trainer.fit(model)