import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import Meetupv1, Meetupv2, TOKENS
from .models import Seq2Seq, Seq2SeqwTag
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils import clip_grad_norm
from .utils import IgnoreLogger, binary_matrix

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Baseline(pl.LightningModule):

    def __init__(self, args):
        super(Baseline, self).__init__()
        # not the best model...
        self.hparams = args
        self.max_group = args.max_group
        self.criterion = nn.NLLLoss(ignore_index=TOKENS['PAD'])
        self.train_dataset = Meetupv2(train=True, 
            sample_ratio=self.hparams.sample_ratio, max_size=self.max_group, city=args.city,
             query=self.hparams.query, min_freq=args.freq)
        stats = self.train_dataset.get_stats()
        if args.model == 'seq2seq':
            Model = Seq2Seq
        else:
            Model = Seq2SeqwTag
        print(Model)
        self.model = Model(
            embed_size=args.user_dim,
            vocab_size=stats['member']+3,
            hidden_size=args.hidden,
            enc_num_layers=args.enc_layer,
            dec_num_layers=args.dec_layer,dropout=0.1,
            st_mode=False,
            use_attn=args.attn,
            tag_size=stats['topic']+3,
        )
        # self.skip_gram = Skipgram()

    def training_step(self, batch, batch_idx):
        existing_users, pred_users, pred_users_cnt, tags = batch

        if self.hparams.model == 'seq2seq':
            decoder_outputs, d_h, hidden = self.model(existing_users, pred_users)
        else:
            decoder_outputs, d_h, hidden = self.model(existing_users, pred_users, tags)

        seq_length = decoder_outputs.shape[1]
        loss = 0
        for t in range(seq_length):
            loss += self.criterion(torch.log(decoder_outputs[:, t, :]), pred_users[:,t+1])
        norm_loss = loss/existing_users.shape[0]
        tensorboard_logs = {'loss/train': loss.item(), 'norm_loss/train': norm_loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        existing_users, pred_users, pred_users_cnt, tags = batch
        if self.hparams.model == 'seq2seq':
            decoder_outputs, d_h, hidden = self.model(existing_users, pred_users)
        else:
            decoder_outputs, d_h, hidden = self.model(existing_users, pred_users, tags)

        seq_length = decoder_outputs.shape[1]
        loss = 0
        for t in range(seq_length):
            loss += self.criterion(torch.log(decoder_outputs[:, t, :]), pred_users[:,t+1], )
        norm_loss = loss/existing_users.shape[0]
        argmax = torch.argmax(decoder_outputs, dim=-1)
        invalid_targets = pred_users[:, 1:].eq(TOKENS['PAD'])
        accuracy = argmax.eq(pred_users[:, 1:]).masked_fill_(invalid_targets, 0)\
            .float().sum()/pred_users_cnt.sum()

        return {'val_loss': loss, 'accuracy': accuracy, 'norm_loss': norm_loss.item()}

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
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.train_dataset, 
            # sampler=self.dist_sampler, 
            batch_size=self.hparams.batch_size, num_workers=4, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        self.dataset = Meetupv2(train=False,
            sample_ratio=self.hparams.sample_ratio, min_freq=self.hparams.freq,
            city=self.hparams.city, max_size=self.max_group, query=self.hparams.query)
        # self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            batch_size=8, num_workers=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MoSAN Group Recommendation Model')
    parser.add_argument('--dataset', type=str, default='meetup')
    parser.add_argument('--query', type=str, default='group')
    parser.add_argument('--user-dim', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--enc-layer', type=int, default=1)
    parser.add_argument('--dec-layer', type=int, default=1)
    parser.add_argument('--freq', type=int, default=10)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--min-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--attn', type=str2bool, default=False, help='use attention')
    parser.add_argument('--sample-ratio', type=float, default=0.8, 
        help='number of users in group selected for input and the rest for prediction')
    parser.add_argument('--max-group', type=int, default=500)
    parser.add_argument('--city', type=str, default='nyc', choices=['nyc', 'sf', 'chicago'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='seq2seq', choices=['seq2seq', 'seq2seqwtag'])

    args = parser.parse_args()
    model = Baseline(args)
    trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=1.0, 
        gpus=[args.gpu], gradient_clip_val=args.clip_grad, 
        # logger=IgnoreLogger,
        )
    trainer.fit(model)