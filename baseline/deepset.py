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

dataset_stats = {
    'meetup_v1': {
       'user_size': 46899,
       'group_size': 40,
    },
    'meetup_v2': {
       'user_size': 1087928,
       'group_size': 20000,
    }
}



class Deepset(nn.Module):

    def __init__(self, user_size, hidden_size, set_features=20):
        super(Deepset, self).__init__()
        self.embedding = FactorizedEmbeddings(user_size, hidden_size, hidden_size//2)
        # self.tag_embedding = nn.Embedding(tag_size, 30)

        self.extractor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, user_size),
            nn.Sigmoid(),
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
        self.user_size=dataset_stats[self.hparams.dataset]['user_size']
        hidden_size=32
        set_features=512
        tag_size=966
        self.model = Deepset(self.user_size, hidden_size, set_features)
        pos_weight = torch.ones([self.user_size])
        pos_weight[:4] = 0.0
        # self.l2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        group_size = dataset_stats[self.hparams.dataset]['group_size']
        self.dataset = Meetupv1(train=True, 
            sample_ratio=self.hparams.sample_ratio, max_size=group_size, query=self.hparams.query)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            batch_size=self.hparams.batch_size, num_workers=4, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        group_size = dataset_stats[self.hparams.dataset]['group_size']
        self.dataset = Meetupv1(train=True, 
            sample_ratio=self.hparams.sample_ratio, max_size=group_size, query=self.hparams.query)
        return DataLoader(self.dataset, 
            # sampler=self.dist_sampler, 
            batch_size=8, num_workers=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepset Recommendation Model')
    parser.add_argument('--dataset', type=str, default='meetup_v1')
    parser.add_argument('--query', type=str, default='group')
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--model', type=str, default='deepset')
    parser.add_argument('--user-dim', type=int, default=50)
    parser.add_argument('--feature-dim', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=40)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sample-ratio', type=float, default=0.8)
    parser.add_argument('--max-group', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    if args.task == 'train':
        model = Model(args)

        trainer = pl.Trainer(max_nb_epochs=args.max_epochs,min_nb_epochs=args.min_epochs, train_percent_check=1.0, 
            gpus=[args.gpu])
        trainer.fit(model)
    else:
        restore_path = 'lightning_logs/version_1/checkpoints/_ckpt_epoch_1.ckpt'
        checkpoint = torch.load(restore_path)
        model = Deepset(user_size=dataset_stats[args.dataset]['user_size'], hidden_size=32, set_features=512)
        model = model.cuda()
        model.eval()

        model.load_state_dict({ key[6:] : value for key, value in checkpoint['state_dict'].items()})
        dataset = Meetupv1(train=False, 
            sample_ratio=args.sample_ratio, max_size=dataset_stats[args.dataset]['group_size'], 
            query=args.query)

        val_iter = DataLoader(dataset, batch_size=8, num_workers=8)
        matched = 0
        group_size = 0
        y_onehot = torch.FloatTensor(8, dataset_stats[args.dataset]['user_size'])
        for batch in val_iter:
            existing_users, target_users, target_users_cnt, _ = batch        
            existing_users = existing_users.cuda()
            target_users = target_users.cuda()

            y_onehot.zero_()
            y_onehot = y_onehot.to(target_users.device)
            y_onehot.scatter_(1, target_users, 1)
            # print(existing_users, target_users)

            pred_users = torch.sigmoid(model(existing_users))
            # print(y_onehot[0].sum(), pred_users[0][:100])

            '''
                [ 0.1 0.9 ]
                [ 1 ]
            '''
            target_users = target_users.cpu()
            pred_users = pred_users.cpu()

            user_size = pred_users.shape[-1]

            for idx in range(len(pred_users)):
                # print(pred_users[idx])
                match_idx = pred_users[idx] > 0.5
                user_idx = torch.arange(0, user_size )[ match_idx ].to(target_users.device)
                if len(user_idx) == 0:
                    continue
                intersection = np.intersect1d(user_idx.numpy(), 
                target_users[idx].numpy())

                matched += (intersection != 1).sum()
                group_size += target_users_cnt[idx]
                print(intersection, len(user_idx))

        print(matched/group_size)