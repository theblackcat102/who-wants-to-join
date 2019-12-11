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



if __name__ == "__main__":
    restore_path = 'lightning_logs/version_5/checkpoints/_ckpt_epoch_17.ckpt'
    checkpoint = torch.load(restore_path)
    model = Seq2Seq(
        user_size=1087928,
        hidden_size=30,
        )
    model.load_state_dict({ key[6:] : value for key, value in checkpoint['state_dict'].items()})
 
    top_k = 10
    model.cuda()
    dataset = Meetupv2(train=False, sample_ratio=0.8, max_size=700, query='group')
    dataloader = DataLoader(dataset, 
            # sampler=self.dist_sampler, 
            batch_size=1, num_workers=1, shuffle=False)
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=TOKENS['PAD'])
    device = 'cuda'
    stats = []
    match_score = []
    pred_cnt = []
    with torch.no_grad():
        for batch in dataloader:
            existing_users, pred_users, pred_users_cnt, tags = batch
            existing_users = existing_users.transpose(0, 1)
            pred_users = pred_users.transpose(0, 1)
            tags = tags.transpose(0, 1).long()

            existing_users = existing_users.cuda()
            pred_users = pred_users.cuda()
            tags = tags.cuda()
            pred_users_cnt = pred_users_cnt.cuda()

            total_users = pred_users_cnt.sum().item()
            if total_users == 0:
                continue
            # print(torch.max(existing_users), torch.max(pred_users), torch.max(tags))
            loss, norm_loss, decoder_outputs = model(existing_users, pred_users, tags, criterion, device=device, 
                teacher_forcing_ratio=1.0)
            _, decoder_outputs_idx = torch.topk(decoder_outputs, k=top_k, dim=-1)
            # print(decoder_outputs_idx.shape)
            # decoder_outputs = torch.argmax(decoder_outputs, dim=2)
            # decoder_outputs = decoder_outputs.transpose(0, 1)
            # pred_users = pred_users.transpose(0, 1)

            decoder_outputs = decoder_outputs_idx.cpu().numpy()
            pred_users = pred_users.cpu().numpy()

            decoder_outputs = np.unique(decoder_outputs.flatten())
            pred_users = pred_users.flatten()
            acc = 0
            for token in decoder_outputs:
                if token not in [0, 1, 2]:
                    if token in pred_users:
                        acc += 1
            print(total_users)
            pred_cnt.append(pred_users_cnt.sum().item())
            match_score.append(acc)
            stats.append(acc / pred_users_cnt.sum().item())

            print(decoder_outputs[:20])
            print(pred_users[:total_users+2])
            print(np.sum(match_score)/np.sum(pred_cnt), stats[-1])

    print(stats)
    print(np.mean(stats))
