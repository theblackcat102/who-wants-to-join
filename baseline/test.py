import numpy as np
import torch
import os, glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import Meetupv1, Meetupv2, TOKENS
from .models import Seq2Seq
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils import clip_grad_norm

def extract_checkpoint_files(path):
    highest_epoch = 0
    for filename in glob.glob(os.path.join(path, '*.ckpt')):
        filename = os.path.basename(filename)
        epoch_ = filename.split('.')[0].split('_')[-1]
        highest_epoch = max(highest_epoch, int(epoch_))
    return highest_epoch

def load_params(filename):
    params = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            key, value = line.split(',')
            params[key] = value
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--version', type=str, default='version_5')
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--topk', type=int, default=1)
    args = parser.parse_args()
    checkpoint_path = os.path.join('lightning_logs/'+args.version, 'checkpoints')
    best_epoch = args.epoch
    if args.epoch < 0:
        best_epoch = extract_checkpoint_files(checkpoint_path)
    restore_path = 'lightning_logs/{}/checkpoints/_ckpt_epoch_{}.ckpt'.format(args.version, best_epoch)
    checkpoint = torch.load(restore_path)
    params_f = 'lightning_logs/{}/meta_tags.csv'.format(args.version)
    train_params = load_params(params_f)
    dataset = Meetupv2(train=False, sample_ratio=float(train_params['sample_ratio']),
         max_size=int(train_params['max_group']), query='group', city=str(train_params['city']))
    print(train_params)
    stats = dataset.get_stats()
    model = Seq2Seq(
        user_size=stats['member']+3,
        hidden_size=int(train_params['user_dim']),
        )
    model.load_state_dict({ key[6:] : value for key, value in checkpoint['state_dict'].items()})
 
    top_k = args.topk
    model.cuda()

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
        for batch in tqdm(dataloader, dynamic_ncols=True):
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

            decoder_outputs = decoder_outputs_idx.cpu().numpy()
            pred_users = pred_users.cpu().numpy()

            decoder_outputs = np.unique(decoder_outputs.flatten())
            pred_users = pred_users.flatten()
            acc = 0

            for token in decoder_outputs:
                if token not in [0, 1, 2]:
                    if token in pred_users:
                        acc += 1
            pred_cnt.append(pred_users_cnt.sum().item())
            match_score.append(acc)
            stats.append(acc / pred_users_cnt.sum().item())

    # print(stats)
    print(np.mean(stats))
