import numpy as np
import torch
import os, glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import AMinerDataset, TOKENS, seq_collate, SocialDataset
from .models import Seq2Seq
from .utils import str2bool, confusion
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

    dataset = SocialDataset(train=False, sample_ratio=float(train_params['sample_ratio']),
         order_shuffle= str2bool(train_params['order_shuffle'])  if 'order_shuffle' in train_params else True,
         max_size=int(train_params['max_group']), query='group', dataset=str(train_params['dataset']), 
         min_freq = int(train_params['freq']) if 'freq' in train_params else 5)


    print(train_params)
    stats = dataset.get_stats()
    model = Seq2Seq(
        user_size=stats['member'],
        hidden_size=int(train_params['hidden']),
        )
    user_size= stats['member']

    model.load_state_dict({ key[6:] : value for key, value in checkpoint['state_dict'].items()})
 
    top_k = args.topk
    model.cuda()

    dataloader = DataLoader(dataset, 
            # sampler=self.dist_sampler, 
            collate_fn=seq_collate,
            batch_size=1, num_workers=1, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKENS['PAD'])
    device = 'cuda'
    stats = []
    match_score = []
    pred_cnt = []
    B = 1

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):

            existing_users, pred_users, pred_users_cnt = batch
            pred_users = pred_users.long()
            existing_users = existing_users.long()

            existing_users = existing_users.transpose(0, 1)
            # pred_users = pred_users.transpose(0, 1)

            existing_users = existing_users.cuda()
            pred_users = pred_users.cuda()
            # pred_users_cnt = pred_users_cnt.cuda()

            total_users = pred_users_cnt.sum().item()
            if total_users == 0:
                continue
            # print(torch.max(existing_users), torch.max(pred_users), torch.max(tags))
            loss, norm_loss, decoder_outputs = model(existing_users, None, None, device=device, 
                teacher_forcing_ratio=1.0, max_length=len(pred_users[0]))

            _, decoder_outputs_idx = torch.topk(decoder_outputs, k=top_k, dim=-1)
            # print('result: ',decoder_outputs_idx.flatten(), pred_users.flatten())

            y_onehot = torch.FloatTensor(B, user_size)
            y_onehot.zero_()
            y_pred = decoder_outputs_idx.flatten().unsqueeze(0)
            y_onehot.scatter_(1, y_pred.cpu(), 1)
            y_onehot[:, :4] = 0.0

            y_target = torch.FloatTensor(B, user_size)
            y_target.zero_()
            y_target.scatter_(1, pred_users.cpu(), 1)
            y_target[:, :4] = 0.0

            TP, FP, TN, FN = confusion(y_onehot.long().float(), y_target.long().float())
            stats.append([TP, FP, TN, FN])

    stats = np.array(stats)
    recall = np.sum(stats[:, 0])/ (np.sum(stats[:, 0])+ np.sum(stats[:, 3]))
    precision = np.sum(stats[:, 0])/ (np.sum(stats[:, 0])+ np.sum(stats[:, 1]))

    if recall != 0:
        f1 = 2*(recall*precision)/(recall+precision)
        accuracy = (np.sum(stats[:, 0]) + np.sum(stats[:, 2]))/ (np.sum(stats))
        print('Accuracy: ',accuracy)
        print('Recall: ',recall)
        print('Precision: ',precision)
        print('F1: ',f1)
    else:
        print('Recall: ',recall)

    # print(stats)
    print(np.mean(stats))
