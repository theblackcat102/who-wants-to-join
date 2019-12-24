import numpy as np
import torch
import os, glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import AMinerDataset, TOKENS, seq_collate, SocialDataset
from .set_transformer import SetTransformer
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
    dataset_class = AMinerDataset
    if train_params['task'] == 'socnet':
        dataset_class = SocialDataset
    dataset = dataset_class(train=False, sample_ratio=float(train_params['sample_ratio']),
         order_shuffle= str2bool(train_params['order_shuffle'])  if 'order_shuffle' in train_params else True,
         max_size=int(train_params['max_group']), query='group', dataset=str(train_params['dataset']), 
         min_freq = int(train_params['freq']) if 'freq' in train_params else 5)
    stats = dataset.get_stats()
    print(train_params['dataset'])
    print(train_params)
    model = SetTransformer(
        user_size=int(stats['member'])+3, 
        hidden=int(train_params['hidden']), 
        heads=int(train_params['heads']),
        layers=int(train_params['enc_layer'])
    )
    model.load_state_dict({ key[6:] : value for key, value in checkpoint['state_dict'].items() if 'l2' not in key})
 
    top_k = args.topk
    model.cuda()

    dataloader = DataLoader(dataset, 
            batch_size=64, num_workers=4, shuffle=False, collate_fn=seq_collate)
    model.eval()
    device = 'cuda'
    
    stats = []

    with torch.no_grad():
        pbar = tqdm(dataloader, dynamic_ncols=True)
        for batch in pbar:
            # print(len(batch))
            existing_users, pred_users, pred_users_cnt = batch
            existing_users = existing_users.cuda()
            pred_users = pred_users.cuda()
            output = model(existing_users)

            B = existing_users.shape[0]
            user_size = output.size(1)
            y_onehot = torch.FloatTensor(B, user_size)
            y_onehot.zero_()
            y_onehot = y_onehot.to(pred_users.device)
            y_onehot.scatter_(1, pred_users, 1)
            # print(existing_users[:, :10])
            # print(existing_users[:,:10], y_onehot[:, pred_users.flatten()],
            #     pred_users[:, :10], y_onehot[:, :10]
            # )

            # y_onehot_pred = (torch.sigmoid(output)>0.5).long()

            # print(torch.mean(pred_users_cnt.float()))
            _, decoder_outputs = torch.topk(output, k=args.topk, dim=-1)

            y_onehot_pred = torch.FloatTensor(B, user_size)
            y_onehot_pred.zero_()
            y_onehot_pred = y_onehot_pred.to(decoder_outputs.device)
            y_onehot_pred.scatter_(1, decoder_outputs, 1)

            TP, FP, TN, FN = confusion(y_onehot_pred, y_onehot)
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