import numpy as np
import torch
import os, glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import Meetupv1, SocialDataset, TOKENS, seq_collate
from .models import Seq2Seq
from .train import str2bool
from .utils import orthogonal_initialization, predict, confusion
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
    checkpoint_path = os.path.join('lightning_logs/seq2seq/'+args.version, 'checkpoints')
    best_epoch = args.epoch
    if args.epoch < 0:
        best_epoch = extract_checkpoint_files(checkpoint_path)
    restore_path = 'lightning_logs/seq2seq/{}/checkpoints/_ckpt_epoch_{}.ckpt'.format(args.version, best_epoch)
    checkpoint = torch.load(restore_path)
    params_f = 'lightning_logs/seq2seq/{}/meta_tags.csv'.format(args.version)
    train_params = load_params(params_f)
    dataset = SocialDataset(train=False, sample_ratio=float(train_params['sample_ratio']),
         order_shuffle= str2bool(train_params['order_shuffle'])  if 'order_shuffle' in train_params else True,
         max_size=int(train_params['max_group']), query='group', dataset=str(train_params['dataset']), 
         min_freq = int(train_params['freq']) if 'freq' in train_params else 5)
    stats = dataset.get_stats()

    Model = Seq2Seq

    model = Model(
        embed_size=int(train_params['user_dim']),
        vocab_size=int(stats['member'])+3,
        enc_num_layers=int(train_params['enc_layer']),
        dec_num_layers=int(train_params['dec_layer']),
        dropout=0.1,
        st_mode=False,
        use_attn=str2bool(train_params['attn']),
        hidden_size=int(train_params['hidden']),
    )
    model.load_state_dict({ key[6:] : value for key, value in checkpoint['state_dict'].items()})
 
    top_k = args.topk
    model.cuda()

    dataloader = DataLoader(dataset, 
            batch_size=1, num_workers=1, shuffle=False, collate_fn=seq_collate)
    model.eval()
    device = 'cuda'
    stats_, match_score, pred_cnt,f1_stats = [], [], [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, dynamic_ncols=True)
        for batch in pbar:
            # print(len(batch))
            existing_users, pred_users, pred_users_cnt = batch
            existing_users = existing_users.cuda()
            pred_users = pred_users.cuda()
            pred_users_cnt = pred_users_cnt.cuda()

            total_users = pred_users_cnt.sum().item()
            if total_users == 0:
                continue


            decoder_outputs, _ = model.decode(existing_users, 
                target_length=pred_users.size(1), topk=top_k)
            # pred_users = pred_users.cpu().numpy()
            if len(decoder_outputs.flatten()) == 0:
                continue

            acc = predict(decoder_outputs, pred_users.cpu())
            pbar.set_description("Match %d" % np.sum(match_score))
            pred_cnt.append(pred_users_cnt.sum().item())
            match_score.append(acc)
            stats_.append(acc / pred_users_cnt.sum().item())


            decoder_outputs = torch.from_numpy(decoder_outputs).to(existing_users.device)

            B = existing_users.shape[0]
            user_size = int(stats['member'])+3

            y_onehot = torch.FloatTensor(B, user_size)
            y_onehot.zero_()
            y_onehot = y_onehot.to(existing_users.device)
            y_onehot.scatter_(1, pred_users, 1)
            y_onehot[:, :4] = 0

            onehot_pred = torch.FloatTensor(B, user_size)
            onehot_pred.zero_()
            onehot_pred = onehot_pred.to(existing_users.device)
            onehot_pred.scatter_(1, decoder_outputs, 1)
            onehot_pred[:, :4] = 0

            TP, FP, TN, FN = confusion(onehot_pred, y_onehot)
            f1_stats.append([TP, FP, TN, FN])

    f1_stats = np.array(f1_stats)
    recall = np.sum(f1_stats[:, 0])/ (np.sum(f1_stats[:, 0])+ np.sum(f1_stats[:, 3]))
    precision = np.sum(f1_stats[:, 0])/ (np.sum(f1_stats[:, 0])+ np.sum(f1_stats[:, 1]))

    if recall != 0:
        f1 = 2*(recall*precision)/(recall+precision)
        print('Recall: ',recall)
        print('Precision: ',precision)
        print('F1: ',f1)
    else:
        print('Recall: ',recall)

    print(np.sum(match_score)/np.sum(pred_cnt))
    print(np.mean(stats_))
