from datetime import datetime
from dataset.aminer import Aminer
from torch_geometric.data import DataLoader
import random
import shutil
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
import json
import os
import os.path as osp
import numpy as np
import torch.nn as nn
from src.skipgram import generate_batch, SkipGramNeg, data_to_networkx_, sample_walks
from src.utils import dict2table, confusion, str2bool, TMP_WRITER_PATH, calculate_f_score
from src.hint import HINT, evaluate, PAD_ID, BOS_ID, EOS_ID
from src.aminer import PaddedDataLoader

# PAD_ID = 874608
# BOS_ID = PAD_ID+1
# EOS_ID = PAD_ID+2

class HINT_Trainer():
    def __init__(self, args):
        dataset = Aminer(cutoff=args.maxhop,
            min_size=args.min_size, max_size=args.max_size,
            baseline=args.baseline)
        # initialize index for train/val/test datasets
        # make sure each runs share the same results
        if osp.exists('dblp_hete_shuffle_idx.pkl'):
            with open('dblp_hete_shuffle_idx.pkl', 'rb') as f:
                shuffle_idx = pickle.load(f)
        else:
            shuffle_idx = [idx for idx in range(len(dataset))]
            split_pos = int(len(dataset)*0.7)
            train_idx = shuffle_idx[:split_pos]
            random.shuffle(train_idx)
            shuffle_idx[:split_pos] = train_idx
            print(shuffle_idx[split_pos: split_pos+10])
            with open('dblp_hete_shuffle_idx.pkl', 'wb') as f:
                pickle.dump(shuffle_idx, f)

        dataset = dataset[shuffle_idx]

        split_pos = int(len(dataset)*0.7)
        train_idx = shuffle_idx[:split_pos]
        valid_idx_ = shuffle_idx[split_pos:]
        # 7: 1: 2 ; train : valid : test
        valid_pos = int(len(valid_idx_)*0.3333)
        valid_idx = valid_idx_[:100]
        test_idx = valid_idx_[valid_pos:]

        self.train_dataset = dataset[train_idx]
        self.test_dataset = dataset[test_idx]
        self.valid_dataset = dataset[valid_idx]


        self.args = args
        # initialize logging writer
        if args.writer is True:
            self.log_path = osp.join(
                "logs", "aminer",
                'hint_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        else:
            shutil.rmtree(TMP_WRITER_PATH, ignore_errors=True)
            self.log_path = TMP_WRITER_PATH
    
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.save_path = osp.join(self.log_path, "models")
        os.makedirs(self.save_path, exist_ok=True)
        print('finish init')

    def save_checkpoint(self, checkpoint, save_path, save_name):
        save_path = osp.join(save_path, 'model_{}.pth'.format(save_name))
        torch.save(checkpoint, save_path)

    def train(self, epochs=200):
        # training loop
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        args = self.args
        train_size = len(self.train_dataset.processed_file_idx)
        val_size = len(self.valid_dataset.processed_file_idx)
        print(val_size)
        # assert (len(set(self.valid_dataset.processed_file_idx+self.train_dataset.processed_file_idx))) == (train_size+val_size)

        train_loader = PaddedDataLoader(self.train_dataset,
                                  pad_idx=PAD_ID,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=6)
        valid_loader = PaddedDataLoader(self.valid_dataset,
                                  pad_idx=PAD_ID,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=6)

        test_loader = PaddedDataLoader(self.test_dataset,
                                 pad_idx=PAD_ID,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=6)
        # with open('aminer/preprocess_dblp.pkl', 'rb') as f:
        #     dblp = pickle.load(f)
        # author2id = dblp['author2id']
        # paper2id = dblp['paper2id']
        # conf2id = dblp['conf2id']
        self.user_size = 874608
        model = HINT(author_size=874608,#len(author2id),
                        paper_size=3605603,#len(paper2id),
                        conf_size=12770,#len(conf2id),
                        user_dim=args.author_dim,
                        paper_dim=args.paper_dim,
                        conf_dim=args.conf_dim,
                        input_channels=args.input_dim,
                        layers=args.layers,
                        dropout=args.dropout)
        if args.pretrain:
            model = self.pretrain_embeddings(model, 256, epoch_num=10)
        model = model.cuda()

        if args.pos_weight <= 0:
            weight = 5  # default
            args.pos_weight = weight
        else:
            weight = args.pos_weight
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'max')
        # balance 1 and 0 distribution: 1=add user, 0=don't add this user
        pos_weight = torch.ones([PAD_ID+3])*weight
        self.pos_weight = pos_weight.cuda()
        pos_weight = torch.ones([1])*4
        pos_weight = pos_weight.cuda()

        binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        n_iter = 0
        best_f1 = 0
        iter_size = 8

        self.writer.add_text('Text', dict2table(vars(args)), 0)
        model.train()
        with tqdm(total=len(train_loader)*epochs, dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                loss_mini_batch = 0

                for i, data in enumerate(train_loader):
                    x, edge_index = data.x, data.edge_index
                    x = x.cuda()
                    rank_loss, label_pred, (seq_pred, seq_label) = model.train_batch(data)

                    # print(target.shape)
                    # pred = output
                    node_loss = criterion(label_pred, x[:, -1])
                    seq_loss = binary_criterion(seq_pred.flatten(), seq_label.flatten())

                    loss = (rank_loss + node_loss*5 + seq_loss * 5)

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    self.writer.add_scalar(
                        "Train/RankLoss", rank_loss.item(), n_iter)
                    self.writer.add_scalar(
                        "Train/NodeLoss", node_loss.item(), n_iter)
                    self.writer.add_scalar(
                        "Train/LengthLoss", seq_loss.item(), n_iter)
                    self.writer.add_scalar(
                        "Train/Loss", loss.item(), n_iter)

                    pbar.update(1)
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(loss.item(), epoch))
                    n_iter += 1

                if epoch % args.eval == 0:
                    print('Epoch: ', epoch)
                    f1, recalls, precisions = evaluate(valid_loader,
                                                            model, user_size=PAD_ID+1, 
                                                            batch_size=args.batch_size)

                    scheduler.step(f1)
                    self.writer.add_scalar("Valid/F1", f1, n_iter)
                    self.writer.add_scalar("Valid/Recalls", recalls, n_iter)
                    self.writer.add_scalar("Valid/Precisions", precisions,
                                           n_iter)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_checkpoint = {
                            'epoch': epoch+1,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'f1': f1
                        }
                    print(f1)

                if epoch % args.save == 0:
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f1': f1
                        },
                        self.save_path,
                        "{}".format(epoch+1)
                    )
        f1, recalls, precisions = evaluate(valid_loader, model, user_size=PAD_ID+1,
                                                            batch_size=args.batch_size)
        self.writer.add_scalar("Valid/F1", f1, n_iter)
        self.writer.add_scalar("Valid/Recalls", recalls, n_iter)
        self.writer.add_scalar("Valid/Precisions", precisions, n_iter)
        if f1 > best_f1:
            best_f1 = f1
            best_checkpoint = {
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'f1': f1
            }
        print(f1)
        print("Testing")
        self.save_checkpoint(best_checkpoint,
                             self.save_path,
                             "best")
        model.load_state_dict(best_checkpoint["model"])
        f1, recalls, precisions = evaluate(test_loader, model, user_size=PAD_ID+1,
                                                    batch_size=args.batch_size)
        self.writer.add_scalar("Test/F1", f1, n_iter)
        self.writer.add_scalar("Test/Recalls", recalls, n_iter)
        self.writer.add_scalar("Test/Precisions", precisions, n_iter)
        self.writer.flush()
        return f1, recalls, precisions

    def pretrain_embeddings(self, model, batch_size, epoch_num=1, neg_num=20):
        import torch.optim as optim
        print('Pretrain embeddings')
        node_types = {
            0: model.author_size,
            1: model.paper_size,
            2: model.conf_size,
        }
        embeddings = {}
        model = model.cpu()

        for node_type, (embed_size, dim) in node_types.items():
            if osp.exists(osp.join(args.pretrain_weight,
                                   'random_walk_{}.pt'.format(node_type))):
                samples = torch.load(
                    osp.join(args.pretrain_weight,
                             'random_walk_{}.pt'.format(node_type)))['samples']
            else:
                samples = sample_walks(self.train_dataset, neg_num, batch_size, node_type, embed_size)
                torch.save({'samples' : samples}, os.path.join(self.save_path, 'random_walk_{}.pt'.format(node_type)))

            skip_model = SkipGramNeg(embed_size, dim)
            skip_model = skip_model.cuda()
            optimizer = optim.SGD(skip_model.parameters(), lr=1e-5, weight_decay=1e-9)
            iteration = list(range(len(self.train_dataset)))
            total_idx = 0
            print('sampling')
            with tqdm(total=len(iteration)*epoch_num) as pbar:
                for e in range(epoch_num):
                    random.shuffle(samples)
                    for idx, sample in enumerate(samples):
                        context, target, negative = sample
                        context = torch.from_numpy(context).to(dtype=torch.long).cuda()
                        target = torch.from_numpy(target).to(dtype=torch.long).cuda()
                        negative = torch.from_numpy(negative).to(dtype=torch.long).cuda()
                        loss = skip_model(target, context, negative)

                        loss.backward()

                        pbar.set_description(
                            "loss {:.4f}, iter {}".format(loss.item(), total_idx))
                        if self.writer != None:
                            self.writer.add_scalar('Skipgram/loss/%d' % node_type, loss.item(), total_idx)
                        total_idx += 1
                        pbar.update(1)
            if total_idx == 0:
                continue

            del samples
            embeddings[node_type] = skip_model.input_emb.weight
            if node_type == 0:
                print('transfer user embeddings')
                model.gcn.embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 1:
                print('transfer topic embeddings')
                model.gcn.paper_embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 2:
                print('transfer category embeddings')
                model.gcn.conf_embeddings.weight.data = skip_model.input_emb.weight.data
        torch.save(embeddings, os.path.join(self.save_path, 'embeddings.pt'))
        return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Deepset Recommendation Model on Amazon with categories')
    # dataset parameters
    parser.add_argument('--min-size', type=int, default=5)
    parser.add_argument('--max-size', type=int, default=100)
    parser.add_argument('--pred-ratio', type=float, default=0.8)
    parser.add_argument('--maxhop', type=int, default=2)
    # training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--baseline', type=str2bool, nargs='?', default=False,
        help='baseline model only takes in previous co-author relationship (no conference, no paper id)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--pos-weight', type=float, default=-1)
    parser.add_argument('--eval', type=int, default=5)
    parser.add_argument('--save', type=int, default=50)
    parser.add_argument('--pretrain', type=str2bool, nargs='?', default=False)
    parser.add_argument('--pretrain-weight', type=str, default='')
    # model parameters
    parser.add_argument('--author-dim', type=int, default=16)
    parser.add_argument('--paper-dim', type=int, default=16)
    parser.add_argument('--conf-dim', type=int, default=8)
    parser.add_argument('--input-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=[32, 32])
    # debug
    parser.add_argument('--repeat-n', type=int, default=1)
    parser.add_argument('--writer', type=str2bool, nargs='?', default=True)
    args = parser.parse_args()

    # trainer = GroupGCN(args)
    # trainer.train(epochs=args.epochs)
    values = {
        'f1': [],
        'recall': [],
        'precision': [],
    }

    for i in range(args.repeat_n):
        trainer = HINT_Trainer(args)
        f1, recalls, precisions = trainer.train(epochs=args.epochs)
        values['f1'].append(f1)
        values['recall'].append(recalls)
        values['precision'].append(precisions)

    results = {}
    for key, value in values.items():
        results['avg_'+key] = np.mean(value)
        results['std_'+key] = np.std(value)
    results['results'] = values
    results['arguments'] = vars(args)
    with open('aminer_node_class'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)
