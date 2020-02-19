from datetime import datetime
from src.layers import StackedGCNDBLP
from dataset.aminer import Aminer
from torch_geometric.data import DataLoader
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
import json
import os
import os.path as osp
import numpy as np
from torch_geometric.nn.models.node2vec import Node2Vec
from src.skipgram import generate_batch, SkipGramNeg, data_to_networkx_, sample_walks
from src.utils import dict2table, confusion, str2bool

class GroupGCN():
    def __init__(self, args):
        dataset = Aminer(train=True,cutoff=args.maxhop,
            min_size=args.min_size, max_size=args.max_size,
            baseline=args.baseline)

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
        split_index = int(len(shuffle_idx)*0.9)
        train_idx = shuffle_idx[:split_index]
        valid_idx = shuffle_idx[split_index:]
        self.train_dataset = dataset[train_idx]
        self.valid_dataset = dataset[valid_idx]
        self.test_dataset = Aminer(train=False,cutoff=args.maxhop,
            min_size=args.min_size, max_size=args.max_size,
            baseline=args.baseline)

        self.args = args

        self.log_path = osp.join(
            "logs", "aminer",
            'multi_dblp_hete_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.save_path = osp.join(self.log_path, "models")
        os.makedirs(self.save_path, exist_ok=True)
        print('finish init')

    def save_checkpoint(self, checkpoint, save_path, save_name):
        save_path = osp.join(save_path, 'model_{}.pth'.format(save_name))
        torch.save(checkpoint, save_path)

    def evaluate(self, dataloader, model):
        model.eval()
        recalls = []
        precisions = []
        B = args.batch_size
        user_size = 874608
        print('Validation')
        with torch.no_grad():
            for val_data in tqdm(dataloader, dynamic_ncols=True):
                x, edge_index = val_data.x, val_data.edge_index
                y = val_data.y
                pred_mask = val_data.label_mask
                pred, _ = model(edge_index.cuda(), x.cuda())
                pred = torch.sigmoid(pred).cpu()

                mask_idx = (pred_mask == 1)
                B = val_data.batch.max()+1
                y_pred = torch.FloatTensor(B, user_size)
                y_target = torch.FloatTensor(B, user_size)
                y_pred.zero_()
                y_target.zero_()
                pred = pred.squeeze(1)
                for batch_idx in range(B):
                    batch_idxes = (val_data.batch == batch_idx)

                    target_idx  = (y == 1)
                    x_idx = x[ batch_idxes & mask_idx & target_idx, 0 ]
                    y_target[batch_idx, x_idx ] = 1

                    target_idx  = (pred > 0.5)
                    x_idx = x[ batch_idxes & mask_idx & target_idx, 0 ]
                    y_pred[batch_idx, x_idx ] = 1


                TP, FP, TN, FN = confusion(y_pred, y_target)

                recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
                precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
                precisions.append(precision)
                recalls.append(recall)

        avg_recalls = np.mean(recalls)
        avg_precisions = np.mean(precisions)
        f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)
        model.train()
        return f1, avg_recalls, avg_precisions

    def train(self, epochs=200):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        args = self.args
        train_size = len(self.train_dataset.processed_file_idx)
        val_size = len(self.valid_dataset.processed_file_idx)
        # assert (len(set(self.valid_dataset.processed_file_idx+self.train_dataset.processed_file_idx))) == (train_size+val_size)

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=6)
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=6)

        test_loader = DataLoader(self.test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=6)
        # with open('aminer/preprocess_dblp.pkl', 'rb') as f:
        #     dblp = pickle.load(f)
        # author2id = dblp['author2id']
        # paper2id = dblp['paper2id']
        # conf2id = dblp['conf2id']
        model = StackedGCNDBLP(
                           author_size=874608,#len(author2id),
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
        print('weight : ', weight)
        pos_weight = torch.ones([1])*weight
        pos_weight = pos_weight.cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        xentropy = torch.nn.CrossEntropyLoss()

        n_iter = 0
        best_f1 = 0

        self.writer.add_text('Text', dict2table(vars(args)), 0)

        with tqdm(total=len(train_loader)*epochs, dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                for data in train_loader:
                    optimizer.zero_grad()
                    x, edge_index = data.x, data.edge_index
                    x = x.cuda()
                    edge_index = edge_index.cuda()
                    pred_mask = data.label_mask.cuda() == 1
                    label = data.y.unsqueeze(-1).cuda().float()

                    node_pred, type_pred = model(edge_index, x)
                    type_loss = xentropy(type_pred, x[:, -1])
                    pred_loss = criterion(node_pred[pred_mask], label[pred_mask])
                    loss = pred_loss + type_loss
                    loss.backward()

                    optimizer.step()
                    self.writer.add_scalar(
                        "Train/BCEWithLogitsLoss", pred_loss.item(), n_iter)
                    self.writer.add_scalar(
                        "Train/CrossEntropyLoss", type_loss.item(), n_iter)
                    pbar.update(1)
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(loss.item(), epoch))
                    n_iter += 1

                if epoch % args.eval == 0:
                    print('Epoch: ', epoch)
                    f1, recalls, precisions = self.evaluate(valid_loader,
                                                            model)
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
        f1, recalls, precisions = self.evaluate(valid_loader, model)
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
        print("Testing")
        self.save_checkpoint(best_checkpoint,
                             self.save_path,
                             "best")
        model.load_state_dict(best_checkpoint["model"])
        f1, recalls, precisions = self.evaluate(test_loader, model)
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
                        optimizer.step()
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
                model.embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 1:
                print('transfer topic embeddings')
                model.paper_embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 2:
                print('transfer category embeddings')
                model.conf_embeddings.weight.data = skip_model.input_emb.weight.data
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pos-weight', type=float, default=-1)
    parser.add_argument('--eval', type=int, default=10)
    parser.add_argument('--save', type=int, default=50)
    parser.add_argument('--pretrain', type=str2bool, nargs='?', default=False)
    parser.add_argument('--pretrain-weight', type=str, default='')
    # model parameters
    parser.add_argument('--author-dim', type=int, default=16)
    parser.add_argument('--paper-dim', type=int, default=16)
    parser.add_argument('--conf-dim', type=int, default=8)
    parser.add_argument('--input-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--repeat-n', type=int, default=1)
    args = parser.parse_args()

    # trainer = GroupGCN(args)
    # trainer.train(epochs=args.epochs)
    values = {
        'f1': [],
        'recall': [],
        'precision': [],
    }

    for i in range(args.repeat_n):
        trainer = GroupGCN(args)
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
