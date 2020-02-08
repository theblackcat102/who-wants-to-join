from datetime import datetime
from src.layers import StackedGCNAmazon, StackedGCNAmazonV2
from src.amazon import AmazonCommunity
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
from src.skipgram import SkipGramNeg, sample_walks
from src.utils import dict2table, confusion, str2bool, TMP_WRITER_PATH


class GroupGCN():
    def __init__(self, args):
        dataset = AmazonCommunity(cutoff=args.maxhop, min_size=args.min_size,
                                  max_size=args.max_size)

        # make sure each runs share the same results
        if osp.exists('amazon_hete_shuffle_idx.pkl'):
            with open('amazon_hete_shuffle_idx.pkl', 'rb') as f:
                shuffle_idx = pickle.load(f)
            assert len(shuffle_idx) == len(dataset)
        else:
            shuffle_idx = [idx for idx in range(len(dataset))]
            random.shuffle(shuffle_idx)
            with open('amazon_hete_shuffle_idx.pkl', 'wb') as f:
                pickle.dump(shuffle_idx, f)

        with open('data/amazon/cat2id.pkl', 'rb') as f:
            cat2id = pickle.load(f)

        self.category_size = len(cat2id)
        print('max: ', max(shuffle_idx))
        print('len: ', len(dataset))
        dataset = dataset[shuffle_idx]

        split_pos = int(len(dataset)*0.7)
        train_idx = shuffle_idx[:split_pos]
        valid_idx_ = shuffle_idx[split_pos:]
        test_pos = int(len(valid_idx_)*0.333)
        test_idx = valid_idx_[:test_pos]
        valid_idx = valid_idx_[test_pos:]

        self.train_dataset = dataset[train_idx]
        self.test_dataset = dataset[test_idx]
        self.valid_dataset = dataset[valid_idx]

        self.args = args
        if args.writer is True:
            self.log_path = osp.join(
                "logs", "amazon",
                'amazon_hete_multi_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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

    def evaluate(self, dataloader, model):
        model.eval()
        recalls = []
        precisions = []
        losses = []
        B = args.batch_size
        user_size = len(self.train_dataset.user2id)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        print('Validation')
        with torch.no_grad():
            for val_data in tqdm(dataloader, dynamic_ncols=True):
                x, edge_index, label_mask = (val_data.x, val_data.edge_index,
                                             val_data.label_mask)
                y = val_data.y.cpu()
                pred_mask = val_data.label_mask
                label = val_data.y.unsqueeze(-1).cuda().float()
                if isinstance(model, StackedGCNAmazonV2):
                    pred, _ = model(edge_index.cuda(), x.cuda(),
                                 label_mask.cuda())
                else:
                    pred, _ = model(edge_index.cuda(), x.cuda())
                loss = criterion(pred[pred_mask], label[pred_mask])
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
                # consider last batch in dataloader for smaller batch size
                y_pred_ = y_pred[:x.size(0)]
                y_target_ = y_target[:x.size(0)]
                TP, FP, TN, FN = confusion(y_pred_, y_target_)

                recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
                precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
                precisions.append(precision)
                recalls.append(recall)
                losses.append((loss.sum()/val_data.num_graphs).item())

        avg_recalls = np.mean(recalls)
        avg_precisions = np.mean(precisions)
        if (avg_recalls+avg_precisions) == 0:
            f1 = np.nan
        else:
            f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)
        model.train()
        return f1, avg_recalls, avg_precisions, np.mean(losses)

    def train(self, epochs=200):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        args = self.args
        train_size = len(self.train_dataset.processed_file_idx)
        val_size = len(self.valid_dataset.processed_file_idx)
        print((len(set(self.valid_dataset.processed_file_idx+self.train_dataset.processed_file_idx))))
        print(train_size+val_size)
        assert (len(set(self.valid_dataset.processed_file_idx+self.train_dataset.processed_file_idx))) == (train_size+val_size)

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
        if args.label_mask:
            print('v2')
            model_class = StackedGCNAmazonV2
        else:
            model_class = StackedGCNAmazon
        model = model_class(len(self.train_dataset.user2id),
                            category_size=self.category_size,
                            user_dim=args.user_dim,
                            category_dim=args.cat_dim,
                            input_channels=args.input_dim,
                            layers=args.layers,
                            dropout=args.dropout)

        if args.pretrain:
            model = self.pretrain_embeddings(model, 64, epoch_num=2)
        model = model.cuda()

        if args.pos_weight <= 0:
            weight = 50  # default
            args.pos_weight = weight
        else:
            weight = args.pos_weight
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'max')
        print('weight : ', weight)
        pos_weight = torch.ones([1])*weight
        self.pos_weight = pos_weight.cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        xentropy = torch.nn.CrossEntropyLoss()

        n_iter = 0
        best_f1 = 0
        self.writer.add_text('Text', dict2table(vars(args)), 0)

        with tqdm(total=len(train_loader)*epochs, dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                for data in train_loader:
                    optimizer.zero_grad()
                    x, edge_index, label_mask = (data.x, data.edge_index,
                                                 data.label_mask)
                    x = x.cuda()
                    edge_index = edge_index.cuda()
                    label_mask = label_mask.cuda()
                    pred_mask = data.label_mask.cuda() == 1
                    label = data.y.unsqueeze(-1).cuda().float()
                    if isinstance(model, StackedGCNAmazonV2):
                        node_pred, type_pred = model(edge_index, x, label_mask)
                    else:
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
                    f1, recalls, precisions, loss = self.evaluate(valid_loader,
                                                                  model)
                    scheduler.step(f1)
                    self.writer.add_scalar("Valid/F1", f1, n_iter)
                    self.writer.add_scalar("Valid/Recalls", recalls, n_iter)
                    self.writer.add_scalar("Valid/Precisions", precisions,
                                           n_iter)
                    self.writer.add_scalar("Valid/avg_BCEWithLogitsLoss", loss,
                                           n_iter)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_checkpoint = {
                            'epoch': epoch+1,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'f1': f1
                        }

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

        print("Testing")
        self.save_checkpoint(best_checkpoint,
                             self.save_path,
                             "best")
        model.load_state_dict(best_checkpoint["model"])
        f1, recalls, precisions, loss = self.evaluate(test_loader, model)
        self.writer.add_scalar("Test/F1", f1, n_iter)
        self.writer.add_scalar("Test/Recalls", recalls, n_iter)
        self.writer.add_scalar("Test/Precisions", precisions, n_iter)
        self.writer.add_scalar("Test/avg_BCEWithLogitsLoss", loss, n_iter)
        self.writer.flush()

        # clean tmp_writer
        if args.writer is False:
            shutil.rmtree(TMP_WRITER_PATH, ignore_errors=True)
        return f1, recalls, precisions, loss

    def pretrain_embeddings(self, model, batch_size, epoch_num=1, neg_num=20):
        import torch.optim as optim
        print('Pretrain embeddings')
        node_types = {
            0: model.user_size,
            1: model.category_size,
        }
        embeddings = {}
        for node_type, (embed_size, dim) in node_types.items():
            if osp.exists(osp.join(args.pretrain_weight,
                                   'random_walk_{}.pt'.format(node_type))):
                samples = torch.load(
                    osp.join(args.pretrain_weight,
                             'random_walk_{}.pt'.format(node_type)))['samples']
            else:
                samples = sample_walks(self.train_dataset, neg_num, batch_size,
                                       node_type, embed_size, cpu_count=40)
                torch.save({'samples': samples},
                           osp.join(self.save_path,
                                    'random_walk_{}.pt'.format(node_type)))

            skip_model = SkipGramNeg(embed_size, dim).cuda()
            optimizer = optim.SGD(skip_model.parameters(), lr=1e-5,
                                  weight_decay=1e-9)
            iteration = list(range(len(self.train_dataset)))
            total_idx = 0
            with tqdm(total=len(iteration)*epoch_num) as pbar:
                for e in range(epoch_num):
                    random.shuffle(samples)
                    for idx, sample in enumerate(samples):
                        context, target, negative = sample
                        context = torch.from_numpy(context).long().cuda()
                        target = torch.from_numpy(target).long().cuda()
                        negative = torch.from_numpy(negative).long().cuda()
                        loss = skip_model(target, context, negative)

                        loss.backward()
                        optimizer.step()
                        pbar.set_description(
                            "loss {:.4f}, iter {}".format(loss.item(),
                                                          total_idx))
                        if self.writer is not None:
                            self.writer.add_scalar(
                                'Skipgram/loss/%d' % node_type,
                                loss.item(), total_idx)
                        total_idx += 1
                        pbar.update(1)
            del samples
            skip_model = skip_model.cpu()
            embeddings[node_type] = skip_model.input_emb.weight
            if node_type == 0:
                print('transfer user embeddings')
                model.embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 1:
                print('transfer category embeddings')
                model.category_embeddings.weight.data = skip_model.input_emb.weight.data
        torch.save(embeddings, osp.join(self.save_path, 'embeddings.pt'))
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
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pos-weight', type=float, default=-1)
    parser.add_argument('--eval', type=int, default=10)
    parser.add_argument('--save', type=int, default=50)
    parser.add_argument('--pretrain', type=str2bool, nargs='?', default=False)
    parser.add_argument('--pretrain-weight', type=str, default='')
    parser.add_argument('--label-mask', type=str2bool, nargs='?',
                        default=False, help="add label_mask as input")
    # model parameters
    parser.add_argument('--user-dim', type=int, default=16)
    parser.add_argument('--cat-dim', type=int, default=8)
    parser.add_argument('--input-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--repeat-n', type=int, default=1)
    # debug
    parser.add_argument('--writer', type=str2bool, nargs='?', default=True)

    args = parser.parse_args()
    values = {
        'f1': [],
        'recall': [],
        'precision': [],
        'loss': []
    }

    for i in range(args.repeat_n):
        trainer = GroupGCN(args)
        f1, recalls, precisions, loss = trainer.train(epochs=args.epochs)
        values['f1'].append(f1)
        values['recall'].append(recalls)
        values['precision'].append(precisions)
        values['loss'].append(loss)

    results = {}
    for key, value in values.items():
        results['avg_'+key] = np.mean(value)
        results['std_'+key] = np.std(value)
    results['results'] = values
    results['arguments'] = vars(args)
    with open('amazon_hete_multi_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)
