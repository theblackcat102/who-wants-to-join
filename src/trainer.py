from datetime import datetime
from src.layers import StackedGCN
from src.dataset import SNAPCommunity
from torch_geometric.data import DataLoader
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import numpy as np


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and
        `truth` tensors, i.e. the amount of positions where the values of
        `prediction` and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


class GroupGCN():
    def __init__(self, args):
        dataset = SNAPCommunity(args.dataset, cutoff=2)
        # make sure each runs share the same results
        if os.path.exists(args.dataset+'_shuffle_idx.pkl'):
            with open(args.dataset+'_shuffle_idx.pkl', 'rb') as f:
                shuffle_idx = pickle.load(f)
        else:
            shuffle_idx = [idx for idx in range(len(dataset))]
            random.shuffle(shuffle_idx)
            with open(args.dataset+'_shuffle_idx.pkl', 'wb') as f:
                pickle.dump(shuffle_idx, f)

        dataset = dataset[shuffle_idx]

        split_pos = int(len(dataset)*0.7)
        train_idx = shuffle_idx[:split_pos]
        valid_idx_ = shuffle_idx[split_pos:]
        test_pos = int(len(valid_idx_)*0.666)
        test_idx = valid_idx_[:test_pos]
        valid_idx = valid_idx_[test_pos:]

        self.train_dataset = dataset[train_idx]
        self.test_dataset = dataset[test_idx]
        self.valid_dataset = dataset[valid_idx]

        # print(len(set(
        #     self.valid_dataset.processed_file_idx +
        #     self.train_dataset.processed_file_idx)))
        # print(len(self.valid_dataset)+ len(self.train_dataset))

        self.args = args

        log_path = os.path.join(
            "logs", "gcn", datetime.now().strftime("%m-%d-%H-%M-%S"))
        self.writer = SummaryWriter(log_dir=log_path)
        print('finish init')

    def train(self, epochs=200):
        args = self.args
        train_loader = DataLoader(
            self.train_dataset, args.batch_size, shuffle=True)
        valid_loader = DataLoader(
            self.valid_dataset, args.batch_size, shuffle=False)

        model = StackedGCN(
            len(self.train_dataset.user2id), args.input_dim, 1, args.layers,
            args.dropout)
        model = model.cuda()

        B = args.batch_size
        user_size = len(self.train_dataset.user2id)

        position_weight = {
            'amazon': 80,
            'dblp': 100,
        }
        weight = 100  # default
        if args.dataset in position_weight:
            weight = position_weight[args.dataset]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=5e-4)
        print('weight : ', weight)
        pos_weight = torch.ones([1])*weight
        pos_weight = pos_weight.cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_iter = 0
        for epoch in range(epochs):
            model.train()
            with tqdm(total=len(train_loader), dynamic_ncols=True) as pbar:
                for data in train_loader:
                    optimizer.zero_grad()
                    x, edge_index = data.x, data.edge_index
                    x = x.cuda()
                    edge_index = edge_index.cuda()
                    label = data.y.unsqueeze(-1).cuda().float()
                    output = model(edge_index, x)
                    loss = criterion(output, label)
                    loss.backward()

                    optimizer.step()
                    self.writer.add_scalar(
                        "Train/BCEWithLogitsLoss", loss.item(), n_iter)
                    pbar.update(1)
                    pbar.set_description("loss {:.4f}".format(loss.item()))
                    n_iter += 1

            if epoch % args.eval == 0:
                print('Epoch: ', epoch)
                model.eval()
                recalls = []
                precisions = []
                print('Validation')
                with torch.no_grad():
                    y_pred = torch.zeros(B, user_size)
                    y_target = torch.zeros(B, user_size)
                    for val_data in tqdm(valid_loader, dynamic_ncols=True):
                        x, edge_index = val_data.x, val_data.edge_index
                        y = val_data.y
                        pred = model(edge_index.cuda(), x.cuda())
                        pred = pred.cpu()
                        y_pred.zero_()
                        y_target.zero_()
                        for idx, batch_idx in enumerate(val_data.batch):
                            if y[idx] == 1:
                                y_target[batch_idx, x[idx]] = 1
                            if pred[idx] > 0.5:
                                y_pred[batch_idx, x[idx]] = 1
                        TP, FP, TN, FN = confusion(y_pred, y_target)
                        recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
                        precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
                        precisions.append(precision)
                        recalls.append(recall)

                avg_recalls = np.mean(recalls)
                avg_precisions = np.mean(precisions)
                f1 = (
                    2*(avg_recalls*avg_precisions) /
                    (avg_recalls+avg_precisions))
                self.writer.add_scalar(
                        "Valid/F1", f1, n_iter)
                print(f1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Deepset Recommendation Model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='amazon',
                        choices=['amazon', 'dblp', 'youtube'])
    # training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval', type=int, default=10)
    # model parameters
    parser.add_argument('--input-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', type=list, default=[16, 16, 16])
    args = parser.parse_args()

    trainer = GroupGCN(args)
    trainer.train(epochs=args.epochs)
