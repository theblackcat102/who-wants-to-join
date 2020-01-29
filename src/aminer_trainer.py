from datetime import datetime
from src.layers import StackedGCNDBLP
from src.aminer import Aminer
from torch_geometric.data import DataLoader
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import os.path as osp
import numpy as np

from src.utils import dict2table


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
        dataset_name = 'amazon'
        dataset = Aminer(cutoff=args.maxhop,
            min_size=args.min_size, max_size=args.max_size)

        # make sure each runs share the same results
        if osp.exists('dblp_hete_shuffle_idx.pkl'):
            with open('dblp_hete_shuffle_idx.pkl', 'rb') as f:
                shuffle_idx = pickle.load(f)
        else:
            shuffle_idx = [idx for idx in range(len(dataset))]
            random.shuffle(shuffle_idx)
            with open('dblp_hete_shuffle_idx.pkl', 'wb') as f:
                pickle.dump(shuffle_idx, f)


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

        # print(len(set(
        #     self.valid_dataset.processed_file_idx +
        #     self.train_dataset.processed_file_idx)))
        # print(len(self.valid_dataset)+ len(self.train_dataset))

        self.args = args

        self.log_path = osp.join(
            "logs", "gcn",
            'dblp_hete_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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
        user_size = len(self.train_dataset.user2id)
        print('Validation')
        with torch.no_grad():
            y_pred = torch.FloatTensor(B, user_size)
            y_target = torch.FloatTensor(B, user_size)
            for val_data in tqdm(dataloader, dynamic_ncols=True):
                x, edge_index = val_data.x, val_data.edge_index
                y = val_data.y
                pred_mask = val_data.label_mask
                pred = model(edge_index.cuda(), x.cuda())
                pred = pred.cpu()
                # y = y[pred_mask]
                y_pred.zero_()
                y_target.zero_()

                for idx, batch_idx in enumerate(val_data.batch):
                    if pred_mask[idx] == 1:
                        if y[idx] == 1:
                            y_target[batch_idx.data, x[idx][0]] = 1
                        if pred[idx] > 0.5:
                            y_pred[batch_idx.data, x[idx][0]] = 1

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
        args = self.args
        train_size = len(self.train_dataset.processed_file_idx)
        val_size = len(self.valid_dataset.processed_file_idx)
        assert (len(set(self.valid_dataset.processed_file_idx+self.train_dataset.processed_file_idx))) == (train_size+val_size)

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=8,
                                  shuffle=True)
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=8,
                                  shuffle=False)

        test_loader = DataLoader(self.test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=8,
                                 shuffle=False)
        # with open('aminer/preprocess_dblp.pkl', 'rb') as f:
        #     dblp = pickle.load(f)
        # author2id = dblp['author2id']
        # paper2id = dblp['paper2id']
        # conf2id = dblp['conf2id']

        model = StackedGCNDBLP(
                           author_size= 874608,#len(author2id),
                           paper_size=3605603,#len(paper2id),
                           conf_size=12770,#len(conf2id),
                           user_dim=args.author_dim,
                           paper_dim=args.paper_dim,
                           conf_dim=args.conf_dim,
                           input_channels=args.input_dim,
                           layers=args.layers,
                           dropout=args.dropout)
        model = model.cuda()

        if args.pos_weight <= 0:
            weight = 50  # default
            args.pos_weight = weight
        else:
            weight = args.pos_weight
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=5e-4)
        print('weight : ', weight)
        pos_weight = torch.ones([1])*weight
        pos_weight = pos_weight.cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
                    pred_mask = data.label_mask.cuda()
                    label = data.y.unsqueeze(-1).cuda().float()
                    output = model(edge_index, x)

                    loss = criterion(output[pred_mask], label[pred_mask])
                    loss.backward()

                    optimizer.step()
                    self.writer.add_scalar(
                        "Train/BCEWithLogitsLoss", loss.item(), n_iter)
                    pbar.update(1)
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(loss.item(), epoch))
                    n_iter += 1

                if epoch % args.eval == 0:
                    print('Epoch: ', epoch)
                    f1, recalls, precisions = self.evaluate(valid_loader,
                                                            model)
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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pos-weight', type=float, default=-1)
    parser.add_argument('--eval', type=int, default=10)
    parser.add_argument('--save', type=int, default=50)
    # model parameters
    parser.add_argument('--author-dim', type=int, default=16)
    parser.add_argument('--paper-dim', type=int, default=16)
    parser.add_argument('--conf-dim', type=int, default=8)
    parser.add_argument('--input-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=[32, 32])

    args = parser.parse_args()

    trainer = GroupGCN(args)
    trainer.train(epochs=args.epochs)
