from datetime import datetime
from src.layers import StackedGCNMeetup
from src.meetup import Meetup, locations_id, MEETUP_FOLDER
from torch_geometric.data import DataLoader
import random
import shutil
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import os.path as osp
import numpy as np
import multiprocessing as mp
from src.skipgram import SkipGramNeg, sample_walks
from src.utils import dict2table, confusion, str2bool, TMP_WRITER_PATH


class GroupGCN():
    def __init__(self, args):
        dataset = Meetup(city_id=locations_id[args.dataset],
                         cutoff=args.maxhop, min_size=args.min_size,
                         max_size=args.max_size)

        # make sure each runs share the same results
        if osp.exists(args.dataset+'_shuffle_idx.pkl'):
            with open(args.dataset+'_shuffle_idx.pkl', 'rb') as f:
                shuffle_idx = pickle.load(f)
            assert len(shuffle_idx) == len(dataset)
        else:
            shuffle_idx = [idx for idx in range(len(dataset))]
            split_pos = int(len(dataset)*0.7)
            train_idx = shuffle_idx[:split_pos]
            random.shuffle(train_idx)
            shuffle_idx[:split_pos] = train_idx
            # print(shuffle_idx[split_pos: split_pos+10])
            with open(args.dataset+'_shuffle_idx.pkl', 'wb') as f:
                pickle.dump(shuffle_idx, f)

        with open(osp.join(MEETUP_FOLDER, 'topic2id.pkl'), 'rb') as f:
            topic2id = pickle.load(f)
        with open(osp.join(MEETUP_FOLDER, 'cat2id.pkl'), 'rb') as f:
            cat2id = pickle.load(f)
        # with open(osp.join(MEETUP_FOLDER, 'group2topic.pkl'), 'rb') as f:
        #     group2topic = pickle.load(f)

        self.category_size = len(cat2id)
        self.topic_size = len(topic2id)
        self.group_size = len(dataset.group2id)
        del cat2id
        del topic2id

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

        if args.writer is True:
            self.log_path = osp.join(
                "logs", "meetup",
                args.dataset+'_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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
        B = args.batch_size
        user_size = len(self.train_dataset.user2id)
        print('Validation')
        with torch.no_grad():
            y_pred = torch.FloatTensor(B, user_size)
            y_target = torch.FloatTensor(B, user_size)
            for val_data in tqdm(dataloader, dynamic_ncols=True):
                x, edge_index = val_data.x, val_data.edge_index
                y = val_data.y.cpu()
                pred_mask = val_data.label_mask.cpu()
                pred = model(edge_index.cuda(), x.cuda())
                pred = torch.sigmoid(pred).cpu()
                # y = y[pred_mask]
                y_pred.zero_()
                y_target.zero_()
                for idx, batch_idx in enumerate(val_data.batch):
                    if pred_mask[idx] == 1:
                        if y[idx] == 1:
                            y_target[batch_idx.data, x[idx][0]] = 1
                        if pred[idx] > 0.5:
                            y_pred[batch_idx.data, x[idx][0]] = 1
                # consider last batch in dataloader for smaller batch size
                y_pred_ = y_pred[:x.size(0)]
                y_target_ = y_target[:x.size(0)]
                TP, FP, TN, FN = confusion(y_pred_, y_target_)

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
        train_val_set_size = len(set(self.valid_dataset.processed_file_idx +
                                     self.train_dataset.processed_file_idx))

        assert train_val_set_size == (train_size+val_size)

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)

        model = StackedGCNMeetup(len(self.train_dataset.user2id),
                                 category_size=self.category_size,
                                 topic_size=self.topic_size,
                                 group_size=self.group_size,
                                 input_channels=args.input_dim,
                                 layers=args.layers,
                                 dropout=args.dropout)
        if args.pretrain:
            model = self.pretrain_embeddings(args, model, 256, epoch_num=10)

        model = model.cuda()

        position_weight = {
            'NY': 50,
            'SF': 50,
        }
        if args.pos_weight <= 0:
            weight = 50  # default
            if args.dataset in position_weight:
                weight = position_weight[args.dataset]
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
                    pred_mask = data.label_mask.cuda() == 1
                    label = data.y.unsqueeze(-1).cuda().float()
                    output = model(edge_index, x)
                    # pred_ = torch.sigmoid(output) > 0.5
                    # print(pred_.sum(), label.sum(), pred_mask.sum())
                    loss = criterion(output[pred_mask], label[pred_mask])
                    loss.backward()

                    optimizer.step()
                    self.writer.add_scalar(
                            "Train/BCEWithLogitsLoss", loss.item(), n_iter)
                    pbar.update(1)
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(loss.item(), epoch))
                    n_iter += 1

                if epoch % args.eval == 0 or epoch == epochs-1:
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

                if epoch % args.save == 0 or epoch == epochs-1:
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
        self.save_checkpoint(best_checkpoint, self.save_path, "best")
        model.load_state_dict(best_checkpoint["model"])
        f1, recalls, precisions = self.evaluate(test_loader, model)

        self.writer.add_scalar("Test/F1", f1, n_iter)
        self.writer.add_scalar("Test/Recalls", recalls, n_iter)
        self.writer.add_scalar("Test/Precisions", precisions, n_iter)
        self.writer.flush()

        # clean tmp_writer
        if args.writer is False:
            shutil.rmtree(TMP_WRITER_PATH, ignore_errors=True)

    def pretrain_embeddings(self, args, model, batch_size, epoch_num=1,
                            neg_num=20):
        # from torch.optim.lr_scheduler import StepLR
        import torch.optim as optim
        print('Pretrain embeddings')
        node_types = {
            0: model.user_size,
            1: model.topic_size,
            2: model.category_size,
            4: model.group_size,
        }
        embeddings = {}
        for node_type, (embed_size, dim) in node_types.items():
            if osp.exists(osp.join(args.pretrain_weight,
                                   'random_walk_{}.pt'.format(node_type))):
                samples = torch.load(
                    osp.join(args.pretrain_weight,
                             'random_walk_{}.pt'.format(node_type)))['samples']
            # cost a lot of time
            else:
                if node_type == 0:
                    iter_step = 10
                elif node_type == 1:
                    iter_step = 5
                elif node_type == 2:
                    iter_step = 5
                elif node_type == 4:
                    iter_step = 3
                samples = sample_walks(self.train_dataset, neg_num, batch_size,
                                       node_type, embed_size,
                                       cpu_count=mp.cpu_count()-2,
                                       iter_steps=iter_step, parallel=True)

            skip_model = SkipGramNeg(embed_size, dim)
            skip_model = skip_model.cuda()
            optimizer = optim.SGD(skip_model.parameters(), lr=1e-5,
                                  weight_decay=1e-9)
            iteration = list(range(len(self.train_dataset)))
            total_idx = 0
            print('sampling: {}'.format(len(samples)))
            torch.save({'samples': samples},
                       osp.join(self.save_path,
                                'random_walk_{}.pt'.format(node_type)))
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
                            "loss {:.4f}, iter {}".format(
                                loss.item(), total_idx))
                        self.writer.add_scalar('Skipgram/loss/%d' % node_type,
                                               loss.item(),
                                               total_idx)
                        total_idx += 1
                        pbar.update(1)
            if total_idx == 0:
                continue
            skip_model = skip_model.cpu()
            embeddings[node_type] = skip_model.input_emb.weight
            if node_type == 0:
                print('transfer user embeddings')
                model.embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 1:
                print('transfer topic embeddings')
                model.topic_embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 2:
                print('transfer category embeddings')
                model.category_embeddings.weight.data = skip_model.input_emb.weight.data
            elif node_type == 4:
                print('transfer group embeddings')
                model.group_embeddings.weight.data = skip_model.input_emb.weight.data
        torch.save(embeddings, osp.join(self.save_path, 'embeddings.pt'))
        return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Deepset Recommendation Model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='SF',
                        choices=['NY', 'SF'])
    parser.add_argument('--min-size', type=int, default=5)
    parser.add_argument('--max-size', type=int, default=100)
    parser.add_argument('--pred-ratio', type=float, default=0.8)
    parser.add_argument('--maxhop', type=int, default=2)
    # training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--pos-weight', type=float, default=-1)
    parser.add_argument('--eval', type=int, default=10)
    parser.add_argument('--save', type=int, default=50)
    parser.add_argument('--pretrain', type=str2bool, nargs='?', default=False)
    parser.add_argument('--pretrain-weight', type=str, default='')
    # model parameters
    parser.add_argument('--input-dim', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=[8, 8, 8])
    # debug
    parser.add_argument('--writer', type=str2bool, nargs='?', default=True)

    args = parser.parse_args()

    trainer = GroupGCN(args)
    trainer.train(epochs=args.epochs)
