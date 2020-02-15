
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool
from tqdm import tqdm


class GNNGraphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        model (model): the model which needs to be pretrained
        input_channels (int): input channels of GCN in model
        output_channels (int): output channels of GCN in model
        graph_pooling (str): sum, mean, max, attention, set2set (default: mean)

    See https://arxiv.org/pdf/1905.12265.pdf
    """
    def __init__(self, model, input_channels=8, output_channels=1,
                 graph_pooling="mean"):
        super(GNNGraphpred, self).__init__()
        # the model
        self.gnn = model
        self.unuse_layer = self.gnn.layers[-1]
        self.gnn.layers = self.gnn.layers[:-1]

        # different kind of graph pooling
        self.pool = global_mean_pool

        # for graph-level binary classification
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.graph_pred_linear = torch.nn.Linear(self.input_channels,
                                                 self.output_channels)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, edges, features, label_masks, batch):
        # predict the graph
        node_representation = self.gnn(edges, features, label_masks)
        pool_feature = self.pool(node_representation, batch)
        return self.graph_pred_linear(pool_feature)

    def get_model(self):
        return self.gnn


class GNNGraphpredTrainer(object):
    def __init__(self, graphpred_model, datasets, writer, batch_size=8,
                 epoch_num=1):
        self.graphpred_model = graphpred_model.cuda()
        self.datasets = datasets
        self.writer = writer
        self.optimizer = optim.SGD(self.graphpred_model.parameters(), lr=1e-5,
                                   weight_decay=1e-9)
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.criterion = nn.L1Loss()

    def train(self):
        dataloader = DataLoader(self.datasets,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=4)
        total_iter = 0
        with tqdm(total=len(dataloader)*self.epoch_num, dynamic_ncols=True) as pbar:
            for epoch in range(self.epoch_num):
                for i, data in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    x, edge_index, label_mask = (data.x, data.edge_index,
                                                 data.label_mask)
                    batch = data.batch
                    x = x.cuda()
                    edge_index = edge_index.cuda()
                    label_mask = label_mask.cuda()
                    batch = batch.cuda()
                    output = self.graphpred_model(edge_index, x, label_mask,
                                                  batch)
                    # create y: y = avg degree of Data (edges/nodes)
                    data_list = data.to_data_list()
                    y = torch.zeros(output.size())
                    for j, data in enumerate(data_list):
                        y[j] = len(data.edge_index[1])/len(data.x)
                    # y = data.y.cuda()
                    # y = global_add_pool(y, batch)
                    y = y.to(output.dtype).to(output.device)
                    # mse loss
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_description(
                        "loss {:.4f}, epoch {}".format(
                            loss.item(), epoch))
                    self.writer.add_scalar('Graph-loss', loss.item(),
                                           total_iter)
                    total_iter += 1
                    pbar.update()


if __name__ == "__main__":
    import os.path as osp
    import pickle
    import random
    import shutil
    from datetime import datetime
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    from src.layers import StackedGCNMeetupV2
    from src.meetup import Meetup, locations_id, MEETUP_FOLDER
    from src.utils import str2bool, TMP_WRITER_PATH
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pos-weight', type=float, default=30)
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
    category_size = 40
    topic_size = len(topic2id)
    group_size = len(dataset.group2id)
    del cat2id
    del topic2id

    dataset = dataset[shuffle_idx]

    split_pos = int(len(dataset)*0.7)
    train_idx = shuffle_idx[:split_pos]
    valid_idx_ = shuffle_idx[split_pos:]
    test_pos = int(len(valid_idx_)*0.333)
    test_idx = valid_idx_[:test_pos]
    valid_idx = valid_idx_[test_pos:]

    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    valid_dataset = dataset[valid_idx]
    if args.writer is True:
        log_path = osp.join(
            "logs", "meetup",
            args.dataset+'_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        shutil.rmtree(TMP_WRITER_PATH, ignore_errors=True)
        log_path = TMP_WRITER_PATH
    writer = SummaryWriter(log_dir=log_path)
    model = StackedGCNMeetupV2(user_size=len(train_dataset.user2id),
                               category_size=category_size,
                               topic_size=topic_size,
                               group_size=group_size,
                               input_channels=args.input_dim,
                               layers=args.layers,
                               dropout=args.dropout)
    gnn_graph_pred = GNNGraphpred(model, input_channels=8, output_channels=1)
    trainer = GNNGraphpredTrainer(gnn_graph_pred, train_dataset, writer,
                                  epoch_num=10)
    trainer.train()
