import torch
from torch import nn
import pickle
import numpy as np
from dataset.aminer import Aminer
import numpy as np
import pickle, os, glob
import os.path as osp

from tqdm import tqdm
from sklearn.metrics import f1_score
from src.utils import dict2table, confusion, str2bool
import random, json
from baseline.models.agree import AGREE
from baseline.dataset import DatasetConvert, AgreeDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import defaultdict

steps_tracker = defaultdict(int)

def train_epoch(dataloader, model, optimizer, writer=None, postfix='user'):
    avg_loss, cnt = 0, 0
    # with tqdm(total=len(dataloader), dynamic_ncols=True) as pbar:
    for batch in dataloader:
        inputs, pos, neg, group_ids = batch
        inputs, pos, neg, group_ids = inputs.cuda(), pos.cuda(), neg.cuda(), group_ids.cuda()
        # seq   B x user-size  B x user-size
        group_ids = group_ids.unsqueeze(-1)

        pos_predict = model(inputs, group_ids, pos)
        neg_predict = model(inputs, group_ids, neg)
        optimizer.zero_grad()
        loss = torch.mean((pos_predict - neg_predict -1) **2)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        cnt += 1
        if writer is not None:
            writer.add_scalar('loss/'+postfix, loss.item(), steps_tracker[postfix])
            steps_tracker[postfix] += 1
        # pbar.update(1)
        # pbar.set_description("loss={:.4f}".format(loss.item()))
    avg_loss /= cnt
    return model, optimizer, avg_loss




def evaluate_score(test_dataloader, model):
    precisions, recalls = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels, masked_target, group_id = batch
            inputs, masked_target, group_id = inputs.cuda(), masked_target.cuda(), group_id.cuda()
            y_pred = model.predict_rank(inputs, group_id.unsqueeze(-1), masked_target, top_k=args.top_k)

            TP, FP, TN, FN = confusion(y_pred, labels)
            recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
            precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
            # print(precision, recall)
            precisions.append(precision)
            recalls.append(recall)
 
    avg_recalls = np.mean(recalls)
    avg_precisions = np.mean(precisions)
    f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)
    return f1, avg_recalls, avg_precisions

if __name__ == "__main__":
    user_size = 399211
    import argparse
    parser = argparse.ArgumentParser(
        description='CF rank method for group expansion')
    parser.add_ = parser.add_argument
    parser.add_('--top-k', type=int, default=5)
    parser.add_('--city', type=str, default='SF',
                        choices=['NY', 'SF'])
    parser.add_('--dataset', type=str, default='aminer',
                    choices=['meetup', 'aminer'])
    parser.add_('--user-node', type=int, default=0, 
                help='integer which user node id is represented in')
    parser.add_('--epochs', type=int, default=100, 
                help='training epochs')
    parser.add_('--seq-len', type=int, default=6, 
                help='known users seq len')
    parser.add_('--neg-sample', type=int, default=5, 
                help='negative users')
    parser.add_('--rank-margin', type=float, default=1.0, 
                help='negative loss margin')
    parser.add_('--batch-size', type=int, default=32,
                help='known users seq len')
    parser.add_('--lr', type=float, default=0.00001, 
                help='training lr')
    parser.add_('--embeddings', type=str,
                help='graphvite embedding pickle')
    args = parser.parse_args()

    with open('graphvite_embeddings/aminer_deduplicate_train_edgelist.txt-64-DeepWalk.pkl', 'rb') as f:
        graphvite_embeddings = pickle.load(f)

    dataset = Aminer()
    data_size = len(dataset)
    if os.path.exists('.cache/{}_user2idx.pkl'.format(str(dataset))):
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'rb') as f:
            user2idx, idx2user, group2id = pickle.load(f)
    else:
        user2idx, idx2user, group2id = reindex_name2id(graphvite_embeddings, dataset)
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'wb') as f:
            pickle.dump((user2idx, idx2user, group2id), f)
    writer = None

    log_path = osp.join(
        "logs", "agree",
        'model_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_path, exist_ok=True)
    with open(log_path+'/params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    writer = SummaryWriter(log_dir=log_path)
    save_path = osp.join(log_path, "models")
    os.makedirs(save_path, exist_ok=True)


    user_size = len(idx2user)
    # print(len(user2idx), len(idx2user))
    train_split, val, test = int(data_size*0.7), int(data_size*0.1), int(data_size*0.2)

    indexes = np.array(list(range(data_size)), dtype=np.long)[train_split+val:]
    test_dataset = dataset[list(indexes)]
    test_dataset = DatasetConvert(test_dataset, user_size, user2idx, group2id, max_seq=6)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False)

    indexes = np.array(list(range(data_size)), dtype=np.long)[train_split:train_split+val]
    val_dataset = dataset[list(indexes)]
    val_dataset = DatasetConvert(val_dataset, user_size, user2idx, group2id, max_seq=6)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers=4, shuffle=False)


    indexes = np.array(list(range(data_size)), dtype=np.long)[:train_split]
    train_dataset = dataset[list(indexes)]
    group_dataset = AgreeDataset(train_dataset, user_size, user2idx, group2id, max_seq=6, mode='group')
    group_dataloader = torch.utils.data.DataLoader(group_dataset, batch_size=args.batch_size, num_workers=4)
    user_dataset = AgreeDataset(train_dataset, user_size, user2idx, group2id, max_seq=6, mode='user')
    user_dataloader = torch.utils.data.DataLoader(group_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    model = AGREE(len(user2idx), len(group2id), 64, 0.1)
    embeddings = model.embeddings
    group_embeddings = model.group_embed
    name2id = graphvite_embeddings['name2id']
    vectors = graphvite_embeddings['embedding']
    embed_weight = embeddings.weight.data.numpy()
    for name, id_ in name2id.items():
        node_type, node_id = name.split('_')
        node_type, node_id = int(node_type), int(node_id)
        if node_type == 0:
            vector = vectors[id_]
            embed_weight[user2idx[node_id]] = vector
    model.embeddings.weight.data.copy_(torch.from_numpy(embed_weight))
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for e in tqdm(range(args.epochs), dynamic_ncols=True):
        model, optimizer, avg_loss_gp = train_epoch(user_dataloader, model, optimizer, writer, postfix='user')
        model, optimizer, avg_loss_usr = train_epoch(group_dataloader, model, optimizer, writer, postfix='group')
        # print('gp: {:.3f}, usr: {:.3f}'.format(avg_loss_usr, avg_loss_usr))

        f1, avg_recalls, avg_precisions = evaluate_score(val_dataloader, model)            
        if writer != None:
            writer.add_scalar('val/f1', f1, e)
            writer.add_scalar('val/recalls', avg_recalls, e)
            writer.add_scalar('val/precision', avg_precisions, e)
        model.train()

        scheduler.step()

    torch.save(model, os.path.join(save_path,'model.pt'))

    f1, avg_recalls, avg_precisions = evaluate_score(test_dataloader, model)

    if writer != None:
        writer.add_scalar('test/f1', f1, 0)
        writer.add_scalar('test/recalls', avg_recalls, 0)
        writer.add_scalar('test/precision', avg_precisions, 0)
        writer.flush()

    print( f1, avg_recalls, avg_precisions)