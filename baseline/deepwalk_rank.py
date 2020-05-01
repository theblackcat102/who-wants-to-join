import torch
# torch.manual_seed(666)
from torch import nn
import pickle
import numpy as np
from dataset.aminer import Aminer
from dataset.meetup import Meetup, locations_id
import numpy as np
import pickle, os, glob
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.utils import dict2table, confusion, str2bool
import random, json
from datetime import datetime
from baseline.models.deepwalk_clf import DeepwalkClf, DeepwalkAttnClf
from baseline.dataset import DatasetConvert

def reindex_name2id(graphvite_embeddings, dataset):
    # -1 : pad, -2 : UNK
    idx2user, user2idx = { 0: -1, 1: -2 }, { -1: 0, -2: 1 }
    name2id = graphvite_embeddings['name2id']
    for name, id_ in name2id.items():
        node_type, node_id = name.split('_')
        node_type, node_id = int(node_type), int(node_id)

        if node_type == 0 and node_id not in user2idx:
            idx = len(idx2user)
            idx2user[idx] = node_id
            user2idx[node_id] = idx

    group2id = {-2: 0 }

    for data in dataset:
        for node in data.x[ data.x[:, 2] == 0]:
            if int(node[0]) not in user2idx:
                idx = len(idx2user)
                idx2user[idx] = int(node[0])
                user2idx[int(node[0])] = idx
        if hasattr(data, 'titleid'):
            group2id[int(data.titleid[0])] = len(group2id)
        else:
            group_id = data.x[ data.x[:, -1] == 2, :][0][2]
            group2id[int(group_id)] = len(group2id)

    return user2idx, idx2user, group2id



def evaluate_score(test_dataloader, model):
    B = 64
    precisions, recalls = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels, masked_target, group_id = d
            inputs, masked_target = inputs.cuda(), masked_target.cuda()
            B = inputs.shape[0]

            # classification only
            if mode == 'classifier':
                pred_index = model.predict(inputs, top_k=args.top_k)

                y_pred = torch.FloatTensor(B, user_size)
                y_pred.zero_()
                for batch_idx in range(B):
                    y_pred[batch_idx, pred_index[batch_idx]] = 1
            else:
                y_pred = model.predict_rank(inputs, masked_target, top_k=args.top_k)

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
    parser.add_('--epochs', type=int, default=5, 
                help='training epochs')
    parser.add_('--seq-len', type=int, default=5, 
                help='known users seq len')
    parser.add_('--neg-sample', type=int, default=5, 
                help='negative users')
    parser.add_('--rank-margin', type=float, default=1.0, 
                help='negative loss margin')
    parser.add_('--batch-size', type=int, default=64, 
                help='known users seq len')
    parser.add_('--lr', type=float, default=1e-4, 
                help='training lr')
    parser.add_('--embeddings', type=str,
                help='graphvite embedding pickle')
    parser.add_('--model', type=str, default='rank',
                help='model type', choices=['rank', 'attention'])
    parser.add_('--name', type=str, default='trial')
    parser.add_('--max-member', type=int, default=6, help='How many know member to select')
    parser.add_('--mode', type=str, default='ranking',
                help='learning method type', choices=['ranking', 'classifier'])
    args = parser.parse_args()
    if args.dataset == 'aminer':
        dataset = Aminer()
    else:
        dataset = Meetup(city_id=locations_id[args.city])

    with open(args.embeddings, 'rb') as f:
        graphvite_embeddings = pickle.load(f)

    data_size = len(dataset)
    if os.path.exists('.cache/{}_user2idx.pkl'.format(str(dataset))):
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'rb') as f:
            user2idx, idx2user, group2id = pickle.load(f)
    else:
        user2idx, idx2user, group2id = reindex_name2id(graphvite_embeddings, dataset)
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'wb') as f:
            pickle.dump((user2idx, idx2user, group2id), f)

    user_size = len(idx2user)
    # print(len(user2idx), len(idx2user))
    train_split, val, test = int(data_size*0.7), int(data_size*0.1), int(data_size*0.2)

    test_indexes = np.array(list(range(data_size)), dtype=np.long)[train_split+val:]
    test_dataset = dataset[list(test_indexes)]
    test_dataset = DatasetConvert(test_dataset, user_size, user2idx, group2id, max_seq=args.max_member)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, 
        num_workers=4, shuffle=False)

    val_indexes = np.array(list(range(data_size)), dtype=np.long)[train_split:train_split+val]
    val_dataset = dataset[list(val_indexes)]
    val_dataset = DatasetConvert(val_dataset, user_size, user2idx, group2id, max_seq=args.max_member)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, 
        num_workers=4, shuffle=False)

    indexes = np.array(list(range(data_size)), dtype=np.long)[:train_split]
    train_dataset = dataset[list(indexes)]
    dataset = DatasetConvert(train_dataset, user_size, user2idx, group2id, max_seq=args.max_member)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
        shuffle=True)

    # Setup from embedding files
    embeddings = nn.Embedding(len(user2idx), 64, padding_idx=0)
    name2id = graphvite_embeddings['name2id']
    vectors = graphvite_embeddings['embedding']
    embed_weight = embeddings.weight.data.numpy()
    for name, id_ in name2id.items():
        node_type, node_id = name.split('_')
        node_type, node_id = int(node_type), int(node_id)
        if node_type == 0:
            vector = vectors[id_]
            embed_weight[user2idx[node_id]] = vector

    embeddings.weight.data.copy_(torch.from_numpy(embed_weight))
    mode = args.mode

    if args.model == 'rank':
        model = DeepwalkClf(embeddings, user_size, mode=mode)
    else:
        model = DeepwalkAttnClf(embeddings, user_size, mode=mode)
        torch.nn.init.zeros_(model.query.weight)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    max_index_id = 0
    criterion = nn.BCEWithLogitsLoss()
    epochs = args.epochs

    '''
        Initialize logger : tensorboardX
    '''
    writer = None
    trial_name = '{}_{}_'.format(args.name, args.model)+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = osp.join(
        "logs", "deepwalk_rank",
        trial_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    save_path = osp.join(log_path, "models")
    with open(log_path+'/params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    os.makedirs(save_path, exist_ok=True)

    iter_ = 0
    model.train()
    with tqdm(total=epochs, dynamic_ncols=True) as pbar:
        for epoch in range(epochs):
            model.train()
            for d in dataloader:
                inputs, labels, masked_target, group_id = d
                # seq   B x user-size  B x user-size
                inputs, labels, masked_target = inputs.cuda(), labels.cuda(), masked_target.cuda()
                optimizer.zero_grad()

                if mode == 'classifier':
                    outputs = model(inputs)
                    outputs = outputs * masked_target
                    loss = criterion(outputs, labels)
                else:
                    loss = model.forward_rank(inputs, masked_target, labels, 
                        margin=args.rank_margin,
                        neg_sample=args.neg_sample)
                # print(masked_target.shape, outputs.shape)
                #   only allow candidate loss
                loss.backward()
                optimizer.step()
                if writer != None:
                    writer.add_scalar('train/loss', loss.item(), iter_)
                    iter_ += 1
                pbar.set_description("loss={:.4f}".format(loss.item()))
            pbar.update(1)
            scheduler.step()

            f1, avg_recalls, avg_precisions  = evaluate_score(val_dataloader, model)
            if writer != None:
                writer.add_scalar('val/f1', f1, epoch)
                writer.add_scalar('val/recalls', avg_recalls, epoch)
                writer.add_scalar('val/precision', avg_precisions, epoch)

    torch.save(model, os.path.join(save_path,'model.pt'))

    f1, avg_recalls, avg_precisions = evaluate_score(test_dataloader, model)
    print(f'[{trial_name}] top-{args.top_k}, F1: {f1} R: {avg_recalls} P: {avg_precisions}')
    print( f1, avg_recalls, avg_precisions)
    if writer != None:
        writer.add_scalar('test/f1', f1, 0)
        writer.add_scalar('test/recalls', avg_recalls, 0)
        writer.add_scalar('test/precision', avg_precisions, 0)
        writer.flush()
