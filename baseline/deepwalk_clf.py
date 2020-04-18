import torch
from torch import nn
import pickle
import numpy as np
from dataset.aminer import Aminer
import numpy as np
import pickle, os, glob
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.utils import dict2table, confusion, str2bool
import random

class DeepwalkClf(torch.nn.Module):

    def __init__(self, embeddings, user_size, mode='classifier'):
        super(DeepwalkClf, self).__init__()
        self.user_size = user_size
        embed_size, embed_dim = embeddings.weight.shape
        self.embeddings = embeddings
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, embed_dim)
        )

        if mode == 'classifier':
            self.output = nn.Linear(embed_dim, user_size)
            print('initialize output proj using embeddings')
            self.output.weight.data.copy_(self.embeddings.weight.data)
        self.log_sigmoid = nn.LogSigmoid()
        self.embed_dim = embed_dim

    def forward_rank(self, candidates, masked_target, target, margin=1, neg_samples=2):
        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)

        latent = self.fc(pooled)

        batch_size = candidates.shape[0]
        total_neg = 0
        total_pos = 0

        for batch_idx in range(batch_size):
            user_latent = latent[batch_idx]
            masked_target[batch_idx] = masked_target[batch_idx] - target[batch_idx]
            negative_index = (masked_target[batch_idx] == 1 ).nonzero()
            neg_latent = self.embeddings(negative_index.squeeze(-1))

            shuffle_idx = list(range(len(neg_latent)))
            random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[:neg_samples]

            neg_latent = neg_latent[shuffle_idx]

            loss = (margin-torch.mm(user_latent.unsqueeze(-1).T, neg_latent.T).flatten()).clamp(min=0)
            neg = self.log_sigmoid(loss)
            total_neg += neg.sum()

            pos_latent = self.embeddings((target[batch_idx] == 1).nonzero().squeeze(-1))

            # print(user_latent.unsqueeze(-1).T.shape, pos_latent.T.shape)
            pos = self.log_sigmoid(torch.mm(user_latent.unsqueeze(-1).T, pos_latent.T).flatten())
            total_pos += pos.sum()

        loss = (total_pos+total_neg) / (batch_size*2)
        # loss = total_neg/(N*2)#log_sigmoid(total_pos + total_neg)/ (batch_size*2)
        return -loss
    
    def forward(self, candidates):
        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)

        latent = self.fc(pooled)
        return self.output(latent)

    def predict(self, candidates, top_k=5):
        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)
        predict = self.fc(pooled)
        values, index = torch.topk(predict, k=top_k)
        return index

    def predict_rank(self, candidates, masked_target, top_k=5):
        batch_size = candidates.shape[0]

        x = self.embeddings(candidates)
        x = x.permute(0, 2, 1)
        pooled = self.pool(x)
        pooled = pooled.view(-1, self.embed_dim)
        latent = self.fc(pooled)

        y_pred = torch.FloatTensor(batch_size, self.user_size)
        y_pred.zero_()

        for batch_idx in range(batch_size):
            user_latent = latent[batch_idx]
            candidate_index = (masked_target[batch_idx] == 1 ).nonzero()
            pos_latent = self.embeddings(candidate_index.squeeze(-1))
            
            rank = torch.sigmoid(torch.mm(user_latent.unsqueeze(-1).T, pos_latent.T)).flatten()
            best_idx = torch.argsort(rank, descending=True)
            y_pred[ batch_idx, candidate_index[best_idx[:top_k]] ] = 1

        return y_pred

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

    for data in dataset:
        for node in data.x[ data.x[:, 2] == 0]:
            if int(node[0]) not in user2idx:
                idx = len(idx2user)
                idx2user[idx] = int(node[0])
                user2idx[int(node[0])] = idx

    return user2idx, idx2user

class DatasetConvert(torch.utils.data.Dataset):
    def __init__(self, dataset, user_size, user2idx, max_seq = 10):
        self.dataset = dataset
        self.user_size = user_size
        self.user2idx = user2idx
        self.max_seq = max_seq
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, user_node_id=0):
        data = self.dataset[idx]
        known_user_node_id = (data.x[:, 2] == user_node_id) & (data.x[:, 1] == 1)
        known_nodes = []
        for node in data.x[known_user_node_id, :]:
            idx = self.user2idx[-2]
            if int(node[0]) in self.user2idx:
                idx = self.user2idx[int(node[0])]
            known_nodes.append(idx)

        known_nodes = list(set(known_nodes))
        while len(known_nodes) < self.max_seq:
            known_nodes.append(self.user2idx[-1])

        known_nodes = np.array(known_nodes)[:self.max_seq]

        target_node_id = []
        for node in data.x[data.y == 1, :]:
            if int(node[0]) in self.user2idx:
                target_node_id.append(self.user2idx[int(node[0])])
            else:
                print(int(node[0]))

        y_target = np.zeros(self.user_size)
        y_target[target_node_id] = 1.0

        mask_node_id = []
        # not known candidate group
        candidate_user_node_id = (data.x[:, 2] == user_node_id) & (data.x[:, 1] != 1)
        for node in data.x[candidate_user_node_id, :]:
            if int(node[0]) in self.user2idx:
                mask_node_id.append(self.user2idx[int(node[0])])

        # candidate_user_node_id = [ self.user2idx[int(node[0])] for node in data.x[candidate_user_node_id, :]]
        masked_target = np.zeros(self.user_size)
        # print(mask_node_id)
        # print(target_node_id)
        masked_target[mask_node_id] = 1.0

        return torch.from_numpy(known_nodes).long(), \
                torch.from_numpy(y_target), torch.from_numpy(masked_target).long()

if __name__ == "__main__":
    user_size = 399211

    with open('graphvite_embeddings/aminer_deduplicate_train_edgelist.txt-64-DeepWalk.pkl', 'rb') as f:
        graphvite_embeddings = pickle.load(f)
    dataset = Aminer()
    data_size = len(dataset)
    if os.path.exists('.cache/{}_user2idx.pkl'.format(str(dataset))):
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'rb') as f:
            user2idx, idx2user, _ = pickle.load(f)
    else:
        user2idx, idx2user = reindex_name2id(graphvite_embeddings, dataset)
        with open('.cache/{}_user2idx.pkl'.format(str(dataset)), 'wb') as f:
            pickle.dump((user2idx, idx2user), f)

    user_size = len(idx2user)
    # print(len(user2idx), len(idx2user))
    train_split, val, test = int(data_size*0.7), int(data_size*0.1), int(data_size*0.2)

    indexes = np.array(list(range(data_size)), dtype=np.long)[train_split+val:]
    test_dataset = dataset[list(indexes)]
    test_dataset = DatasetConvert(test_dataset, user_size, user2idx, max_seq=6)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=4)


    indexes = np.array(list(range(data_size)), dtype=np.long)[:train_split]
    train_dataset = dataset[list(indexes)]
    dataset = DatasetConvert(train_dataset, user_size, user2idx, max_seq=6)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)

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
    # print(embeddings.weight.shape)
    mode = 'ranking'

    model = DeepwalkClf(embeddings, user_size, mode=mode)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)    
    max_index_id = 0
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    with tqdm(total=epochs, dynamic_ncols=True) as pbar:
        for epoch in range(epochs):
            for d in dataloader:
                inputs, labels, masked_target = d
                inputs, labels, masked_target = inputs.cuda(), labels.cuda(), masked_target.cuda()
                # print(inputs.shape)
                optimizer.zero_grad()

                if mode == 'classifier':
                    outputs = model(inputs)
                    outputs = outputs * masked_target
                    loss = criterion(outputs, labels)
                else:
                    loss = model.forward_rank(inputs, masked_target, labels)
                # print(masked_target.shape, outputs.shape)
                #   only allow candidate loss
                loss.backward()
                optimizer.step()
                pbar.set_description("loss={:.4f}".format(loss.item()))

            pbar.update(1)

    B = 64
    precisions, recalls = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels, masked_target = d
            inputs, labels, masked_target = inputs.cuda(), labels.cuda(), masked_target.cuda()
            B = inputs.shape[0]

            y_target = labels.cpu()

            # classification only
            if mode == 'classifier':
                pred_index = model.predict(inputs, top_k=5)

                y_pred = torch.FloatTensor(B, user_size)
                y_pred.zero_()
                for batch_idx in range(B):
                    y_pred[batch_idx, pred_index[batch_idx]] = 1
            else:
                y_pred = model.predict_rank(inputs, masked_target, top_k=5)

            TP, FP, TN, FN = confusion(y_pred, y_target)
            recall = 0 if (TP+FN) < 1e-5 else TP/(TP+FN)
            precision = 0 if (TP+FP) < 1e-5 else TP/(TP+FP)
            # print(precision, recall)
            precisions.append(precision)
            recalls.append(recall)

    avg_recalls = np.mean(recalls)
    avg_precisions = np.mean(precisions)
    f1 = 2*(avg_recalls*avg_precisions)/(avg_recalls+avg_precisions)

    print( f1, avg_recalls, avg_precisions)