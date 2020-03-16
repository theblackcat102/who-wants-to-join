import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.gcn import StackedGCNDBLP
from tqdm import tqdm
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from src.utils import dict2table, confusion, str2bool, calculate_f_score

PAD_ID = 874608
BOS_ID = PAD_ID+1
EOS_ID = PAD_ID+2

class LuongAttention(nn.Module):
    """Implementation of Luong Attention
    reference:
        Effective Approaches to Attention-based Neural Machine Translation
        Minh-Thang Luong, Hieu Pham, Christopher D. Manning
        https://arxiv.org/abs/1508.04025
    """
    def __init__(self, encoder_hidden_size, decoder_hidden_size, score='dot'):
        super(LuongAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.score = score
        if score == 'dot':
            assert(encoder_hidden_size == decoder_hidden_size)
        elif score == 'general':
            self.linear = nn.Linear(decoder_hidden_size, encoder_hidden_size)
        else:
            assert(False)

    def compute_energy(self, decoder_outputs, encoder_outputs):
        if self.score == 'dot':
            # [B x Tou x H_decoder] x [B x Tin x H_encoder] -> [B x Tou x Tin]
            attn_weight = torch.bmm(
                decoder_outputs, encoder_outputs.transpose(1, 2))
        if self.score == 'general':
            # [B x Tou x H_encoder]
            decoder_outputs = self.linear(decoder_outputs)
            # [B x Tou x H_decoder] x [B x Tin x H_encoder] -> [B x Tou x Tin]
            attn_weight = torch.bmm(
                decoder_outputs, encoder_outputs.transpose(1, 2))
        return attn_weight

    def forward(self, decoder_outputs, encoder_outputs):
        """Support batch operation.
        Output size of encoder and decoder must be equal.
        Args:
            decoder_outputs: float tensor, shape = [B x Tou x H_decoder]
            encoder_outputs: float tensor, shape = [B x Tin x H_encoder]
        Returns:
            output: float tensor, shape = [B x Tou x (2 x H_decoder)]
            attn_weight: float tensor, shape = [B x Tou x Tin]
        """
        decoder_outputs = decoder_outputs.transpose(1, 0)
        encoder_outputs = encoder_outputs.transpose(1, 0)

        attn_weight = self.compute_energy(decoder_outputs, encoder_outputs)
        attn_weight = self.softmax(attn_weight)
        # [B x Tou x Tin] * [B x Tin x H] -> [B, Tou, H]
        attn_encoder_outputs = torch.bmm(attn_weight, encoder_outputs)
        # concat [B x Tou x H], [B x Tou x H] -> [B x Tou x (2 x H)]
        output = torch.cat([decoder_outputs, attn_encoder_outputs], dim=-1)
        output = output.transpose(1, 0)
        return output, attn_weight


class SingleHeadAttention(nn.Module):
    ''' Single-Head Attention module '''
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        n_head = 1
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        o, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        o = o.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        o = self.dropout(self.fc(o))
        o += residual

        return o, attn, q, k, v

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class SAN(nn.Module):
    '''
    Batch first self attention
    '''
    def __init__(self, input_dim, head_dim=16, dropout=0.1,
            output_query=False):
        super().__init__()
        self.self_attn = SingleHeadAttention(input_dim, head_dim, head_dim,
            dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        # self.linear1 = nn.Linear(input_dim, ff_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(ff_dim, input_dim)
        # self.activation = nn.ReLU()

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.output_query = output_query

    def forward(self, src, src_mask=None):
        if self.output_query:
            src2, attn, q, k, v = self.self_attn(src, src, src, mask=src_mask)
            return src2, attn, q, k, v
        else:
            src2 = self.self_attn(src, src, src, mask=src_mask)[0]
            src = self.norm1(src2)
            return src

def sequence_pad(embeddings, data, pad_vector=None):
    batch = data.batch
    batch_size = data.size
    input_mask = data.input_mask

    padding_mask = torch.ones(len(batch_size), batch_size.max(), embeddings.shape[-1])

    features = torch.zeros(len(batch_size), batch_size.max(), embeddings.shape[-1])
    features = features.to(embeddings.device)
    max_seq_len = batch_size.max()
    if pad_vector is not None:
        pad_vector = pad_vector.to(embeddings.device)
    trunc_seq_len = 0

    for idx in range(len(batch_size)):
        seq_len_mask = (batch == idx) & (input_mask == 1)
        seq_len = seq_len_mask.sum()
        if trunc_seq_len < seq_len:
            trunc_seq_len = seq_len
        features[idx,  :seq_len] = embeddings[ seq_len_mask ]
        if pad_vector is not None and (max_seq_len-seq_len) > 0 :
            padded_ = pad_vector.repeat(1, max_seq_len-seq_len, 1)
            features[idx, seq_len:] = pad_vector.repeat(1, max_seq_len-seq_len, 1)
            padding_mask[idx, seq_len:] = torch.zeros(padded_.shape)

    # print(max_seq_len)
    return features[:, :trunc_seq_len+1, :].contiguous(), padding_mask

def obtain_loss_mask(x, label_mask, embedding_size, batch):
    batch_size = batch.max()+1
    mask = torch.zeros(batch_size, embedding_size+3)
    label_bool = label_mask == 1

    for idx in range(batch_size):
        label_mask_id = x[ (batch == idx) & label_bool ]
        mask[ idx, label_mask_id ] = 1
        mask[idx, -1] = 1
        mask[idx,  [embedding_size, embedding_size+1, embedding_size+2] ] = 1
    return mask

def output2seq(data, pad_id, max_len=-1,):
    batch_size = len(data.size)
    batch = data.batch
    sequences = torch.zeros(batch_size, data.size.max())
    for idx in range(batch_size):
        y_id = data.x[ (data.y == 1) & (batch == idx), 0 ]
        sequences[idx, :len(y_id)] = y_id
        sequences[idx, len(y_id): ] = pad_id
    if max_len > 0:
        return sequences[:, :max_len].long()
    return sequences.long()


def label_sequence(embeddings, data, pad_vector, max_len=-1, negative_sample=50): 
    # sample positive and negative latent feature
    batch_size = len(data.size)
    batch = data.batch
    positive_latents = []
    negative_latents = []
    sequence_len = []
    x = data.x


    for idx in range(batch_size):
        paper_idx = (data.batch == idx).cuda()
        # select not known author nodes of the current batch
        author_node_idx = ((x[:, -1] == 0).cuda() & paper_idx & (x[:, 1 ] == 0).cuda()).cuda()

        pos_embeddings_ = embeddings[ (data.y==1).cuda() & author_node_idx ]
        neg_embedding_ = embeddings[  (data.y==0 ).cuda() & author_node_idx ]
        pos_latent_t, neg_latent_t, seq_ = [], [], []

        for t in range(max_len):
            if t < len(pos_embeddings_):
                pos_latent_t.append(pos_embeddings_[t].squeeze(0))
                seq_.append(1)
            else:
                pos_latent_t.append(pad_vector.squeeze(0))
                seq_.append(0)

            if t > 0 and t <= len(pos_embeddings_):
                neg_embedding_[t] = pos_latent_t[t-1]

            shuffle_idx = list(range(len(neg_embedding_)))
            random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[:negative_sample]

            # neg_embedding_[ shuffle_idx ]
            neg_latent_t.append( neg_embedding_[shuffle_idx, :].squeeze(0))
        sequence_len.append(seq_)
        positive_latents.append(torch.stack(pos_latent_t).squeeze(0))
        negative_latents.append(torch.stack(neg_latent_t).squeeze(0))

    sequence_len = torch.from_numpy(np.asarray(sequence_len)).float().cuda()  
    negative_latents = torch.stack(negative_latents)
    positive_latents = torch.stack(positive_latents)
    # print(positive_latents.shape, negative_latents.shape)
    return positive_latents, negative_latents, sequence_len

class HINT(nn.Module):
    def __init__(self, san_dim=16,
        author_size=874608, paper_size=3605603, conf_size=12770,
        user_dim=8, paper_dim=8,
        conf_dim=4,
        input_channels=16,
        layers=[32, 32],
        dropout=0.1,
        san_head_dim=32,
        use_attn=False
        ):
        super().__init__()
        self.author_size = ( author_size+3, user_dim)
        self.paper_size =  (paper_size, paper_dim)
        self.conf_size = ( conf_size, conf_dim)
        self.gcn = StackedGCNDBLP(
                    author_size=author_size+3,#len(author2id),
                    paper_size=paper_size,#len(paper2id),
                    conf_size=conf_size,
                    output_channels=san_dim,
                    user_dim=user_dim, paper_dim=paper_dim, conf_dim=conf_dim, input_channels=input_channels,
                    layers=layers, dropout=dropout)

        self.special_token = nn.Embedding(3, san_dim)
        self.d_model = san_dim

        self.encoder = SAN(san_dim)
        self.encoder2 = SAN(san_dim)

        self.node_class = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(san_dim, 3)
        )
        self.seq_pred = nn.Sequential(
            nn.LayerNorm(san_dim),
            nn.ReLU(),
            nn.Linear(san_dim, 1)
        )

    def forward(self, data, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        x, edge_index = data.x, data.edge_index
        _, _, embeddings = self.gcn(edge_index.cuda(), x.cuda())

        feature = embeddings  # [ data.y == 1 ]
        feature = feature.reshape(len(data.titleid), data.length, -1 )
        # the last token should replace as EOS, but not sure how to execute this elegantly
        feature = feature[ :, :data.known.max()+1, : ]

        if isinstance(self.encoder, SAN ): # use SAN
            output = self.encoder(feature, src_mask=src_mask)
            output = self.encoder2(output, src_mask=src_mask) + output
        else:
            output = self.encoder(feature, mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask)

        return output[:, [0], :], self.node_class(embeddings), embeddings

    def train_batch(self, data, src_mask=None, margin=5, batch_size=32):
        x, edge_index = data.x, data.edge_index
        _, _, embeddings = self.gcn(edge_index.cuda(), x.cuda())

        # Note I suspect this will cause info leak
        # feature = embeddings  # [ data.y == 1 ]
        feature, pad_mask = sequence_pad(embeddings, data, self.gcn.embeddings.weight[PAD_ID])
        feature = feature[ :, :data.known.max()+1, :]
        pad_mask = pad_mask[:, :data.known.max()+1, :].cuda()

        pos_embeds, neg_embeds, seq_len_encode = label_sequence(embeddings,
                                                                data,
                                                                self.gcn.embeddings.weight[PAD_ID],
                                                                max_len=data.known.max()+1,
                                                                negative_sample=10)

        log_sigmoid = nn.LogSigmoid()
        total_pos, total_neg = 0, 0

        max_time = feature.shape[1]
        pred_results = []
        for idx in range(max_time):

            output = self.encoder(feature, src_mask=None)
            output = self.encoder2(output, src_mask=None) + output
            # latent = output[ :, [0], :]
            output = output * pad_mask
            latent = output.sum(1).unsqueeze(1)

            pred = self.seq_pred(latent)
            pred_results.append(pred)

            target_embed = pos_embeds[ :, idx, : ].unsqueeze(-1)

            pos = log_sigmoid(torch.bmm(latent, target_embed))
            total_pos += pos.sum()

            target_embed = neg_embeds[:, idx, :]
            target_embed = target_embed.permute(0, 2, 1)

            neg = (margin - torch.bmm(latent, target_embed).flatten()) .clamp(min=0)
            # print(neg[:10])
            neg = log_sigmoid(neg)
            total_neg += neg.sum()

            feature = torch.cat([ pos_embeds[:, [idx], :], feature ], dim=1).contiguous()
            pad_mask = torch.cat([ pad_mask[ :, [0], : ], pad_mask ], dim=1).contiguous()

        loss = (total_pos+total_neg) / (batch_size*2 )
        pred_results = torch.cat(pred_results, dim=1)
        return -loss, self.node_class(embeddings), (pred_results, seq_len_encode) 

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def inference(self, data, top_k=1, user_size=874608):
        # greedy decoding
        B = torch.max(data.batch)+1
        x, edge_index = data.x, data.edge_index

        _, _, embeddings = self.gcn(edge_index.cuda(), x.cuda())
        feature, pad_mask = sequence_pad(embeddings, data, self.gcn.embeddings.weight[PAD_ID])
        feature = feature[ :, :data.known.max()+1, :]


        y_pred = torch.FloatTensor(B, user_size)
        y_pred.zero_()
        max_time = feature.shape[1]

        # pred_seq = []
        for t in range(max_time):
            output = self.encoder(feature, src_mask=None)
            output = self.encoder2(output, src_mask=None) + output
            latent = output[ :, [0], :]



            author_node_idx = ((x[:, -1] == 0).cuda() & paper_idx & (x[:, 1 ] == 0).cuda()).cuda()
            candidate_embeddings = embeddings[ data.x[:, 1, :] == 0 ]


        return y_pred


def evaluate( dataloader, model, batch_size=8, top_k=5, user_size=874608):
    model.eval()
    recalls = []
    precisions = []
    R, P = [], []
    B = batch_size
    val_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader, dynamic_ncols=True):
            label_mask_id = obtain_loss_mask(data.x[ :, 0 ],
                data.label_mask,
                874608,
                data.batch )
            x, edge_index = data.x, data.edge_index
            x = x.cuda()
            rank_loss, label_pred, (seq_pred, seq_label) = model.train_batch(data)
            print(seq_pred[0].flatten(), seq_label[0].flatten())
            break

def calculate_loss( gcn_outputs, author_embed, batch, batch_size, margin=5):
    # calculate triple ranking loss
    total_neg = 0
    total_pos = 0
    log_sigmoid = nn.LogSigmoid()
    # iterate through all datasets
    batch_size = torch.max(batch.batch)+1
    for batch_idx in range(batch_size):
        paper_idx = (batch.batch == batch_idx)
        # author node in this data

        # candidate_embed = author_embed[batch_idx]
        # latent = candidate_embed.sum(0)
        latent = author_embed[batch_idx, :, :]
        # print(latent.shape)
        target_embed = gcn_outputs[ (batch.y==1) & paper_idx]
        shuffle_idx = list(range(len(target_embed)))
        random.shuffle(shuffle_idx)
        target_embed = target_embed[shuffle_idx]

        pos = log_sigmoid(torch.mm(latent, target_embed.T).flatten())

        # pos = torch.mm(latent.unsqueeze(-1).T, target_embed.T).flatten()
        total_pos += pos.sum()
        # not label not known users and node type = user
        negative_node_idx = (batch.y == 0) & (batch.x[:, 1 ] == 0) & (batch.x[:, 2] == 0) & paper_idx
        negative_embed = gcn_outputs[negative_node_idx]
        if len(negative_embed) == 0:
            # print('no negative found!')
            continue

        shuffle_idx = list(range(len(negative_embed)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:len(target_embed)]

        negative_embed = negative_embed[shuffle_idx]

        # torch.dot(target_embed[0],  latent)
        loss = (margin-torch.mm(latent, negative_embed.T).flatten()).clamp(min=0)
        neg = log_sigmoid(loss)
        total_neg += neg.sum()

    loss = (total_pos+total_neg) / (batch_size*2)
    # loss = total_neg/(N*2)#log_sigmoid(total_pos + total_neg)/ (batch_size*2)
    return -loss

if __name__ == "__main__":
    from torch_geometric.data import DataLoader
    from src.aminer import Aminer, PaddedDataLoader
    from src.gcn import StackedGCNDBLP
    import os.path as osp
    import pickle

    dataset = Aminer()
    val_data = Aminer()
    model = HINT(
        paper_dim=8,
        user_dim=16, 
        layers=[16, 16],
    ).cuda()
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

    split_pos = int(len(dataset)*0.7)
    train_idx = shuffle_idx[:split_pos]
    valid_idx_ = shuffle_idx[split_pos:]
    batch_size = 32
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = PaddedDataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = PaddedDataLoader(dataset[valid_idx_], batch_size=batch_size, shuffle=False, num_workers=4)

    binary_criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(50):
        loss_mini_batch = 0
        for i, data in enumerate(loader):
            label_mask_id = obtain_loss_mask(data.x[ :, 0 ],
                data.label_mask,
                874608,
                data.batch )
            x, edge_index = data.x, data.edge_index
            x = x.cuda()
            rank_loss, label_pred, (seq_pred, seq_label) = model.train_batch(data)

            # output = output.transpose(1, 0) # batch_size x 1 x 16
            # label_mask_id = label_mask_id.unsqueeze(1)
            # label_mask_id = label_mask_id.repeat(1, output.shape[1], 1).cuda()
            # target = output2seq(data, PAD_ID, max_len=output.shape[1]).cuda()

            # print(target.shape)
            # pred = output
            node_loss = criterion(label_pred, x[:, -1])
            seq_loss = binary_criterion(seq_pred.flatten(), seq_label.flatten())
            loss =  rank_loss + node_loss + seq_loss
            # loss = node_loss
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            if i % 70 == 0:
                print(epoch, i, loss.item(), rank_loss.item(), node_loss.item(), seq_loss.item())
                # print(seq_pred[0].flatten(), seq_label[0].flatten())

            if i % 200 == 0 and i > 0:
                model.eval()
                with torch.no_grad():
                    evaluate(val_loader, model, user_size=PAD_ID+1, batch_size=batch_size)
                model.train()
        # break
