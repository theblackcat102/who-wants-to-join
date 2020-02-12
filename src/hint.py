import torch
import torch.nn as nn

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.gcn import StackedGCNDBLP

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
PAD_ID = 874608
BOS_ID = PAD_ID+1
EOS_ID = PAD_ID+2

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

    def __init__(self, input_dim, head_dim=16 ,dropout=0.1, output_query=False):
        super().__init__()
        self.self_attn = SingleHeadAttention(input_dim, head_dim, head_dim, dropout=dropout)
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
        features[idx,  :seq_len ] = embeddings[ seq_len_mask]
        if pad_vector is not None and (max_seq_len-seq_len) > 0 :
            features[idx, seq_len: ] = pad_vector.repeat(1, max_seq_len-seq_len, 1 )

    # print(max_seq_len)
    return features[:, :trunc_seq_len+1, :].contiguous()

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

def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros= ( masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums



class HINT(nn.Module):
    def __init__(self, san_dim=16, 
        author_size=874608, paper_size=3605603, conf_size=12770,
        user_dim=8,
        paper_dim=8,
        conf_dim=4,
        input_channels=16,
        layers=[32, 32],
        dropout=0.1,
        san_head_dim=32, 
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

        encoder_layer = TransformerEncoderLayer(san_dim, 1, san_head_dim, 0.1)
        encoder_norm = nn.LayerNorm(san_dim)
        self.encoder = TransformerEncoder(encoder_layer, 2, encoder_norm)

        decoder_layer = TransformerDecoderLayer(san_dim, 1, san_head_dim, 0.1)
        decoder_norm = nn.LayerNorm(san_dim)
        self.decoder = TransformerDecoder(decoder_layer, 2, decoder_norm)
        self.decoder_embeddings = nn.Sequential(self.gcn.embeddings,
            nn.Linear(user_dim, san_dim))

        # self.stacked_encoder = nn.Sequential(
        #     # SAN(san_dim, dropout=dropout),
        #     SAN(san_dim, dropout=dropout, output_query=True),
        # )

        # self.stacked_decoder = nn.Sequential(
        #     # SAN(san_dim, dropout=dropout),
        #     SAN(san_dim, dropout=dropout, output_query=True),
        # )
        self.linear = nn.Sequential(
            nn.LayerNorm(san_dim),
            nn.LeakyReLU(),
            nn.Linear(san_dim, author_size+3) # include pad
        )
        self.node_class = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(san_dim, 3)
        )
        self.attn = ScaledDotProductAttention(1)

    def forward(self, data, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        x, edge_index = data.x, data.edge_index
        _, _, embeddings = self.gcn(edge_index.cuda(), x.cuda())
        

        feature = embeddings#[ data.y == 1 ]
        feature = feature.reshape(len(data.titleid), data.length, -1 )
        # the last token should replace as EOS, but not sure how to execute this elegantly
        feature = feature[ :, :data.known.max()+1, : ] 
        
        target = output2seq(data, PAD_ID, max_len=feature.shape[1]-1).cuda()
        bos_idx = torch.zeros( target.size(0), 1 ) + PAD_ID+1
        bos_idx = bos_idx.long().cuda()
        target = torch.cat([bos_idx, target], dim=1)
        target = self.decoder_embeddings(target)

        # src2, attn, q, k, v = self.stacked_encoder(feature)
        # src, _, q_, k_, v_ = self.stacked_decoder(src2)
        # sz_b, len_q, len_k, len_v = q.size(0), q.size(2), k.size(1), v.size(1)
        # out, attn = self.attn(q_, k, v)
        # out = out.contiguous().view(sz_b, len_q, -1)

        feature = feature.transpose(1, 0)
        target = target.transpose(1, 0)

        if feature.size(1) != target.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if feature.size(2) != self.d_model or target.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(feature, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt_mask = self.generate_square_subsequent_mask(target.size(0)).cuda()
        output = self.decoder(target, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        return self.linear(output), self.node_class(embeddings)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def inference(self, data):
        # greedy decoding
        x, edge_index = data.x, data.edge_index
        _, _, embeddings = self.gcn(edge_index.cuda(), x.cuda())
        feature = embeddings#[ data.y == 1 ]
        feature = feature.reshape(len(data.titleid), data.length, -1 )
        max_length = data.known.max()+1
        feature = feature[ :, :data.known.max()+1, : ]
        feature = feature.transpose(1, 0)
        memory = self.encoder(feature)
        tokens = []
        bos_idx = torch.zeros( 1, memory.size(1) ) + PAD_ID + 1
        targets = bos_idx.long().cuda()
        pred_seq = []

        label_mask_id = obtain_loss_mask(data.x[ :, 0 ], 
                        data.label_mask, 
                        PAD_ID, 
                        data.batch )
        label_mask_id = label_mask_id.unsqueeze(0).cuda()

        for idx in range(max_length):
            target = self.decoder_embeddings(targets)
            output = self.decoder(target, memory)
            memory = torch.cat([memory, output], dim=0)

            output = self.linear(output)
            pred = masked_softmax(output, label_mask_id)
            output_idx = pred.argmax(2)

            targets = output_idx
            pred_seq.append(output_idx)

        pred_seq = torch.cat(pred_seq, dim=0)
        print(pred_seq.shape)
        print(pred_seq[0])
        return pred_seq.transpose(1, 0)

if __name__ == "__main__":
    from torch_geometric.data import DataLoader
    from src.aminer import Aminer,PaddedDataLoader
    from src.gcn import StackedGCNDBLP
    dataset = Aminer()
    model = HINT(
        paper_dim=6,
        layers=[16, 16],
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = PaddedDataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
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
            output, label_pred = model(data)

            output = output.transpose(1, 0)

            label_mask_id = label_mask_id.unsqueeze(1)
            label_mask_id = label_mask_id.repeat(1, output.shape[1], 1).cuda()
            target = output2seq(data, PAD_ID, max_len=output.shape[1]).cuda()

            # pred = masked_softmax(output, label_mask_id)
            pred = output
            node_loss = criterion(label_pred, x[:, -1])

            pred_loss = criterion(pred.reshape(-1, PAD_ID+3), target.flatten()) / pred.shape[1]
            loss =  5*pred_loss + node_loss
            # loss = node_loss
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            print(loss.item(), pred_loss.item(), node_loss.item())

