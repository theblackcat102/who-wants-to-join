import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import TOKENS
import random
from .utils import gumbel_softmax, softmax, binary_matrix, orthogonal_initialization

MAX_LENGTH = 203

class FactorizedEmbeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, vocab_size, hidden, bottle_neck=16):
        super(FactorizedEmbeddings, self).__init__()
        self.tok_embed1 = nn.Embedding(vocab_size, bottle_neck)
        self.tok_embed2 = nn.Linear(bottle_neck, hidden)

        self.norm = LayerNorm(hidden)

    def forward(self, x):
        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        return self.norm(e)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation used to generate
            the noise. Relative means that it will be multiplied by the
            magnitude of the value your are adding the noise to. This means
            that sigma can be the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable
            before computing the scale of the noise. If `False` then the scale
            of the noise won't be seen as a constant but something to optimize:
            this will bias the network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            if torch.cuda.is_available():
                self.noise = self.noise.cuda()
            if self.is_relative_detach:
                scale = self.sigma * x.detach()
            else:
                scale = self.sigma * x
            sampled_noise = \
                self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
            del sampled_noise
        return x


class Encoder(nn.Module):
    """A general Encoder, keep it as general as possible."""
    def __init__(self, inputs_size, hidden_size, num_layers, dropout,
                 bidirectional, cell):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # self.noise_layer = GaussianNoise(sigma=0.1)
        self.rnn = cell(
            inputs_size, hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=bidirectional,
            batch_first=True)

    def forward(self, inputs, hidden=None):
        """
        Args:
            inputs: int tensor, shape = [B x T x inputs_size]
            hidden: float tensor,
                shape = shape = [num_layers * num_directions x B x hidden_size]
            is_discrete: boolean, if False, inputs shape is
                [B x T x vocab_size]
        Returns:
            outputs: float tensor, shape = [B x T x (hidden_size x dir_num)]
            hidden: float tensorf, shape = [B x (hidden_size x dir_num)]
        """

        outputs, hidden = self.rnn(inputs, hidden)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.bidirectional or self.num_layers > 0:
            outputs = outputs.view(-1, outputs.size(1), 2, self.hidden_size)
            hidden = torch.mean(hidden, axis=0).unsqueeze(0)
            outputs = outputs[:, :, 0] + outputs[:, :, 1]
        return outputs, hidden


class Decoder(nn.Module):
    """A general Decoder, keep it as general as possible."""
    def __init__(self, inputs_size, vocab_size, hidden_size,
                 num_layers, dropout, st_mode, cell, attention=None):
        super(Decoder, self).__init__()

        self.attention = attention
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.st_mode = st_mode

        self.rnn = cell(
            inputs_size, hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            batch_first=True)
        if attention is not None:
            self.outputs2vocab = nn.Linear(hidden_size * 2, vocab_size)
        else:
            self.outputs2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden, encoder_outputs=None, temperature=1):
        """
        Args:
            inputs: float tensor, shape = [B x T x inputs_size]
            hidden: float tensor, shape = [num_layers x B x H]
            encoder_outputs: float tensor, shape = [B x Tin x H]
            temperature: float, gumbel softmax
        Returns:
            outputs: float tesor, shape = [B x T x vocab_size], probability
            hidden: float tensor, shape = [num_layers, B x H]
        """
        # print('decoder input')
        # print(inputs[0, -3:, :5])
        if self.num_layers > 0:
            hidden = hidden.repeat(self.num_layers, 1, 1)
        if isinstance(self.rnn, nn.LSTM):
            hidden = (hidden, hidden)
        outputs, hidden = self.rnn(inputs, hidden)
        if self.attention is not None:
            outputs, attn_weight = self.attention(outputs, encoder_outputs)
        outputs = self.outputs2vocab(outputs)
        if self.training:
            outputs = softmax(outputs, temperature, st_mode=self.st_mode)
        else:
            outputs = softmax(outputs, temperature=1, st_mode=False)
        return outputs, hidden


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
        attn_weight = self.compute_energy(decoder_outputs, encoder_outputs)
        attn_weight = self.softmax(attn_weight)
        # [B x Tou x Tin] * [B x Tin x H] -> [B, Tou, H]
        attn_encoder_outputs = torch.bmm(attn_weight, encoder_outputs)
        # concat [B x Tou x H], [B x Tou x H] -> [B x Tou x (2 x H)]
        output = torch.cat([decoder_outputs, attn_encoder_outputs], dim=-1)

        return output, attn_weight



class Seq2Seq(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size,
                 enc_num_layers, dec_num_layers, dropout, st_mode, 
                 enc_cell=nn.LSTM, dec_cell=nn.LSTM, 
                 enc_bidirect=True, use_attn=True,tag_size=965
                 ):
        super(Seq2Seq, self).__init__()
        # self.embedding = FactorizedEmbeddings(vocab_size, embed_size, embed_size//2)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(embed_size, hidden_size, enc_num_layers, dropout, bidirectional=enc_bidirect, cell=nn.LSTM)
        self.embed_dropout = nn.Dropout(0.1)
        attn = None
        if use_attn:
            print('Use Attention')
            attn = LuongAttention(encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size)
        self.decoder = Decoder(embed_size, vocab_size, hidden_size, dec_num_layers, dropout=dropout, 
            st_mode=st_mode, cell=nn.LSTM, attention=attn)
        orthogonal_initialization(self)

    def forward(self, inputs, labels, temperature=1):
        embed = self.embedding(inputs)
        embed = self.embed_dropout(embed)
        output_embed = self.embedding(labels)
        latent, hidden = self.encoder(embed)
        outputs, d_h = self.decoder(output_embed[:, :-1, :], hidden, 
            encoder_outputs=latent, temperature=temperature)
        return outputs, d_h, hidden

    def decode(self, inputs, tags=None, temperature=1, target_length=-1, topk=1):
        embed = self.embedding(inputs)
        embed = self.embed_dropout(embed)
        latent, hidden = self.encoder(embed)
        if target_length < 0:
            raise ValueError('decode length must be longer than 0')

        decoder_tokens, decoder_outputs = [], []
        decoder_input = self.embedding(torch.tensor([[TOKENS['BOS']]], device=inputs.device))
        for di in range(target_length):
            outputs, hidden = self.decoder(decoder_input, hidden, 
                encoder_outputs=latent, temperature=temperature)
            _, topi = outputs.topk(1)
            hidden = hidden[0]
            decoder_outputs.append(outputs.squeeze(1))
            decoder_input = self.embedding(topi.detach()).squeeze(0)
            _, topi_ = outputs.topk(topk)

            decoder_tokens.append(topi_.cpu().detach().squeeze().numpy().tolist())
            if topi.item() == TOKENS['EOS']:
                break
        return np.array(decoder_tokens), torch.stack(decoder_outputs, dim=1)

class Seq2SeqwTag(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size,
                 enc_num_layers, dec_num_layers, dropout, st_mode, 
                 enc_cell=nn.LSTM, dec_cell=nn.LSTM, 
                 enc_bidirect=True, use_attn=True, tag_size=965
                 ):
        super(Seq2SeqwTag, self).__init__()
        # self.embedding = FactorizedEmbeddings(vocab_size, embed_size, embed_size//2)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.tag_embedding = nn.Embedding(tag_size, embed_size)
        self.tag_encoder = Encoder(embed_size, hidden_size, 1, 0, 
            bidirectional=False, cell=nn.LSTM)
        self.embed_proj = nn.Linear(hidden_size*2, hidden_size)

        self.encoder = Encoder(embed_size, hidden_size, enc_num_layers, dropout, bidirectional=enc_bidirect, cell=nn.LSTM)
        self.embed_dropout = nn.Dropout(0.1)
        attn = None
        if use_attn:
            print('Use Attention')
            attn = LuongAttention(encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size)
        self.decoder = Decoder(embed_size, vocab_size, hidden_size, dec_num_layers, dropout=dropout, 
            st_mode=st_mode, cell=nn.LSTM, attention=attn)
        orthogonal_initialization(self)

    def forward(self, inputs, labels, tags=None, temperature=1):
        embed = self.embedding(inputs)
        embed = self.embed_dropout(embed)
        output_embed = self.embedding(labels)
        latent, hidden = self.encoder(embed)

        if tags is not None:
            tags_embed = self.tag_embedding(tags)
            _, tag_hidden = self.encoder(tags_embed)
            merged_hidden_ = torch.cat((tag_hidden, hidden), axis=-1)
            hidden = self.embed_proj(merged_hidden_)

        outputs, d_h = self.decoder(output_embed[:, :-1, :], hidden, 
            encoder_outputs=latent, temperature=temperature)
        return outputs, d_h, hidden
    
    def decode(self, inputs, tags=None, temperature=1, target_length=-1, topk=1):
        embed = self.embedding(inputs)
        embed = self.embed_dropout(embed)
        latent, hidden = self.encoder(embed)

        if tags is not None:
            tags_embed = self.tag_embedding(tags)
            _, tag_hidden = self.encoder(tags_embed)
            merged_hidden_ = torch.cat((tag_hidden, hidden), axis=-1)
            hidden = self.embed_proj(merged_hidden_)

        decoder_outputs,decoder_tokens = [], []
        decoder_input = self.embedding(torch.tensor([[TOKENS['BOS']]], device=inputs.device))
        for di in range(target_length):
            outputs, hidden = self.decoder(decoder_input, hidden, 
                encoder_outputs=latent, temperature=temperature)
            _, topi = outputs.topk(1)
            hidden = hidden[0]
            decoder_outputs.append(outputs.squeeze(1))
            decoder_input = self.embedding(topi.detach()).squeeze(0)
            _, topi_ = outputs.topk(topk)

            decoder_tokens.append(topi_.cpu().detach().squeeze().numpy())
            if topi.item() == TOKENS['EOS']:
                break
        return np.array([decoder_tokens]), torch.stack(decoder_outputs, dim=1)

if __name__ == "__main__":
    encoder = Encoder(128, 128, 2, 0.1, bidirectional=True, cell=nn.LSTM)
    embedding = nn.Embedding(46895, 128)
    attn = LuongAttention(encoder_hidden_size=128, decoder_hidden_size=128)
    decoder = Decoder(128, 46895, 128, 2, dropout=0.1, 
        st_mode=False, cell=nn.LSTM, attention=attn)

    inputs = torch.randint(0, 46895,(50, 64)) # B x T
    labels = torch.randint(0, 46895,(50, 100)) # B x T
    labels[:, 3:] = TOKENS['PAD']
    criterion = nn.NLLLoss(reduction='mean', ignore_index=TOKENS['PAD'])

    embed = embedding(inputs)
    output_embed = embedding(labels)

    latent, hidden = encoder(embed)
    print('encoder input') # include EOS
    print(output_embed[0, -3:, :5])
    outputs, d_h = decoder(output_embed[:, :-1, :], hidden, encoder_outputs=latent)
    loss = 0

    seq_length = outputs.shape[1]
    for t in range(seq_length):
        loss += criterion(torch.log(outputs[:, t, :]), labels[:,t+1], )
    # loss.backward()
    print(loss)