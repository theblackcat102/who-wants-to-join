import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import TOKENS
import random

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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        print('embedding_size ', input_size)
        print('embedding_dim ', hidden_size)
        # self.embedding = FactorizedEmbeddings(input_size, hidden_size, hidden_size//2)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=0.1)

    def forward(self, input, hidden):
        T, B = input.shape
        embedded = self.embedding(input)
        output = embedded

        hidden = hidden.repeat(1, B, 1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = embedding
        print('embedding_size ', output_size)
        print('embedding_dim ', hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=0.1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):

        output = self.embedding(input)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])

        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, hidden, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta  = nn.Parameter(torch.zeros(hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta



class Seq2Seq(nn.Module):
    def __init__(self, user_size, hidden_size, tag_size=966):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(user_size+3, hidden_size)
        self.tag_enc = EncoderRNN(tag_size, hidden_size)

        self.proj = nn.Sequential(
            nn.Dropout(0.4),
            # nn.BatchNorm1d(hidden_size*2),
            nn.Linear(hidden_size*2, hidden_size)
        )

        self.decoder = DecoderRNN(hidden_size, user_size+3, embedding=self.encoder.embedding)
    

    def forward(self, input_tensor, target_tensor, tag_tensor, criterion, device, max_length=MAX_LENGTH, 
        teacher_forcing_ratio=0.5, mapper=None):
        T, B = input_tensor.shape
        
        _encoder_hidden = self.encoder.initHidden(device=device)

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)

        decoder_loss = 0
        encoder_output, encoder_hidden = self.encoder( input_tensor, _encoder_hidden)
        
        _, tag_hidden = self.tag_enc(tag_tensor, _encoder_hidden)
        # print(tag_hidden.shape, encoder_hidden.shape)

        merged_hidden = torch.cat((tag_hidden, encoder_hidden), dim=2)
        # print(merged_hidden.shape)
        merged_hidden = self.proj(merged_hidden)
        # print(merged_hidden.shape)
        decoder_input = torch.tensor([[TOKENS['BOS']]*B], device=device)
        decoder_hidden = merged_hidden#[:, [0], :]

        decoder_outputs = []

        use_teacher_forcing = True if teacher_forcing_ratio > random.random() else False
        if self.training is False:
            use_teacher_forcing = False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[[di], :]  # Teacher forcing
                decoder_outputs.append(decoder_output.unsqueeze(0))
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                if len(decoder_input.shape) < 2:
                    decoder_input = decoder_input.unsqueeze(0)
                # print(decoder_input.shape, decoder_hidden.shape)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach().transpose(0, 1)  # detach from history as input
                # print(decoder_input.shape, decoder_hidden.shape)
                decoder_loss += criterion(decoder_output, target_tensor[di])
                decoder_outputs.append(decoder_output.unsqueeze(0))

                if B == 1 and decoder_input[0].item() == TOKENS['EOS']:
                    break
        # print(decoder_outputs[0].shape)
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        return decoder_loss, decoder_loss.item() / target_length, decoder_outputs



if __name__ == "__main__":
    model = Seq2Seq(hidden_size=128, user_size=46895)
    inputs = torch.randint(0, 46895,(50, 24))
    outputs = torch.randint(0, 46895,(50, 24))
    criterion = nn.CrossEntropyLoss()
    loss, norm_loss = model(inputs, outputs, criterion, device='cpu', use_teacher_forcing=False)