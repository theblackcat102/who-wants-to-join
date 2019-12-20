import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
try: 
    from queue import PriorityQueue
except ImportError:
    from Queue import PriorityQueue
from .dataset import TOKENS
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = TOKENS['PAD']
        self.bos = TOKENS['BOS']
        self.eos = TOKENS['EOS']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            print(self.scores.unsqueeze(1).shape)
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]


if __name__ == "__main__":
    import torch
    from .train import str2bool
    from .test import load_params
    from .models import Seq2Seq
    from .dataset import Meetupv2, seq_collate
    from torch.utils.data import Dataset, DataLoader
    test = Meetupv2(train=False, sample_ratio=0.8, query='group', max_size=500, city='nyc', min_freq=10)
    stats = test.get_stats()
    # restore_path = 'lightning_logs/{}/checkpoints/_ckpt_epoch_{}.ckpt'.format('version_2', 9)
    # checkpoint = torch.load(restore_path)
    # params_f = 'lightning_logs/{}/meta_tags.csv'.format('version_2')
    # train_params = load_params(params_f)
    # model = Seq2Seq(
    #     embed_size=int(train_params['user_dim']),
    #     vocab_size=int(stats['member'])+3,
    #     enc_num_layers=int(train_params['enc_layer']),
    #     dec_num_layers=int(train_params['dec_layer']),
    #     dropout=0.1,
    #     st_mode=False,
    #     use_attn=str2bool(train_params['attn']),
    #     hidden_size=int(train_params['hidden']),
    #     tag_size=int(stats['topic'])+3,
    # )
    # model.load_state_dict({ key[6:] : value for key, value in checkpoint['state_dict'].items()})
    model = Seq2Seq(
        embed_size=16,
        vocab_size=int(stats['member'])+3,
        enc_num_layers=1,
        dec_num_layers=1,
        dropout=0.1,
        st_mode=False,
        use_attn=True,
        hidden_size=16,
        tag_size=int(stats['topic'])+3,
    )
 
    top_k = 1
    data = DataLoader(test, batch_size=1, num_workers=8, collate_fn=seq_collate)
    for batch in data:
        temperature = 1
        beam = Beam(10, cuda=True)
        existing_users, pred_users, cnts, tags = batch
        embed = model.embedding(existing_users)
        embed = model.embed_dropout(embed)
        latent, hidden = model.encoder(embed)
        target_length = pred_users.size(1)
        decoder_input = model.embedding(torch.tensor([[TOKENS['BOS']]], device=existing_users.device))
        for di in range(target_length):
            print(decoder_input.shape, hidden.shape)
            outputs, hidden = model.decoder(decoder_input, hidden, 
                encoder_outputs=latent, temperature=temperature)
            beam.advance(outputs[0])
            score, topi = beam.get_best()
            topi = topi.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            print(topi, score)
            decoder_input = model.embedding(topi.detach()).squeeze(0)
            hidden = hidden[0]

        