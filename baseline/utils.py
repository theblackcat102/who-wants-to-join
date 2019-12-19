
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.init as init

from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

class IgnoreLogger(LightningLoggerBase):

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass

def format_token(token_name):
    return int(token_name.split('_')[1])


def extract_relation(filename):
    event_user_pair = open(filename, 'r')
    relation_event = defaultdict(list)
    for line in event_user_pair.readlines():
        tokens = line.split(' ')
        event = format_token(tokens[0])
        users = tokens[1:]
        relation_event[event] += [ format_token(u) for u in users ]
    return relation_event


def create_inv_map(relationgraph):
    relation_event = defaultdict(list)
    for key, value in relationgraph.items():
        for v in value:
            if key not in relation_event[v]:
                relation_event[v].append(key)
    return relation_event


def straight_through_estimate(p):
    shape = p.size()
    ind = p.argmax(dim=-1)
    p_hard = torch.zeros_like(p).view(-1, shape[-1])
    p_hard.scatter_(1, ind.view(-1, 1), 1)
    p_hard = p_hard.view(*shape)
    return ((p_hard - p).detach() + p)


def sample_gumbel(shape, eps=1e-20):
    u = torch.rand(shape)
    if torch.cuda.is_available():
        u = u.cuda()
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax(logits, temperature, st_mode=False):
    """
    Gumble Softmax
    Args:
        logits: float tensor, shape = [*, n_class]
        temperature: float
        st_mode: boolean, Straight Through mode
    Returns:
        return: gumbel softmax, shape = [*, n_class]
    """
    logits = logits + sample_gumbel(logits.size())
    return softmax(logits, temperature, st_mode)


def softmax(logits, temperature=1, st_mode=False):
    """
    Softmax
    Args:
        logits: float tensor, shape = [*, n_class]
        st_mode: boolean, Straight Through mode
    Returns:
        return: gumbel softmax, shape = [*, n_class]
    """
    y = torch.nn.functional.softmax(logits, dim=-1)
    if st_mode:
        return straight_through_estimate(y)
    else:
        return y

def binary_matrix(seq, pad_token=1):
    # 將targets裡非pad部分標記為1，pad部分標記為0
    m = []
    for token in seq:
        if token == pad_token:
            m.append(0)
        else:
            m.append(1)
    return m



def orthogonal_regularization(model, reg=1e-6):
    '''
    heavily reliant on memory
    '''
    with torch.enable_grad():
        orth_loss = torch.zeros(1)
        orth_loss = orth_loss.cuda()
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0]).cuda()
                orth_loss = orth_loss + (reg * sym.abs().sum())
    return orth_loss

def orthogonal_initialization(model):
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            nn.init.orthogonal(m.weight)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
    return model