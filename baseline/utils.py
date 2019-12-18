
from collections import defaultdict
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