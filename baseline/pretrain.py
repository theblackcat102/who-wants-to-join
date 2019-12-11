import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from .dataset import Meetupv1, Meetupv2, TOKENS
from .models import Seq2Seq
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils import clip_grad_norm

def pretrain_model():
    

if __name__ == "__main__":
    FG = nx.Graph()