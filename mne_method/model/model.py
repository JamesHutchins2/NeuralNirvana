from positional_encoding import AbsolutePositionalEncoding
from positional_encoding import tAPE
from positional_encoding import LearnablePositionalEncoding
from torch import nn
from . import EEG_patch
import torch


config = {
    "num_channels": 63,
    "epoch_length": 128,
    "embedding_dim": 64,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
    
}

class Transformer(nn.Module):
    
    def __init__(self, config):
        
        self.num_channels = config["num_channels"]
        self.epoch_length = config["epoch_length"]
        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        
        
    


