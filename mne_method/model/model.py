from positional_encoding import AbsolutePositionalEncoding
from positional_encoding import tAPE
from positional_encoding import LearnablePositionalEncoding
from torch import nn
from . import EEG_patch
import torch


config = {
    "num_channels": 63,
    "epoch_length": 128,
    "embedding_dim": 63,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
    
}

class Transformer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_channels = config["num_channels"]
        self.epoch_length = config["epoch_length"]
        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        
        self.pos_encoder = tAPE(self.embedding_dim, self.dropout, self.epoch_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads, dropout=self.dropout)
        self.encoder_layers = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.embedding = nn.Embedding(self.epoch_length, self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, self.num_channels)
        
        #call weight init
        self._init_weights()
        
    def _init_weights(self):
        #use the xavier uniform initializer
        #for p in self.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_uniform_(p)
        pass
    
    
    def forward(self, x):
        #x = self.embedding(x)
        #x = self.pos_encoder(x)
        #x = self.encoder_layers(x)
        #x = self.linear(x)
        #return x
        pass
        
    


