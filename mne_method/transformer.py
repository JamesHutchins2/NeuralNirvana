
from torch import nn
import torch
import numpy as np
from timm.models.vision_transformer import Block


class Transformer(nn.Module):
    
    def __init__(self, time = 512, patch_len = 4, channels = 63, enc_depth = 12, 
                 dec_depth = 4, num_heads=16, embedd_dim = 512,
                 decoder_emb_dim=512, decoder_num_heads = 16, 
                 mlp_ratio = 4.0
                 ):
        super().__init__()
        
    
        self.patch_len = patch_len
        self.time = time
        self.channels = channels
        self.patch_dim = channels * patch_len
        self.num_patches = int(time / patch_len)
        self.embedd_dim = embedd_dim   
        #patchitfy
        #add cls
        #add positional encoding
        
        #__________________________encoder________________________________
        
        self.blocks = nn.ModuleList([
            Block(
                self.embedd_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
                )
            for i in range(enc_depth)
            self.norm = nn.LayerNorm(self.embedd_dim)
            ])
        
        
        
        
        
        