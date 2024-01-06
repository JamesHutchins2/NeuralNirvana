from positional_encoding import AbsolutePositionalEncoding
from torch import nn
import torch
import torch.nn.functional as F
config = {
    "num_channels": 63,
    "epoch_length": 128,
    "embedding_dim": 64,
    "num_heads": 7,
    "num_layers": 6,
    "dropout": 0.1,
    
}

class EEGDataPatcher(nn.Module):
    def __init__(self, time_len=512, patch_size=4, in_chans=63, embed_dim=128):
        super().__init__()
        num_patches = time_len // patch_size
        self.patch_shape = patch_size
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Define the convolutional layer here
        self.conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

    def forward(self, x, **kwargs):
        #print(f"Shape of x: {x.shape}")
        
        x = x.view(-1, 63, 4)
        x_shaped = x.permute(0, 2, 1)

        # Use the conv layer defined in __init__
        x = self.conv(x_shaped)
        x = x.permute(0, 2, 1)
        
        return x

class masker:
    def __init__(self, mask_ratio=0.8, mask_value=0):
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def mask(self, x):
        N, L, D = x.shape  # N: Batch size, L: Sequence length, D: Feature dimension
        keep = int(L * (1 - self.mask_ratio))

        # Generate noise and sort it for each feature dimension
        noise = torch.rand(N, L, D, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)

        # Create a mask for the positions to keep
        binary_mask = torch.ones_like(x, device=x.device)
        for d in range(D):
            binary_mask[:, :, d].scatter_(1, ids_shuffle[:, :, d][:, :keep], 0)

        # Apply the mask
        x_masked = x.clone()
        x_masked[binary_mask.bool()] = self.mask_value
        
        #print(f"Shape of x_masked: {x_masked.shape}")
        #print(f"Shape of binary_mask: {binary_mask.shape}")
        
        binary_mask = binary_mask.transpose(-1, 0)

        return x_masked, binary_mask, 0
            
        
       
    


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)

    def forward(self, x):
        # The input x is expected to be of shape (N, L, E)
        # where N is the batch size, L is the sequence length, and E is the embedding dimension.

        # Permute x to the format (L, N, E) for MultiheadAttention
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)

        # Apply multi-head attention. 
        # Since the key, query, and value are all the same, we can pass x three times.
        # We don't use a mask here, but it can be added as an argument to forward if needed.
        attn_output, _ = self.multihead_attn(x, x, x)

        # Permute back to (N, L, E) for further processing in the network.
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output

        
class Transformer(nn.Module):
    def __init__(self, num_channels, epoch_length, embedding_dim, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.num_channels = num_channels
        self.epoch_length = epoch_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.norm = nn.LayerNorm(embedding_dim) # ** May need to be changed to 63
        # Positional encoding
        self.pos_encoder = AbsolutePositionalEncoding.tAPE()
        # patching method
        self.patch_encoder = EEGDataPatcher()
        
        # masking method
        self.masker = masker()
        
        """self.upsample = nn.ConvTranspose1d(in_channels=embedding_dim, 
                                           out_channels=embedding_dim, 
                                           kernel_size=4, stride=4)"""
        
        #create batch norm layer
        self.batch_norm = nn.BatchNorm1d(1)
        self.batch_norm_decoder = nn.BatchNorm1d(1)
        
        intermediate_dim = embedding_dim * 4
        
        # Custom self-attention layers
        self.self_attentions = nn.ModuleList([SelfAttention(embed_size=63, heads=7) for _ in range(num_layers)])
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(63, intermediate_dim),  # Adjust the input dimension
                nn.ReLU(),
                nn.Linear(intermediate_dim, 63),  # Adjust the output dimension if necessary
                nn.Dropout(dropout_rate)
            ) for _ in range(num_layers)
        ])
        # Decoder layers - excluding self-attention
        self.decoder_feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(63, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, 63),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_layers)
        ])

        # Final linear layer to produce the output
        self.final_linear = nn.Linear(63, 128)
        
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using Xavier uniform initialization
        print("init weights")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward_encoder(self, x):
        # patch the data
        #print(f"Shape of x (original): {x.shape}")
        x = self.patch_encoder.forward(x)
        
        patched_data = x
        
        #print(f"patched x shape: {x.shape}")
        x = x.squeeze(-1)
        #apply positional encoding
       
        x = self.pos_encoder(x)
        
        
        #apply masking
        x, mask, ids_restore = self.masker.mask(x)
        
        #print(f"Shape of x (post masking): {x.shape}")
        
        x = x.squeeze(0)
        
        #print(f"Shape of x (post squeeze): {x.shape}")
        
        #apply self attention and feed forward layers
        
        for self_attention, feed_forward in zip(self.self_attentions, self.feed_forward):
            #print("Before self-attention:", x.shape)
            x = self_attention(x)
            #print("After self-attention:", x.shape)
            x = feed_forward(x)
            #print("After feed-forward:", x.shape)
            
        # Apply global average pooling
        x = x.mean(dim=1, keepdim=True)

        self.batch_norm.to(x.device)

        # Apply global average pooling
        x = x.mean(dim=1, keepdim=True)

        # Apply Batch Normalization
        x = self.batch_norm(x)


        return x, mask, ids_restore, patched_data
    
    def forward_decoder(self, x):
        #print(f"Shape of x (start of decoder) (pre reshape): {x.shape}")
        decoded = x
        for feed_forward in self.decoder_feed_forward:
            decoded = feed_forward(decoded)
            #print(decoded.size(1))

        
        decoded = self.batch_norm_decoder(decoded)

        # Reshape and pass through final linear layer
        decoded = decoded.squeeze(1)
        output = self.final_linear(decoded)
        

        return output
        
        
        
    def forward(self, x):
        
        x, masks, ids_restore, patched_data = self.forward_encoder(x)
        #print(x.size(1))
        output = self.forward_decoder(x)
        return output, masks, ids_restore, patched_data
    
