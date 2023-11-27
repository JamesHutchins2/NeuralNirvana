import torch 
import torch.nn as nn

class convelutionalEncoder(nn.Module):
        
    #passing in 1 epoch of the data at a time with a width of 63 for each of the channels. than the length in time/token size of the data
     
     #here we will proform a convelution to reduce dimenstionality of the EEg input data
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super(convelutionalEncoder, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.batchnorm = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

        

#self attention
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        #deifine the linear layers

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size) #square


    def forward(self, values, keys, query, mask):
            N = query.shape[0] #how many examples we are passing at once
            value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] #source input length and target input length

            #split embedding into self.heads pieces

            values = values.reshape(N, value_len, self.heads, self.head_dim)
            keys = keys.reshape(N, key_len, self.heads, self.head_dim)
            queries = query.reshape(N, query_len, self.heads, self.head_dim)


            #multiplying the keys and values
            energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) #nqhd = N, query_len, heads, head_dim, nkhd = N, key_len, heads, head_dim

            # queries shape: (N, query_len, heads, head_dim)
            # keys shape: (N, key_len, heads, head_dim)
            # energy shape: (N, heads, query_len, key_len)

            if mask is not None:
                energy = energy.masked_fill(mask==0, float("-1e20"))

            #normalize the attention scores
            attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

            out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)

            #attention shape: (N, heads, query_len, key_len)
            #values shape: (N, value_len, heads, heads_dim)
            #out shape: (N, query_len, heads, head_dim) then flatten

            #maps embed size to embed size
            out = self.fc_out(out)
            
            return out
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        #add skip connection, run through normalization and dropout
        self.norm1(attention + query) #skip connection
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)

        out = self.dropout(self.norm2(forward + x))

        return out
    


class Encoder(nn.Module):
         
        def __init__(
                    self,
                    src_freq_range,
                    embed_size,
                    num_layers,
                    heads,
                    devices,
                    forward_expansion,
                    dropout,
                    max_length,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    
            ):
                super(Encoder, self).__init__()
                self.embed_size = embed_size
                self.devices = devices
                self.conv = convelutionalEncoder(in_channels, out_channels, kernel_size, stride, padding)
                self.eeg_embedding = nn.Embedding(src_freq_range, embed_size)
                self.pos_embedding = nn.Embedding(max_length, embed_size)

                self.layers = nn.ModuleList(
                    [
                         #convolutionalEncoder(in_channels, out_channels, kernel_size, stride, padding)
                        convelutionalEncoder(
                             in_channels = in_channels, 
                             out_channels = out_channels, 
                             kernel_size = kernel_size, 
                             stride = stride, 
                             padding = padding
                        ),
                        TransformerBlock(
                            embed_size,
                            heads,
                            dropout=dropout,
                            forward_expansion=forward_expansion,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):
                N, seq_length, _ = x.shape
                #preform convolution
                x = self.conv(x)
                #p
                positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.devices)
                out = self.dropout(
                    (self.eeg_embedding(x) + self.pos_embedding(positions))
                )

                for layer in self.layers:
                    out = layer(out, out, out, mask)

                return out
     

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, devices):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        #undo convolution
        out = self.conv(out)

        return out
    

# for actual implementation we will want the following

# 1 a convolutional encoder to reduce the dimensionality of the data (on singular channels)
# 2 we will want 1 attention head per EEG channel
# 3 we will want to use the positional encoding to encode the time series data
# 4 we will want a cross attention layer to provide context between the attention head of each input channel
# 5 a series of feed forward layers to combine the data (may also need attention layers here)