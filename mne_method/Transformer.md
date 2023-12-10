# Transfomer Architecture and Process

## Intro

The function of the proposed transformer is to mask, encode and decode EEG data across 63 channels with the goal of re constructing the original unmasked data. The approach taken to this problem reflects heavily that of approaches taken to natural language processing (NLP). Data in previous steps has been arranged in such a way that subsets of the data reflect a series of tokens in a time series. Similar to NLP uses where word chunks are treated as successive tokens in a sentence. 



## Architecture

The architecture of the transformer is as follows:

1. Positional Encoding Mechanism
2. Masking Mechanism
3. Self Attention Mechanism
4. Encoder
5. Decoder
6. Output Layer
7. Loss Function