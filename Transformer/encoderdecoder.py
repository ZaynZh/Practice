import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable



def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(1-subsequent_mask)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)


    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn,value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        vocab=source_vocab, d_model=d_model
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model



class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings,self).__init__()
        self.lat = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self,x):
        return self.lat(x) * math.sqrt(self.d_model)

    

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):

        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,dim)
        position = torch.arange(0,max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,dim,2) * -math.log(10000.0) / dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):

    def __init__(self, head, embedding_dim, dropout = 0.1):
        super(MultiHeadedAttention,self).__init__()

        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        self.linears = clones(nn.Linear(embedding_dim,embedding_dim), 4)

        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2) 
             for model,x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.embedding_dim)

        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):

    def __init__(self, dim1, dim2, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(dim1, dim2)
        self.linear2 = nn.Linear(dim2, dim1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return x
    

class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(dim))
        self.b2 = nn.Parameter(torch.zeros(dim))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std =  x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection,self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))        
    

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, size, masked_self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.masked_self_attn = masked_self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
        self.size = size

    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayer[0](x, lambda x: self.masked_self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, source_mask))
        return self.sublayer[2](x, self.feed_forward)
        

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)

        return self.norm(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embede, vocab, d_model):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embede
        self.out = nn.Linear(d_model, vocab)
        # self.generator = generator

    def encode(self, source, source_mask):
        encode_re = self.encoder(self.src_embed(source), source_mask)
        return encode_re
    
    def decode(self, memory, source_mask, target, target_mask):
        decode_re = self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)
        return decode_re
    
    def forward(self, source, target, source_mask, target_mask):
        rep = self.decode(self.encode(source, source_mask), source_mask, target, target_mask)
        output = self.out(rep)
        return output
    
