import torch
import torch.nn as nn 
import math

class InputEmbeddings(nn.Module):
    # Embedding the word, each word in the gpt's vocab is a row in the matrix and dim is how many dimensions we want embedding to be
    def __init__(self, dim: int = 256, vocab_size: int=65):
        super().__init__()
        self.dim = dim
        self.words = vocab_size
        self.embeddings = nn.Embedding(vocab_size, dim)

    #x is the token id, so the input mapped as tensor
    def forward(self,x):
        return self.embeddings(x) * math.sqrt(self.dim)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len:int, dropout: float, dim: int = 256)->None:
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create a matrix of shape (seq_len, dim)
        pe = torch.zeros(seq_len,dim)
        #create a vector of shape (seq_len)
        #create the equations
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))
        #apply the sin to even and odd to cosine
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1,seq_len,dim)

        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
