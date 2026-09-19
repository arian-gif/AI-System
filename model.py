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
#layer normalization normalizes the activations and then optimizes those parameters
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6)->None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim= True)
        std = x.std(dim =-1,keepdim=True)
        return self.weight*(x-mean)/(std+self.eps)+self.bias

#feed forward layer: after attention has mixed information between tokens, this runs on each token
#separately (expand dim -> d_ff, relu, shrink back to dim) so the model can process what it gathered
#the dropout inside is what helps with overfitting, it stops neurons relying too heavily on any pathway
class FeedForwardBlock(nn.Module):
    def __init__(self,dim:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(dim,d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,dim)
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# multi head attention: how similar each word is to previous words in the
# sequence (and itself), future words are masked out and never attended to
# each token's embedding gets projected into 3 separate vectors: Query, Key, Value
# we dot Query with every token's Key to get a similarity score:
# high dot product = more similar/relevant, low = less relevant
# Then we run it through a softmax function to determine what percent of
# relevance that token has to every other token (each row sums to 1)
# We use our Value tensor combined with those softmax weights (weighted sum)
# to produce the self-attention output
# We run multiple attention heads in parallel, each with its own learned
# Q/K/V projections, so different heads can specialize in different kinds
# of relationships between tokens
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim:int,h:int,dropout:float)->None:
        super().__init__()
        self.dim = dim
        self.h=h
        assert dim %h==0, "dim/head to get dk"
        self.d_k= dim //h
        self.w_q=nn.Linear(dim,dim)
        self.w_k=nn.Linear(dim,dim)
        self.w_v=nn.Linear(dim,dim)

        self.w_o = nn.Linear(dim,dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        return self.w_o(x)


#residual connection: add the self attention values with the word and position encoded values after
class ResidualConnection(nn.Module):

    def __init__(self,dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class DecoderBlock(nn.Module):
    def __init__(self,dim:int,h:int,d_ff:int,dropout:float,seq_len:int):
        super().__init__()

        self.attention = MultiHeadAttentionBlock(dim,h,dropout)
        self.feed_forward = FeedForwardBlock(dim,d_ff,dropout)
        self.attention_wrapper = ResidualConnection(dropout)
        self.feed_forward_wrapper = ResidualConnection(dropout)

        #causal mask of shape (1,1,seq_len,seq_len): lower triangle of 1s, so token i can only see tokens <= i
        mask = torch.tril(torch.ones(seq_len,seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask",mask)

    def forward(self, x):
        #x is (batch, T, dim), T can be shorter than seq_len so we cut the mask down to (T,T)
        T = x.shape[1]
        mask = self.mask[:,:,:T,:T]
        #self attention: q, k and v all come from the same x
        x = self.attention_wrapper(x, lambda x: self.attention(x,x,x,mask))
        x = self.feed_forward_wrapper(x, self.feed_forward)
        return x

# the whole gpt: embed the tokens, add position info, run through a stack of
# decoder blocks, then project back to vocab_size so every token gets a score
# (logits) for what the next token should be
class Transformer(nn.Module):
    def __init__(self, vocab_size:int=65, seq_len:int=256, dim:int=256, n_layers:int=6, h:int=8, d_ff:int=1024, dropout:float=0.1)->None:
        super().__init__()
        self.embed = InputEmbeddings(dim,vocab_size)
        self.pos = PositionalEncoding(seq_len,dropout,dim)
        self.blocks = nn.ModuleList([DecoderBlock(dim,h,d_ff,dropout,seq_len) for _ in range(n_layers)])
        #residual connections normalize before each sublayer, so the last block's output needs one more norm
        self.norm = LayerNormalization()
        self.projection = nn.Linear(dim,vocab_size)

        #xavier init on the weight matrices (skips 1d params like biases and the layer norm)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #x is token ids of shape (batch, T), returns logits of shape (batch, T, vocab_size)
    def forward(self, x):
        x = self.pos(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.projection(self.norm(x))
