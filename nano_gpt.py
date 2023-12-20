# load data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# reproducibility
torch.manual_seed(42) 

# *****************************************************
# HYPERPARAMETERS
# *****************************************************
data_frac = 0.01                    # fraction of data to use
batch_size = 32                     # batch size
block_size = 20                     # context size for next token prediction
emb_dim = 512                       # embedding dimension
num_heads = 8                       # number of heads in multi-head attention
num_layers = 1                      # number of transformer blocks
lr = 1e-3                           # learning rate
max_iters = 1000                    # number of training iterations
eval_interval = max_iters // 50     # evaluate loss every eval_interval iterations
eval_iters = 20                     # number of batches to evaluate loss on


# *****************************************************
# DATASET: TINY SHAKESPEARE
# *****************************************************
corpus = open('tiny_shakespeare.txt', 'r').read()

# map each character to an integer and vice versa
chars = sorted(set(corpus))
idx2char = dict(enumerate(chars))
char2idx = {v: k for (k, v) in idx2char.items()}
vocab_size = len(chars)

# encode corpus (char to int)
corpus = corpus[:int(len(corpus)*data_frac)] if data_frac < 1 else corpus
enc_corpus = [char2idx[i] for i in corpus]

# split into train and validation sets
split_idx = int(len(enc_corpus)*0.9)
enc_train_corpus = enc_corpus[:split_idx]
enc_val_corpus = enc_corpus[split_idx:]

# get batch of data
def get_batch(split):
    enc_corpus = enc_train_corpus if split == 'train' else enc_val_corpus
    idxs = torch.randint(len(enc_corpus) - block_size, (batch_size,))
    xb = torch.stack([torch.tensor(enc_corpus[i:i+block_size]) for i in idxs]) # stack along batch dimension (0)
    yb = torch.stack([torch.tensor(enc_corpus[i+1:i+1+block_size]) for i in idxs])
    return xb, yb


# *****************************************************
# MODEL: NANO GPT MODULES
# *****************************************************
class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.W_emb = nn.Embedding(vocab_size, emb_dim)
        self.W_pos = nn.Parameter(torch.randn(emb_dim))
    
    def forward(self, x):
        x_emb = self.W_emb(x) + self.W_pos
        return x_emb
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.W_kqv = nn.Linear(emb_dim, 3*emb_dim)
        self.W_out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
       B, T, C = x.shape
       k, q, v = self.W_kqv(x).split(C, dim=-1) # (batch_size, seq_len, embedding_dim) (3x)
       k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) 
       q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
       v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

       wei = q @ k.transpose(-2, -1) / ((C // self.num_heads) ** 1/2)
       tril_mask = torch.ones((T, T)).tril()
       wei = wei.masked_fill(tril_mask == 0, -torch.inf)
       wei = F.softmax(wei, -1) # can add dropout after softmax
       x_att = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
       return self.dropout(self.W_out(x_att))
    
class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.att = MultiHeadAttention(emb_dim, num_heads)
        self.ff = FeedForward(emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
    def forward(self, x):
        x_att = self.att(x)
        x_att = self.norm1(x + x_att)
        x_ff = self.ff(x_att)
        x_ff = self.norm2(x_att + x_ff)
        return x_ff
       
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, num_layers):
        super().__init__()
        self.emb = Embedding(vocab_size, emb_dim)
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, num_heads) for _ in range(num_layers)])
        self.W_out = nn.Linear(emb_dim, vocab_size)
        
    def forward(self, x):
        x_emb = self.emb(x)
        for block in self.blocks:
            x_emb = block(x_emb)
        x_emb = self.W_out(x_emb)
        return x_emb # careful not to apply softmax here as nn.CrossEntropy already does that for us!
    
    @torch.no_grad()
    def generate(self, enc_text, max_new_tokens):
        self.eval()
        # keep generated text in tensor instead of a list and converting to tensor
        # in each loop. That is not efficient + alot of GPU I/O if using CUDA.
        for _ in range(max_new_tokens):
            probs = F.softmax(self(enc_text[:, -block_size:]), dim=-1) # (B=1, ≤block_size, vocab_size) 
            idx_next = torch.multinomial(probs[0, -1, :], num_samples=1) 
            enc_text = torch.cat((enc_text, idx_next.view(1, 1)), dim=1) # (B=1, ≤block_size)
            
        return enc_text


# *****************************************************
# TRAINING LOOP
# *****************************************************
model = TransformerLM(vocab_size, emb_dim, num_heads, num_layers)
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# model evaluation loop
# provides a more accurate estimate than the batch-wise loss
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        total_loss = 0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            logits = model(xb).view(-1, vocab_size)
            total_loss += loss_fn(logits, yb.view(-1)).item()
        out[split] = total_loss / eval_iters
    return out

# training loop
for i in range(max_iters):
    # set model to train mode
    model.train()

    # sample batch of data
    xb, yb = get_batch('train')

    # forward pass & update
    logits = model(xb).view(-1, vocab_size)
    loss = loss_fn(logits, yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # every so often evaluate loss on train and validation sets
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {i}: train_loss={losses["train"]:.5f}, val_loss={losses["val"]:.5f}')


# *****************************************************
# GENERATE TEXT
# *****************************************************
num_gen_tokens = 1000
start = torch.zeros((1, 1), dtype=torch.long)
enc_text = model.generate(start, max_new_tokens=num_gen_tokens)
print(''.join([idx2char[i] for i in enc_text.flatten().tolist()]))