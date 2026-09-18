import torch
import torch.nn as nn
from torch.nn import functional as F


with open('Eminescu.txt','r',encoding='utf-8') as f:
    text = f.read()
chars=sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

n_embd=32
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
batch_size = 64
block_size = 256
max_iters = 1000
eval_interval=500
learning_rate=3e-4
eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2


# tokenization process
# we create 2 map stoi and itos
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# we are using a simple char to integer tokenizator not a sub-word one
def encode(string):
    tokens = []
    for c in string:
        tokens.append(stoi[c])
    return tokens
def decode(tokens):
    string = ''
    for i in tokens:
        string+=itos[i]
    return string

# we created a tensor from all data we got from input file (eminescu poems)
# now we split the data into train and validation sets
data=torch.tensor(encode(text),dtype=torch.long)
torch.manual_seed(1337)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y
# so we basically got our input for the transformer 

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill((self.tril[:T,:T]) == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embd,n_embd)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    # this is a transformer block

    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size= n_embd // n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BiagramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),  
            nn.LayerNorm(n_embd),          
        )
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb=self.token_embedding_table(idx) 
        pos_emb=self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)
        # idx and targets are 2d (B,T) tensor of integers
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # cross_entropy expects logits of 2d dim and targets of 1d 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

    
m = BiagramLanguageModel().to(device)
 # now we are printing random char because we haven't trained our model yet 

# now we need to create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)
print(f'This is currently running on {device}')
for steps in range(max_iters):
    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 250 == 0:
        print(f"Pasul {steps}/{max_iters} | Loss: {loss.item():.4f}")
print(loss.item())
context = torch.zeros((1,1), dtype=torch.long,device=device)
with open('exemplu.txt','a',encoding=utf-8) as f:
    f.write(decode(m.generate(idx=context,max_new_tokens=1000)[0].tolist()))
# some improvement but not there yet!