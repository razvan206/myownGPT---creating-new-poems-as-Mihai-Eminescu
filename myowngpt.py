with open('Eminescu.txt','r',encoding='utf-8') as f:
    text = f.read()
chars=sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

n_embd=32


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
print(encode("hi my name is razvan"))
print(decode(encode('hi my name is razvan')))
    
import torch 
data=torch.tensor(encode(text),dtype=torch.long)
print (data.shape,data.dtype)
print(data[:1000])

# we created a tensor from all data we got from input file (eminescu poems)
# now we split the data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
x = train_data[:block_size] # this is the context
y = train_data[1:block_size+1] # and this is the target that we need to find so if we have the first 8 chars we predict the ninth
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} the target:{target}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8 

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb,yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('------')
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target :{target}")

# so we basically got our input for the transformer 
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C=x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill((self.tril[:T,:T]) == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class BiagramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb=self.token_embedding_table(idx) 
        pos_emb=self.poisition_embedding_table(torch.arange(T,device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
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
            logits,loss=self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

    
m = BiagramLanguageModel()
logits,loss=m(xb,yb)
print(logits.shape)
print(loss)
 # now we are printing random char because we haven't trained our model yet 
print(decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))

# now we need to create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)
batch_size=32
for steps in range(10000):
    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))
# some improvement but not there yet!