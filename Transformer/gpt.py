import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

torch.manual_seed(203)

# -*- coding: utf-8 -*-

# Load data
with open('input.txt') as file:
  content = file.read()

chars = sorted(set(list(content)))
stoi = {c:i for i, c in enumerate(chars)}
itoc= {i:c for i, c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s ]
decode = lambda e: ''.join([itoc[i] for i in e])

tensor = torch.tensor(encode(content), dtype=torch.long)
train_sz = int(len(tensor)*0.9)
# Split train and valid
train_ts = tensor[:train_sz]
valid_ts = tensor[train_sz:]

#===============================================================================
# Parameters
#===============================================================================

# model parameters
block_size = 32 # number of characters to predict
vocab_size = len(chars)
batch_size = 4
n_embs = 32
num_heads = 4
num_layers = 4
dropout_rate = 0.2

# training parameters
epoches = 5000
# epoches = 100
learning_rate = 1e-2
eval_iter = 400
# Dry run/Debug
dry_run = True
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#===============================================================================
# Utils
#===============================================================================
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(epoches).to(device)
        for k in range(epoches):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_batch(ds:str = 'train'):
  data = train_ts if ds == 'train' else valid_ts
  ix = torch.randint(data.shape[0] - block_size, (batch_size,))
  xb = torch.stack([data[i:i+block_size] for i in ix])
  yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
  xb, yb = xb.to(device), yb.to(device)
  return xb, yb


#===============================================================================
# Model definition
#===============================================================================

class Head(nn.Module):
  def __init__(self, head_dim):
    super().__init__()
    self.query = nn.Linear(n_embs, head_dim)
    self.key = nn.Linear(n_embs, head_dim)
    self.value = nn.Linear(n_embs, head_dim)
    self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
    self.dropout = nn.Dropout(dropout_rate) 
  def forward(self, x):
    B, T, C = x.shape
    
    q = self.query(x)
    k = self.key(x)
    
    head_dim = k.shape[-1]
    wei = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
  
    v = self.value(x) # B, T, C
    out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
    return out

class MultiHead(nn.Module):
  def __init__(self, head_size, num_heads) -> None:
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embs)
    self.dropout = nn.Dropout(dropout_rate)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(out)
    return self.proj(out)
  
class Feedforward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embs, n_embs * 4),
      nn.ReLU(),
      nn.Linear(n_embs * 4, n_embs),
      nn.Dropout(dropout_rate)
    )
    
  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.sa = MultiHead(n_embs // num_heads, num_heads)
    self.ffwd = Feedforward()
    self.ln1 = nn.LayerNorm(n_embs)
    self.ln2 = nn.LayerNorm(n_embs)
    
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
    

class BigramLM(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embd = nn.Embedding(vocab_size, n_embs)
    self.position_embd = nn.Embedding(block_size, n_embs)
    self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
    self.lm_head = nn.Linear(n_embs, vocab_size)
    
    
  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_em = self.token_embd(idx) # (B, T, C) and C = vocab_size
    position_em = self.position_embd(torch.arange(T, device=idx.device)) # (T, C)
    x = token_em + position_em # (B, T, C)
    x = self.blocks(x)
    logits = self.lm_head(x) # (B, T, C)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # (B*T, C)
      targets = targets.view(B*T)
      # Smoothed loss

      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self,idx,max_generate):
    for _ in range(max_generate):
      idx_cond = idx[:,-block_size:] # B, T
      logits, _ = self(idx_cond) # B, T, C
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=-1) # B, C
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1) # B, T+1
    return idx

def generate_text(m, max_size):
  init = torch.zeros((1, 1), dtype=torch.long).to(device)
  o = m.generate(init, max_size)
  return decode(list(o[0].tolist()))

def estimate_params(m):
  return sum(p.numel() for p in m.parameters() if p.requires_grad)

def save_model(m, path):
  torch.save(m.state_dict(), path)

m = BigramLM()
m = m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
print("Number of parameters: ", estimate_params(m))
print("Running on", device)
print("Mode: ", "Dry run" if dry_run else "Normal")
eval_iter = 1 if dry_run else eval_iter

for iter in tqdm(range(epoches)):
  xb, yb = get_batch('train')
  optimizer.zero_grad(set_to_none=True)
  logits, loss = m(xb, yb)
  loss.backward()
  optimizer.step()
  # Pretty print loss every 10 epoches
  if iter % eval_iter == 0 or iter == epoches - 1:
    losses = estimate_loss(m)
    print(f'Epoch {iter} | Train loss: {losses["train"]:.2f} | Val loss: {losses["val"]:.2f}')
    if dry_run:
      break
   
  
        
print(loss.item())
print(generate_text(m, 500))
# save_model(m, 'model.pt')