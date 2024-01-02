import torch
import torch.nn as nn
import torch.nn.functional as F

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
block_size = 8
vocab_size = len(chars)
batch_size = 32
n_embs = 32
epoches = 5000
# epoches = 100
learning_rate = 1e-2

#===============================================================================
# Utils
#===============================================================================
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(epoches)
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
  return xb, yb


#===============================================================================
# Model definition
#===============================================================================

class Head(nn.Module):
  def __init__(self, n_embs, head_dim):
    super().__init__()
    self.query = nn.Linear(n_embs, head_dim)
    self.key = nn.Linear(n_embs, head_dim)
    self.value = nn.Linear(n_embs, head_dim)
    self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
    
  def forward(self, x):
    B, T, C = x.shape
    
    q = self.query(x)
    k = self.key(x)
    
    head_dim = k.shape[-1]
    wei = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
  
    v = self.value(x) # B, T, C
    out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
    return out

class BigramLM(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embd = nn.Embedding(vocab_size, n_embs)
    self.position_embd = nn.Embedding(block_size, n_embs)
    self.sa = Head(n_embs, head_dim=n_embs)
    self.lm_head = nn.Linear(n_embs, vocab_size)
    
    
  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_em = self.token_embd(idx) # (B, T, C) and C = vocab_size
    position_em = self.position_embd(torch.arange(T, device=idx.device)) # (T, C)
    x = token_em + position_em # (B, T, C)
    x = self.sa(x) # (B, T, C)
    
    logits = self.lm_head(x) # (B, T, C)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # (B*T, C)
      targets = targets.view(B*T)
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
  init = torch.zeros((1, 1), dtype=torch.long)
  o = m.generate(init, max_size)
  return decode(list(o[0].tolist()))


m = BigramLM()
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for _ in range(epoches):
  xb, yb = get_batch('train')
  optimizer.zero_grad(set_to_none=True)
  logits, loss = m(xb, yb)
  loss.backward()
  optimizer.step()
  # Pretty print loss every 10 epoches
  if _ % 400 == 0:
    losses = estimate_loss(m)
    print(f'Epoch {_} | Train loss: {losses["train"]:.2f} | Val loss: {losses["val"]:.2f}')
   
  
        
print(loss.item())

print(generate_text(m, 500))