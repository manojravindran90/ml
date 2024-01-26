# %%
#imports
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


# %%
# data load and dict creation

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
print(''.join(chars))

# %%
# helper function for string and index conversions

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda i: ''.join(itos[int] for int in i)

# %%


# %%
# creating torch tensor from data
data = encode(text)
data = torch.tensor(data, dtype=torch.long)
print(data.shape, data.dtype)

batch_size = 4
block_size = 8

# creating train and val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[idx:idx+block_size] for idx in ix])
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in ix])
    return x,y


# %%
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx) #B,T,C
        if targets == None:
            return logits
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits,loss        
    
    def generate(self, idx, max_iterations):
        for _ in range(max_iterations):
            logits = self(idx, None)
            # take just the last example since this is a bigram model
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# %%
for i in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss)

# %%
print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long), max_iterations=500)[0].tolist()))

# %%



