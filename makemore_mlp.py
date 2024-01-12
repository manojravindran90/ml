# %%
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# %%
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {s:i for i,s in stoi.items()}
vocab_size = len(stoi)

# %%
# creating the dataset
context_len = 3

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * context_len
        for ch in w+'.':
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# creating the test/train/validation sets
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

X_train, Y_train = build_dataset(words[:n1]) # 80% of the data
X_test, Y_test = build_dataset(words[n1:n2]) # 10% of the data
X_dev, Y_dev = build_dataset(words[n2:]) # 10% of the data

def calculate_loss():
    return None

print('X_train.shape -->', X_train.shape, 'Y_train.shape -->', Y_train.shape)
print('X_test.shape -->', X_test.shape, 'Y_test.shape -->', Y_test.shape)
print('X_dev.shape -->', X_dev.shape, 'Y_dev.shape -->', Y_dev.shape)


# %%
# network parameters
n_embd = 10
n_hidden = 200
batch_size = 32
g = torch.Generator().manual_seed(2147483647)
C = torch.randn(vocab_size, n_embd, generator=g) 
W1 = torch.randn(n_embd * context_len, n_hidden, generator=g) * (5/3) / ((n_embd * context_len) ** 0.5) # 5/3 since the tanh saturates at 5/3
# b1 = torch.randn(n_hidden, generator=g)# * 0.01
W2 = torch.randn(n_hidden, vocab_size, generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0
bngain = torch.ones(n_hidden)
bnbias = torch.zeros(n_hidden)
bnrunning_mean = torch.zeros(n_hidden)
bnrunning_std = torch.ones(n_hidden)
parameters = [C, W1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True


# %%
# model
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    idx = torch.randint(0 ,X_train.shape[0], (batch_size,), generator=g) # (batch_size,)
    Xb, Yb = X_train[idx], Y_train[idx] # (batch_size, context_len), (batch_size,)
    emb = C[Xb]  # (batch_size, context_len, n_embd)
    embcat = emb.view(emb.shape[0], -1) # (batch_size, n_embd * context_len)
    hpreact = embcat @ W1# + b1 # (batch_size, n_hidden)
    # adding batch normalization
    bnmeani = hpreact.mean(dim=0, keepdim=True)
    bnstdi = hpreact.std(dim=0, keepdim=True)
    hpreact = (hpreact - bnmeani) / bnstdi
    with torch.no_grad():
        bnrunning_mean = 0.999 * bnrunning_mean + 0.001 * bnmeani
        bnrunning_std = 0.999 * bnrunning_std + 0.001 * bnstdi
    hpreact = (bngain  * hpreact) + bnbias
    h = torch.tanh(hpreact) # (batch_size, n_hidden)
    logits = h @ W2 + b2 # (batch_size, vocab_size)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()
    # update parameters
    lr = 0.1 if i < int(max_steps/2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    # append loss to lossi
    if i%10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

# %%
# with torch.no_grad():
#     #calcultae the bnmean and bnbias as stage 2
#     emb = C[X_train]  # (batch_size, context_len, n_embd)
#     embcat = emb.view(emb.shape[0], -1) # (batch_size, n_embd * context_len)
#     hpreact = embcat @ W1 # + b1 # (batch_size, n_hidden)
#     bnmean = hpreact.mean(dim=0, keepdim=True)
#     bnstd = hpreact.std(dim=0, keepdim=True)

# %%
@torch.no_grad()
def split_loss(split):
    x, y = {
        'test' : (X_test, Y_test),
        'train' : (X_train, Y_train),
        'dev' : (X_dev, Y_dev)
    }[split]
    emb = C[x]  # (batch_size, context_len, n_embd)
    embcat = emb.view(emb.shape[0], -1) # (batch_size, n_embd * context_len)
    hpreact = embcat @ W1 # + b1 # (batch_size, n_hidden)
    # adding batch normalization
    hpreact = (hpreact - bnrunning_mean) / bnrunning_std
    hpreact = (bngain  * hpreact) + bnbias
    h = torch.tanh(hpreact) # (batch_size, n_hidden)
    logits = h @ W2 + b2 # (batch_size, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(f'{split} loss: {loss:.4f}')

split_loss('train')
split_loss('test')
split_loss('dev')

# %%
# before any optimizations
    # train loss: 2.1190
    # test loss: 2.1607
    # dev loss: 2.1675

# Fixing the initialization of the w2 and setting b2 to zero 
    # train loss: 2.0693
    # test loss: 2.1332
    # dev loss: 2.1368

# Fixing the oversaturation of the tanh layer by scaling down the w1 and b1
    # train loss: 2.0343
    # test loss: 2.1023
    # dev loss: 2.1058

# Loss after adding batch normalization
    # train loss: 2.0668
    # test loss: 2.1048
    # dev loss: 2.1065


# %%
# plt.figure(figsize=(10, 20))
# plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')

# %%
# h_act = h.view(-1).tolist()
# plt.hist(h_act, bins=50);
# hpreact_act = hpreact.view(-1).tolist()
# # plt.hist(hpreact_act, bins=50);

# %%
# plt.plot(lossi)

# %%
# sampling from the model
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(10):
    str = ''
    idx = [0] * context_len
    while True:
        emb = C[torch.tensor([idx])]
        embcat = emb.view(1, -1) # (batch_size, n_embd * context_len)
        hpreact = embcat @ W1# + b1 # (batch_size, n_hidden)
        # adding batch normalization
        hpreact = (hpreact - bnrunning_mean) / bnrunning_std
        hpreact = (bngain  * hpreact) + bnbias
        h = torch.tanh(hpreact) # (batch_size, n_hidden)
        logits = h @ W2 + b2 # (batch_size, vocab_size)
        probs = F.softmax(logits, dim=1)
        new_idx = torch.multinomial(probs, num_samples=1, generator=g).item()
        if new_idx == 0:
            break;
        str += itos[new_idx]
        idx = idx[1:] + [new_idx]
    print(str)
    

# %%


# %%


# %%



