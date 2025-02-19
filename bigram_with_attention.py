import torch
import torch.nn as nn
import requests
from torch.nn import functional as F

# Hyperparameter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
context_size = 100
batch_size = 32
d_model = 384
nums_head = 6
nums_block = 6
# d_model = 384 / nums_head = 6
# head_size = 64
dropout = 0.1

epochs = 1
test_epochs = 1
lr = 3e-4
# Download data
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

# Write file
file_name = 'input.txt'
with open(file_name, 'w') as f:
   f.write(response.text)

# Open file and take data
with open(file_name, 'r', encoding='utf-8') as f:
   text = f.read()

# Unique chars in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping char to int and int to char
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}

# Encode and Decode


def encode(chars): return [stoi[char] for char in chars]
def decode(idx): return ''.join([itos[id] for id in idx])


# Train and text split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# Batch of data


def get_batch(split: str):
   data = train_data if split == 'train' else test_data

   # Random id
   idx = torch.randint(0, len(data) - context_size, (batch_size, ))
   x = torch.stack([data[id:id+context_size] for id in idx]).to(device)
   y = torch.stack([data[id+1:id+context_size+1] for id in idx]).to(device)

   return x, y

# One head attention


class Head(nn.Module):
   def __init__(self, head_size):
      super().__init__()
      self.wq = nn.Linear(d_model, head_size, bias=False)
      self.wk = nn.Linear(d_model, head_size, bias=False)
      self.wv = nn.Linear(d_model, head_size, bias=False)

      self.register_buffer('tril', torch.tril(
          torch.ones(context_size, context_size)))
      self.dropout = nn.Dropout(dropout)

   def forward(self, x: torch.Tensor):
      # x.shape = batch_size, context_length, d_model

      # shape = batch_size, context_length, head_size
      B, T, C = x.shape
      k = self.wk(x)
      q = self.wq(x)
      v = self.wv(x)

      # batch, context_length, context_length
      wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** -0.5)
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      wei = wei.softmax(dim=-1)
      wei = self.dropout(wei)
      return wei @ v  # batch_size, context_length, head_size

# Define Multihead attention


class MultiHead(nn.Module):
   def __init__(self, nums_head, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(nums_head)])
      # Change dimension of multihead to initial dimension
      self.projection = nn.Linear(head_size * nums_head, d_model)
      self.dropout = nn.Dropout(dropout)

   def forward(self, x: torch.Tensor):
      out = torch.cat([head(x) for head in self.heads], dim=-1)
      return self.dropout(self.projection(out))

# Define FeedForward


class FeedForward(nn.Module):
   def __init__(self, d_model):
      super().__init__()
      self.network = nn.Sequential(
          nn.Linear(d_model, 4 * d_model),
          nn.ReLU(),
          nn.Linear(4 * d_model, d_model),
          nn.Dropout(dropout)
      )

   def forward(self, x: torch.Tensor):
      return self.network(x)

# Define block (MultiHead attention + FeedForward)


class Block(nn.Module):
   def __init__(self, nums_head):
      super().__init__()
      head_size = d_model // nums_head
      self.heads = MultiHead(nums_head, head_size)
      self.ffw = FeedForward(d_model)
      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)

   def forward(self, x: torch.Tensor):
      # Nếu không có LayerNorm, giá trị sau khi cộng có thể tăng nhanh, gây bất ổn định.
      # LayerNorm giúp giữ phân phối ổn định trước khi truyền qua Attention hoặc FeedForward.

      x = x + self.heads(self.norm1(x))

      x = x + self.ffw(self.norm2(x))
      return x

# Define model


class GPT(nn.Module):
   def __init__(self):
      super().__init__()

      self.embedding = nn.Embedding(vocab_size, d_model)
      self.position_encoding = nn.Embedding(context_size, d_model)

      self.blocks = nn.Sequential(*[Block(nums_head)
                                  for _ in range(nums_block)])
      self.linear = nn.Linear(d_model, vocab_size)

   def forward(self, x: torch.Tensor, target: torch.Tensor = None):
      # x.shape = target.shape = batch_size, context_length
      B, T = x.shape
      # batch_size, context_length, d_model
      x = self.embedding(
          x) + self.position_encoding(torch.arange(T, device=device))
      x = self.blocks(x)  # batch_size, context_length, d_model
      logits = self.linear(x)  # batch_size, context_length, vocab_size

      if target is None:
         loss = None
      else:
         B, T, C = logits.shape
         logits = logits.view(B * T, C)
         target = target.view(B * T)
         loss = F.cross_entropy(logits, target)

      return logits, loss

   def generate(self, idx: torch.Tensor, max_length: int):
      # x.shape = batch, context
      for _ in range(max_length):
         idx_context = idx[:, -context_size:]
         logits, _ = self.forward(idx_context)
         logits = logits[:, -1, :]
         prob = F.softmax(logits, dim=-1)
         id_next = torch.multinomial(prob, num_samples=1)
         idx = torch.cat([idx, id_next], dim=1)  # add column
      return idx


def estimate_loss(model: nn.Module):
   out = {}
   with torch.inference_mode():
      model.eval()
      for split in ['train', 'test']:
         losses = torch.zeros(test_epochs)
         for test_epoch in range(test_epochs):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[test_epoch] = loss.item()
         out[split] = losses.mean()
      model.train()
      return out


# Create model
model = GPT().to(device)
# Create pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
   model.train()
   if epoch % 500 == 0:
      losses = estimate_loss(model)
      print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

   # Get batch of data
   x, y = get_batch('train')
   logits, loss = model(x, y)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

context = torch.zeros((1, 1), device=device, dtype=torch.long)
print(decode(model.generate(context, 100).squeeze().tolist()))
