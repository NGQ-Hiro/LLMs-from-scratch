import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


torch.manual_seed(1337)

# Download file text
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

response = requests.get(url)

with open("input.txt", 'w') as f:
   f.write(response.text)

with open("input.txt", 'r') as f:
   text = f.read()

# Unique chars in text
chars  = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping char to int and int to char
stoi = {char : i for i, char in enumerate(chars)}
itos = {i : char for i, char in enumerate(chars)}

# Encode and Decode
encode = lambda chars: [stoi[char] for char in chars]
decode = lambda ids: ''.join([itos[id] for id in ids])

# Train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# Batch of data
def get_batch(split):
   if split == 'train':
      data = train_data
   else:
      data = test_data
   # Get random id for sample
   idx = torch.randint(0, len(data) - block_size, (batch_size,))
   x = torch.stack([data[id:id+block_size] for id in idx])
   y = torch.stack([data[id+1:id+block_size+1] for id in idx])

   # Change to device
   x, y = x.to(device), y.to(device)
   
   return x, y

# Get loss
def estimate_loss(model: nn.Module):
   out = {}
   with torch.inference_mode():
      for split in ['train', 'val']:
         losses = torch.zeros(eval_iters)
         for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss
         out[split] = losses.mean()
   return out


# Define Bigram model
class BigramModel(nn.Module):
   def __init__(self, vocab_size: int):
      super().__init__()
      self.embedding = nn.Embedding(vocab_size, vocab_size)

   def forward(self, x: torch.Tensor, target: torch.Tensor = None):
      # target.shape = batch_size, context_length
      logits = self.embedding(x) # batch_size, context_length, vocab
      
      if target is None:
         loss = None
      else:
         B, T, C = logits.shape
         logits = logits.view(B * T, C)
         target = target.view(B * T)
         
         loss = F.cross_entropy(logits, target)
      return logits, loss
   
   def generate(self, idx, max_length):
      for _ in range (max_length):
         logits, loss = self(idx)
         # Focus on last 
         logits = logits[:, -1, :]
         # Softmax
         probs = F.softmax(logits, dim=-1)
         # Next token
         idx_next = torch.multinomial(probs, num_samples=1)
         # Append to idx
         idx = torch.cat((idx, idx_next), dim=1)
      return idx
   
# Create model
model = BigramModel(vocab_size=vocab_size).to(device)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for iter in range(max_iters):
   if iter % 300 == 0:
      out = estimate_loss(model)
      print(f"Step: {iter} | train loss: {out['train']:.4f} | val loss: {out['val']:.4f}")
   
   model.train()
   # Get data
   x, y = get_batch('train')
   # Eval loss
   logits, loss = model(x, y)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))



   
