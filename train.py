import os
import urllib.request

import torch
import torch.nn.functional as F

from model import Transformer
from visualization import Recorder

seq_len = 256
batch_size = 64
max_steps = 5000
lr = 5e-4
eval_every = 250
save_every = 1000
probe_len = 32 

device = "cuda" if torch.cuda.is_available() else "cpu"

# data: the tiny shakespeare text, one token per character
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(DATA_URL, "input.txt")
text = open("input.txt", encoding="utf-8").read()

vocab = sorted(set(text)) #65 characters
stoi = {c: i for i, c in enumerate(vocab)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(which):
    d = train_data if which == "train" else val_data
    ix = torch.randint(len(d) - seq_len - 1, (batch_size,))
    x = torch.stack([d[i:i + seq_len] for i in ix])
    y = torch.stack([d[i + 1:i + seq_len + 1] for i in ix]) #y is x shifted by one: the next character at every position
    return x.to(device), y.to(device)


def compute_loss(x, y):
    logits = model(x) #(batch, T, vocab)
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for which in ["train", "val"]:
        out[which] = sum(compute_loss(*get_batch(which)).item() for _ in range(20)) / 20
    model.train()
    return out


model = Transformer(vocab_size=len(vocab), seq_len=seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, training on {device}")

# the visualization follows the very first line of the dataset ("First Citizen:\nBefore we proceed...")
recorder = Recorder(model, data[:probe_len + 1], vocab)
recorder.snap(0) #the model before it has learned anything


def should_snap(step):
    #snapshot often at the start (that's when the model changes fastest), then every 100 steps
    return (step <= 100 and step % 10 == 0) or step % 100 == 0


for step in range(1, max_steps + 1):
    x, y = get_batch("train")
    loss = compute_loss(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #stops one bad batch from blowing up the weights
    optimizer.step()

    if should_snap(step):
        recorder.snap(step)

    if step % eval_every == 0:
        losses = estimate_loss()
        recorder.log_loss(step, losses["train"], losses["val"])
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if step % save_every == 0 or step == max_steps:
        #saved as we go so a colab disconnect doesn't lose everything
        recorder.save("training_viewer.html")
        torch.save(model.state_dict(), "checkpoint.pt")

print("done, download training_viewer.html and open it in your browser")
