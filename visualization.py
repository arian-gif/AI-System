import base64
import json
from pathlib import Path

import numpy as np
import torch


def quantize(t, lo, hi):
    #squash the floats into bytes (0..255) so the recording stays small, the viewer turns them back into colors
    a = t.detach().float().cpu().numpy()
    q = np.clip(np.round((a - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii")


# Records what the model does on ONE fixed chunk of text while it trains.
# Every time you call snap() we run that same chunk through the model and save:
#   - the numbers after the embedding and after every decoder block
#   - every head's attention table in every block
#   - the model's top 3 guesses for the next character at every position
#   - the embedding table (one row per character)
# save() then writes a single html file you open in a browser to play it back
class Recorder:
    def __init__(self, model, probe, vocab):
        #probe is T+1 token ids: the first T are the input, the last T are what the model should predict
        self.model = model
        self.vocab = vocab
        device = next(model.parameters()).device
        self.x = probe[:-1].unsqueeze(0).to(device)
        self.T = self.x.shape[1]
        self.text = [vocab[i] for i in probe[:-1].tolist()]
        self.targets = [vocab[i] for i in probe[1:].tolist()]
        self.snaps = []
        self.emb_tables = []
        self.losses = []

    def log_loss(self, step, train_loss, val_loss):
        self.losses.append([step, round(train_loss, 4), round(val_loss, 4)])

    @torch.no_grad()
    def snap(self, step):
        model = self.model
        was_training = model.training
        model.eval() #dropout off so the chunk looks the same every time

        #forward hooks let us peek at a layer's output without touching model.py
        #they fire in order: position encoding first, then block 1, block 2, ...
        stages = []
        grab = lambda module, inputs, output: stages.append(output[0]) #output[0] drops the batch dim -> (T, dim)
        hooks = [model.pos.register_forward_hook(grab)]
        hooks += [b.register_forward_hook(grab) for b in model.blocks]
        logits = model(self.x)
        for hook in hooks:
            hook.remove()

        probs = logits[0].softmax(dim=-1) #(T, vocab)
        top_p, top_i = probs.topk(3, dim=-1)
        #each attention block keeps its last (1, h, T, T) table in .attention_scores
        attn = torch.stack([b.attention.attention_scores[0] for b in model.blocks]) #(layers, h, T, T)

        self.emb_tables.append(model.embed.embeddings.weight.detach().cpu().clone())
        self.snaps.append({
            "step": step,
            "topi": top_i.tolist(),
            "topp": np.round(top_p.cpu().numpy(), 4).tolist(),
            #each layer gets its own color scale (centered on 0) since later layers have bigger numbers
            "acts": [quantize(s, -lim, lim) for s in stages for lim in [float(s.abs().max()) or 1.0]],
            "attn": quantize(attn, 0.0, 1.0),
        })

        if was_training:
            model.train()

    def _project_embeddings(self):
        #squash each 256-d embedding to 2d with PCA. we pick the axes from the FINAL table and reuse them
        #for every snapshot, otherwise the dots would jump around between frames for no reason
        final = self.emb_tables[-1]
        mean = final.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(final - mean, full_matrices=False)
        basis = vh[:2].T #(dim, 2)
        return [np.round(((w - mean) @ basis).numpy(), 3).tolist() for w in self.emb_tables]

    def save(self, path="training_viewer.html"):
        assert self.snaps, "call snap() at least once before saving"
        for snap, emb in zip(self.snaps, self._project_embeddings()):
            snap["emb"] = emb

        model = self.model
        data = {
            "T": self.T,
            "dim": model.projection.in_features,
            "heads": model.blocks[0].attention.h,
            "stages": ["embedding + position"] + [f"block {i + 1}" for i in range(len(model.blocks))],
            "vocab": self.vocab,
            "text": self.text,
            "targets": self.targets,
            "losses": self.losses,
            "snaps": self.snaps,
        }
        template = Path(__file__).with_name("viewer_template.html").read_text(encoding="utf-8")
        payload = json.dumps(data).replace("</", "<\\/") #so the data can never close the <script> tag early
        Path(path).write_text(template.replace("__DATA__", payload), encoding="utf-8")
