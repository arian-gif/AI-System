"""
Generates from GPT-2 one token at a time with plain HuggingFace transformers
(not vLLM - see the note in infer_vllm.py for why) and records what happens
inside the model at every step: the residual stream after every block, every
head's attention, and the MLP neuron activations for the token about to be
produced. Writes a single playable inference_viewer.html.

pip install transformers torch
"""
import base64
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PROMPT = "To be, or not to be, that is the"
MAX_NEW_TOKENS = 20  # kept small: the html grows with every token generated

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

n_layers = model.config.n_layer  # 12
n_heads = model.config.n_head  # 12
dim = model.config.n_embd  # 768
d_ff = model.config.n_inner or 4 * dim  # 3072, the width of the MLP's hidden layer


def quantize(t, lo, hi):
    #squash floats into bytes (0..255) so the recording stays small, the viewer turns them back into colors
    a = t.detach().float().cpu().numpy()
    q = ((a - lo) / (hi - lo) * 255).clip(0, 255).round().astype("uint8")
    return base64.b64encode(q.tobytes()).decode("ascii")


def vis(tok_str):
    #gpt2's tokenizer marks a leading space with 'Ġ' and a newline with 'Ċ', make those readable
    return tok_str.replace("Ġ", "␣").replace("Ċ", "↵")


# a forward hook on each block's MLP activation function grabs the "neuron firing"
# layer: d_ff (3072) numbers per token, right after the GELU nonlinearity and
# before they get projected back down to `dim`. this is the widest, most literal
# view of "neurons" the model has - each of those 3072 numbers is one neuron
neuron_acts = [None] * n_layers


def make_hook(i):
    def hook(module, inputs, output):
        neuron_acts[i] = output[0, -1]  # just the newest token's row, shape (d_ff,)
    return hook


for i, block in enumerate(model.transformer.h):
    block.mlp.act.register_forward_hook(make_hook(i))

ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
prompt_len = ids.shape[1]
snaps = []

for step in range(MAX_NEW_TOKENS + 1):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, output_attentions=True)

    tokens = [vis(t) for t in tokenizer.convert_ids_to_tokens(ids[0])]
    logits = out.logits[0]  #(T, vocab)
    top_p, top_i = logits.softmax(dim=-1).topk(3, dim=-1)
    top_str = [[vis(tokenizer.convert_ids_to_tokens([tid])[0]) for tid in row] for row in top_i.tolist()]

    #hidden_states is (embeddings, block 1 output, ... block 12 output), one entry longer than n_layers
    hidden = out.hidden_states
    #attentions is one (batch, heads, T, T) tensor per block, stack them into (layers, heads, T, T)
    attn = torch.stack([a[0] for a in out.attentions])

    snaps.append({
        "step": step,
        "tokens": tokens,
        "topstr": top_str,
        "topp": top_p.cpu().numpy().round(4).tolist(),
        "acts": [quantize(h[0], -lim, lim) for h in hidden for lim in [float(h.abs().max()) or 1.0]],
        "neurons": [quantize(n, 0.0, float(n.max()) or 1.0) for n in neuron_acts],
        "attn": quantize(attn, 0.0, 1.0),
    })

    if step == MAX_NEW_TOKENS:
        break
    next_id = logits[-1].argmax().view(1, 1)  #greedy: always pick the top guess, so playback is reproducible
    ids = torch.cat([ids, next_id], dim=1)

print("generated:", tokenizer.decode(ids[0]))

data = {
    "dim": dim, "d_ff": d_ff, "heads": n_heads, "layers": n_layers,
    "stages": ["embedding + position"] + [f"block {i + 1}" for i in range(n_layers)],
    "prompt_len": prompt_len,
    "snaps": snaps,
}
template = Path(__file__).with_name("inference_viewer_template.html").read_text(encoding="utf-8")
payload = json.dumps(data).replace("</", "<\\/")  #so the data can never close the <script> tag early
Path("inference_viewer.html").write_text(template.replace("__DATA__", payload), encoding="utf-8")
print("wrote inference_viewer.html")
