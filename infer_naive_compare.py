"""
The "before vLLM" baseline: the same 64 prompts from infer_vllm.py's part 2,
generated one at a time with plain HuggingFace `generate()` - no batching, no
paged KV cache, nothing clever. Run this AFTER infer_vllm.py's process has
exited, not alongside it - vLLM reserves most of the GPU's memory up front for
its KV cache, so there isn't room to load a second model while it's still running.

pip install transformers torch
"""
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MAX_NEW_TOKENS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()

prompts = [f"Tell me a fact about the number {i}." for i in range(64)]

start = time.time()
tokens = 0
with torch.no_grad():
    for p in prompts:  # one prompt at a time - no batching at all
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        tokens += out.shape[1] - ids.shape[1]
elapsed = time.time() - start

print(f"HF, one prompt at a time: {tokens} tokens across {len(prompts)} prompts in {elapsed:.2f}s -> {tokens / elapsed:.1f} tok/s")
print("\ncompare this tok/s number against part 2's output from infer_vllm.py - same GPU, same model, same prompts.")
