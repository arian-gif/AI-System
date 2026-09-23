"""
Run text generation with vLLM on GPT-2. This is the actual vLLM part of the
exercise: vLLM handles the KV cache, batching and sampling internally, so from
here it's just "give it a prompt, get text back". None of the internals are
visible, which is normal - that's what makes vLLM fast. For a look inside the
model layer by layer, see infer_visualize.py, which recomputes the same kind
of generation with plain HuggingFace transformers so it can hook into it.

Part 1 below (one prompt) never actually needs vLLM - any inference library
does that. Part 2 is the reason vLLM exists: it sends 64 unrelated prompts in
ONE call, so vLLM's continuous batching (slot a new prompt in the instant a
GPU slot frees up, don't wait for the whole batch to finish) and PagedAttention
(the KV cache - each token's remembered attention state - is handed out in
small pages instead of one big reserved block per prompt, so far more requests
fit in GPU memory at once) actually get exercised. A single prompt has nothing
to pack efficiently or interleave, so it looks the same in any library.

To see what vLLM is saving you from, run infer_naive_compare.py separately
AFTER this script's process has exited (so its reserved GPU memory is freed -
vLLM grabs most of the GPU's memory up front for its KV cache, so a second
model can't safely load alongside it in the same process).

pip install vllm
"""
import time

from vllm import LLM, SamplingParams

PROMPT = "To be, or not to be, that is the"
MAX_NEW_TOKENS = 40

#"gpt2" alone is rejected by newer huggingface_hub versions - HF renamed the repo
#under the openai-community namespace, so the full id is required now
llm = LLM(model="openai-community/gpt2")  # downloads gpt2 (~500MB) from HuggingFace the first time
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_NEW_TOKENS, logprobs=5)

[output] = llm.generate([PROMPT], params)
completion = output.outputs[0]

print("prompt:", PROMPT)
print("generated:", completion.text)
print()
print("token by token, with the top-5 alternatives vLLM considered at each step:")
for tok_id, logprob_dict in zip(completion.token_ids, completion.logprobs):
    chosen = logprob_dict[tok_id]
    top5 = sorted(logprob_dict.items(), key=lambda kv: -kv[1].logprob)[:5]
    alt_str = ", ".join(f"{lp.decoded_token!r}:{lp.logprob:.2f}" for _, lp in top5)
    print(f"  chose {chosen.decoded_token!r:12s} logprob {chosen.logprob:6.2f}   top5: {alt_str}")

# ---- part 2: the actual reason vLLM exists ---------------------------------
# 64 unrelated prompts, sent as a SINGLE call. vLLM schedules and runs them
# together instead of one at a time - that's continuous batching. compare the
# tok/s this prints against infer_naive_compare.py's plain-HuggingFace loop.
print("\n" + "=" * 70)
print("part 2: 64 prompts in one vLLM call (continuous batching + PagedAttention)")
batch_prompts = [f"Tell me a fact about the number {i}." for i in range(64)]
batch_params = SamplingParams(temperature=0, max_tokens=50)  # temperature=0: always take the top guess

start = time.time()
batch_outputs = llm.generate(batch_prompts, batch_params)
elapsed = time.time() - start
tokens = sum(len(o.outputs[0].token_ids) for o in batch_outputs)
print(f"vLLM (batched): {tokens} tokens across {len(batch_prompts)} prompts in {elapsed:.2f}s -> {tokens / elapsed:.1f} tok/s")
print("\nsample output:", batch_prompts[0], "->", batch_outputs[0].outputs[0].text)
