"""
Run text generation with vLLM on GPT-2. This is the actual vLLM part of the
exercise: vLLM handles the KV cache, batching and sampling internally, so from
here it's just "give it a prompt, get text back". None of the internals are
visible, which is normal - that's what makes vLLM fast. For a look inside the
model layer by layer, see infer_visualize.py, which recomputes the same kind
of generation with plain HuggingFace transformers so it can hook into it.

pip install vllm
"""
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
