# LLM Evaluation Guide

## Smoke Test
- Load model with `transformers` (AutoModelForCausalLM + AutoTokenizer)
- Generate a short response: `model.generate(tokenizer("Hello", return_tensors="pt").input_ids, max_new_tokens=50)`
- Verify output is coherent text, not garbage

## Quality Check — Test Prompts

Run at least 3 diverse prompts:

1. **Instruction following**: "Explain quantum computing in 3 bullet points."
2. **Reasoning**: "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?"
3. **Code generation**: "Write a Python function that checks if a string is a palindrome."
4. **Japanese** (if multilingual): "日本の四季について簡潔に説明してください。"

Evaluate each output for:
- Coherence and fluency
- Factual accuracy
- Instruction adherence
- Token efficiency (verbose vs concise)

## Performance Measurement

```python
import time

# Warmup
model.generate(input_ids, max_new_tokens=10)

# Measure
times = []
for _ in range(5):
    start = time.perf_counter()
    output = model.generate(input_ids, max_new_tokens=128)
    elapsed = time.perf_counter() - start
    tokens_generated = output.shape[1] - input_ids.shape[1]
    times.append(tokens_generated / elapsed)

tokens_per_sec = sorted(times)[len(times) // 2]  # median
```

Key metrics:
- **tokens/sec** (median of 5 runs, 128 new tokens)
- **time to first token** (TTFT) if streaming is supported
- **latency p50/p99** for single-request scenarios

## Common Issues
- OOM: Try `torch_dtype=torch.bfloat16`, `device_map="auto"`, or quantization (`load_in_8bit=True`)
- Slow generation: Check if model is on CPU accidentally
- Gibberish output: Verify tokenizer matches model; check if chat template is needed
