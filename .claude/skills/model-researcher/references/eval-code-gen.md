# Code Generation Model Evaluation Guide

## Representative Models
- Qwen2.5-Coder (0.5B-32B), Qwen3-Coder
- DeepSeek-Coder-V2 (MoE), DeepSeek-V3
- StarCoder2 (3B/7B/15B, Apache 2.0)
- CodeGemma (2B/7B)

## Framework
Standard `transformers` with `AutoModelForCausalLM`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Smoke Test
- Load model
- Generate code for a simple prompt: "Write a Python function to reverse a string."
- Verify output is syntactically valid Python (try `compile()`)

## Quality Check — Test Tasks

Run at least 4 tasks covering different capabilities:

1. **Function generation**:
   "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes."
   - Check: Correct algorithm, handles edge cases (n=0, n=1)

2. **Bug fixing**:
   ```
   Fix this code:
   def factorial(n):
       if n == 0: return 0
       return n * factorial(n-1)
   ```
   - Check: Identifies `return 0` should be `return 1`

3. **Multi-language** (if applicable):
   "Write the same binary search function in Python, JavaScript, and Rust."
   - Check: Idiomatic code in each language

4. **Explanation**:
   "Explain what this code does: `lambda f: (lambda x: x(x))(lambda x: f(lambda *a: x(x)(*a)))`"
   - Check: Correctly identifies Y combinator

### Execution Verification
Actually run generated code when possible:
```python
import subprocess, tempfile
with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
    f.write(generated_code)
    f.flush()
    result = subprocess.run(
        ["python", f.name], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
```

## Fill-in-the-Middle (FIM) Testing
Many code models support FIM. Use model-specific tokens:
```python
# Qwen2.5-Coder FIM format
prompt = "<|fim_prefix|>def add(a, b):\n<|fim_suffix|>\n    return result<|fim_middle|>"
```
Check that the model correctly fills the gap.

## Performance Measurement
Same as LLM: tokens/sec (median of 5 runs, 128 new tokens).

Key metrics: tokens/sec, pass@1 on test cases

## Standard Benchmarks (for reference)
- **HumanEval / HumanEval+**: 164 Python problems (saturated at top)
- **BigCodeBench-Hard**: Real-world library API usage (better discriminator)
- **LiveCodeBench**: Competitive programming, harder
- **SWE-bench**: GitHub issue resolution (hardest, most practical)

## Common Issues
- **FIM token mismatch**: Each model uses different special tokens. Wrong tokens = garbage output.
- **Quantization degrades code quality more than prose**: Q4_K_M is minimum safe level.
- **Instruct vs base**: Base models are better for FIM; instruct for chat-style generation.
- **Context length**: 32k minimum for real codebases; prefer 128k models.
- **HumanEval scores are misleading**: Top models all score 90+; use harder benchmarks.
