# Modal GPU Execution Guide

## Overview

Modal provides serverless GPU compute with Python SDK. Best developer experience.
$30/month free tier. Supports T4, L4, A100 (40/80GB), H100, H200.

## Authentication

Environment variables (loaded from `.env`):
```
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret
```

Or run `modal token new` for interactive setup.

## Basic Pattern

```python
import modal

app = modal.App("agentic-bench-run")

# Define a GPU function
@app.function(
    gpu="A100",
    image=modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "transformers", "accelerate"),
    timeout=600,
)
def run_inference(model_id: str, prompt: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch, time

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=128)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    tokens = output.shape[1] - inputs["input_ids"].shape[1]

    return {
        "output": text,
        "tokens_generated": tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
    }

@app.local_entrypoint()
def main():
    result = run_inference.remote("google/gemma-3-27b", "Hello, world!")
    print(result)
```

## GPU Specification

```python
@app.function(gpu="T4")          # 16GB
@app.function(gpu="L4")          # 24GB
@app.function(gpu="A100")        # 40GB (default A100)
@app.function(gpu="A100-40GB")   # 40GB explicit
@app.function(gpu="A100-80GB")   # 80GB
@app.function(gpu="H100")        # 80GB
@app.function(gpu="A100:2")      # 2x A100 for multi-GPU
```

## Running

```bash
# Run locally (executes @app.local_entrypoint)
modal run script.py

# Deploy as persistent service
modal deploy script.py
```

## Image with HuggingFace Models

For faster cold starts, cache the model in the image:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub")
    .env({"HF_TOKEN": modal.Secret.from_name("hf-token")})
)
```

## Returning Files

```python
@app.function(gpu="A100")
def generate_image(prompt: str) -> bytes:
    # ... generate image ...
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
```

## Common Issues

- **Cold start**: First run takes 1-5 min for image build. Subsequent runs are fast.
- **Timeout**: Default 300s. Set `timeout=600` or higher for large models.
- **OOM**: Increase GPU size or use quantization in the function body.
- **Auth failure**: Verify MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in `.env`.
