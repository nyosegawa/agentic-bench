# beam.cloud GPU Execution Guide

## Overview

beam.cloud provides serverless GPU endpoints with Python SDK.
Supports T4, A10G (24GB), RTX4090 (24GB), H100 (80GB).

## Authentication

Environment variable (loaded from `.env`):
```
BEAM_TOKEN=your_auth_token
```

Or run `beam config create` for interactive setup.

## Basic Pattern — Endpoint Decorator

```python
from beam import endpoint, Image

@endpoint(
    name="model-inference",
    gpu="A10G",
    cpu=2,
    memory="16Gi",
    image=Image(
        python_version="python3.11",
        python_packages=["torch", "transformers", "accelerate"],
    ),
)
def predict(context, **inputs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch, time

    model_id = inputs.get("model_id", "google/gemma-3-27b")
    prompt = inputs.get("prompt", "Hello!")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    tok_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    output = model.generate(**tok_inputs, max_new_tokens=128)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    tokens = output.shape[1] - tok_inputs["input_ids"].shape[1]

    return {
        "output": text,
        "tokens_generated": tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
    }
```

## GPU Specification

```python
@endpoint(gpu="T4")        # 16GB
@endpoint(gpu="A10G")      # 24GB
@endpoint(gpu="RTX4090")   # 24GB
@endpoint(gpu="H100")      # 80GB

# Multiple GPUs
@endpoint(gpu="A10G", gpu_count=2)

# GPU priority/fallback
@endpoint(gpu=["H100", "A10G", "T4"])
```

## Deployment and Calling

```bash
# Deploy
beam deploy app.py:predict
```

After deployment, call the endpoint:

```python
import requests
import os

response = requests.post(
    "https://apps.beam.cloud/serve/NAMESPACE/DEPLOYMENT_ID",
    json={"model_id": "google/gemma-3-27b", "prompt": "Hello!"},
    headers={"Authorization": f"Bearer {os.environ['BEAM_TOKEN']}"},
)
result = response.json()
```

## App Pattern (Alternative)

```python
from beam import App, Runtime, Image

app = App(
    name="model-bench",
    runtime=Runtime(
        cpu=2,
        memory="16Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.11",
            python_packages=["torch", "transformers"],
        ),
    ),
)

@app.rest_api()
def handler(request):
    # ... inference logic ...
    return {"result": "..."}
```

## Common Issues

- **Cold start**: First deployment builds the image. Cache dependencies for speed.
- **Timeout**: Default varies; set explicitly for long-running inference.
- **Auth failure**: Verify BEAM_TOKEN in `.env`. Run `beam config create` if needed.
- **GPU unavailable**: Use GPU priority list `gpu=["H100", "A10G"]` for fallback.
- **Package version conflicts**: Pin specific versions in `python_packages`.
