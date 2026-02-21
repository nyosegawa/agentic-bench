---
name: gpu-runner
description: >-
  Execute model inference on GPU cloud providers. Handles code generation, deployment,
  execution, and result collection across HF Inference API/Endpoints, Colab, Modal,
  and beam.cloud. Use when running models on GPU, deploying to cloud, executing notebooks,
  or troubleshooting GPU execution failures.
  Triggers on "run on GPU", "execute model", "deploy to modal", "colab notebook",
  "beam deploy", "HF inference", "HF endpoints".
---

# GPU Runner

You are executing model inference on the appropriate GPU cloud provider.

## Your Goal

Given a model, its requirements, and a chosen provider:
1. Write inference code tailored to the model
2. Execute it on the selected provider
3. Collect outputs (text, images, audio, metrics)
4. Handle errors and retry with alternatives if needed

## Provider Selection (if not pre-selected)

Check `.env` for available credentials, then sort by **cheapest hourly cost**:

1. **HF Inference API** — Free with HF Pro. Requires `HF_TOKEN`. Catalog models only.
2. **HF Inference Endpoints** — Any HF model on dedicated GPU. `HF_TOKEN` only. $0.50–2.50/hr.
3. **Colab Pro** — Chrome MCP. No token needed. $9.99/month subscription. Up to ~30B.
4. **Modal** — Requires `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`. $30/month free tier. $0.59–3.95/hr.
5. **beam.cloud** — Requires `BEAM_TOKEN`. Existing credit. $0.54–3.50/hr.

**Token availability check**: If a provider's env vars are not set, skip it.

## Provider-Specific Guides

Before executing, read the relevant provider reference:

| Provider | Reference | When to Use |
|----------|-----------|------------|
| HF Inference API | (inline below) | Model on HF, API-supported, free |
| HF Inference Endpoints | `references/hf-endpoints.md` | Any HF model, cheapest dedicated GPU |
| Colab Pro | `references/colab-chrome-mcp.md` | Up to ~30B, interactive debugging |
| Modal | `references/modal.md` | 30B+, serverless, reliable GPUs |
| beam.cloud | `references/beam-cloud.md` | Dedicated endpoints, existing credit |

### HF Inference API (inline — simple enough)

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.environ["HF_TOKEN"])

# Text generation
response = client.text_generation("Hello, ", model="MODEL_ID", max_new_tokens=100)

# Image generation
image = client.text_to_image("A cat", model="MODEL_ID")
image.save("output.png")
```

## Execution Workflow

### Step 0: Dependency Research (BEFORE writing code)

**The most expensive mistake is building a cloud image 10 times.** Resolve ALL dependencies
before writing the script, not through trial-and-error.

1. **Read the model card install instructions** — if it says `pip install X`, use that exactly
2. **Check for heavy framework dependencies** (nemo-toolkit, fairseq, detectron2, mmdet, etc.):
   - Search PyPI or GitHub for the package's `requirements.txt` / `setup.py`
   - List ALL transitive dependencies upfront
   - Use `--no-deps` to install the framework, then install its dependencies explicitly
3. **Use `uv` instead of `pip`** for faster, more reliable dependency resolution
4. **Pin versions only when the model card specifies them** — otherwise let the resolver decide

**Anti-pattern (never do this):**
Adding one missing package at a time → rebuild → discover next missing package → rebuild.
This wastes 5-10 minutes per cycle. Instead, get the full dependency list right once.

### Step 1: Write Inference Code

Read `references/inference-patterns.md` for code snippets per model type (LLM, VLM,
image-gen, TTS, STT, embedding, timeseries, video-gen, object-detection, 3D, etc.).

Then write a self-contained script:
- Import all dependencies
- Load model (with appropriate dtype/device settings)
- Run inference with test inputs
- Save outputs to files
- Print structured metrics (timing, token counts, etc.)

**Always check the model card first** — it overrides the generic patterns in
inference-patterns.md. Each model may have a unique API, custom pipeline class,
or special dependencies.

Save the script to `results/YYYY-MM-DD_modelname/workspace/run.py`.

### Step 2: Execute

- **HF Inference API**: Run directly in the current environment
- **Colab**: Use Chrome MCP to create/run notebook cells
- **Modal**: Deploy function and call `.remote()`
- **beam.cloud**: Deploy endpoint and call via HTTP

### Step 3: Collect Results

Ensure all outputs are saved to `results/YYYY-MM-DD_modelname/`:
- `artifacts/` — Generated files (images, audio, text outputs)
- `workspace/run.py` — The execution script (for reproducibility)

### Step 4: Handle Failures

Common failure patterns and recovery:

| Error | Recovery |
|-------|---------|
| OOM (CUDA out of memory) | Try quantization (int8/int4), smaller batch, or bigger GPU |
| Colab GPU unavailable | Fall back to Modal |
| Modal timeout | Increase timeout, or use beam.cloud |
| Import error | Install missing dependency in the execution environment |
| Model not found | Verify model ID, check if gated (needs HF token) |

If a provider fails after 2 attempts, try the next provider in priority order.

## Important

- Always use `torch.bfloat16` or `torch.float16` for GPU models (never fp32)
- Set `device_map="auto"` for large models
- Include timing measurements in the execution script
- Save ALL outputs — even errors are valuable for the report
- Load `.env` with `python-dotenv` for API tokens
