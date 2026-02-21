---
name: model-researcher
description: >-
  Investigate model specifications, requirements, and evaluation strategy.
  Use when researching a model before benchmarking: reading HuggingFace model cards,
  estimating VRAM requirements, selecting GPU providers, and determining evaluation approach.
  Triggers on "model research", "investigate model", "model info", "VRAM estimate",
  "which provider", "model card".
---

# Model Researcher

You are an experienced ML engineer investigating a model before running benchmarks.

## Your Goal

Given a model name, produce a structured research summary:
1. Model identity (type, architecture, parameter count, license)
2. Hardware requirements (VRAM, recommended GPU)
3. Recommended provider (cheapest viable option)
4. Evaluation strategy (what to test and how)

## Workflow

### Step 1: Gather Model Information

Run the helper script to pull structured metadata:
```bash
python .claude/skills/model-researcher/scripts/hf_model_info.py MODEL_ID
```

To check if the model is available on HF Inference API (serverless):
```bash
python .claude/skills/model-researcher/scripts/hf_inference_check.py MODEL_ID
```

To search for similar or alternative models:
```bash
python .claude/skills/model-researcher/scripts/hf_model_search.py --task llm --sort downloads --limit 10
python .claude/skills/model-researcher/scripts/hf_model_search.py --search "qwen" --limit 5
```

If the script fails or the model is not on HuggingFace:
- Search the web for the model's official page, paper, or GitHub repo
- Manually gather: architecture, parameter count, input/output modalities, license

### Step 2: Estimate VRAM Requirements

Run the estimator:
```bash
python .claude/skills/model-researcher/scripts/gpu_estimator.py --params PARAM_COUNT [--quant fp16|int8|int4]
```

This outputs recommended GPU and provider. Use this as a starting point; adjust based on:
- Model-specific requirements (e.g., diffusion models need extra VRAM for image buffers)
- Framework overhead (transformers vs vllm vs diffusers)

### Step 3: Determine Model Type and Evaluation Strategy

Classify the model and load the appropriate evaluation guide:

| Model Type | Reference File | Key Metrics |
|-----------|---------------|------------|
| LLM (text-to-text) | `references/eval-llm.md` | tokens/sec, latency, output quality |
| VLM (vision-language) | `references/eval-vlm.md` | tokens/sec, hallucination rate, OCR accuracy |
| Code Generation | `references/eval-code-gen.md` | tokens/sec, pass@1, FIM accuracy |
| Embedding | `references/eval-embedding.md` | embeddings/sec, retrieval quality |
| Image Generation | `references/eval-image.md` | sec/image, visual quality |
| TTS (text-to-speech) | `references/eval-tts.md` | RTF, audio quality |
| Time Series | `references/eval-timeseries.md` | MAE/RMSE, prediction accuracy |

Read the relevant reference file for detailed evaluation guidance.

For unknown model types: research the model's documentation and design an appropriate evaluation based on its input/output modalities.

### Step 4: Select Provider

Priority chain (cheapest first, prefer already-paid services):

1. **Direct API** — For API-hosted models (GPT-4o, Claude, Gemini). No GPU needed.
2. **HF Inference API** — If model is on HF and API-supported. Free with HF Pro.
3. **Colab Pro** — Up to ~30B params. Chrome MCP for notebook control. Already paid.
4. **Modal** — 30B+ or when Colab GPUs unavailable. $30/month free tier.
5. **beam.cloud** — Alternative for dedicated deployments. Existing credit.

### Step 5: Output Research Summary

Structure your findings as:
```
## Research Summary: {model_name}
- **Type**: {model_type}
- **Parameters**: {param_count}
- **Architecture**: {architecture}
- **License**: {license}
- **VRAM Required**: {vram_estimate}
- **Recommended GPU**: {gpu}
- **Recommended Provider**: {provider}
- **Evaluation Strategy**: {brief description}
- **Key Metrics**: {metrics to measure}
```

## Important

- Always check the model card FIRST — it often has sample code and requirements
- When in doubt about VRAM, round up and pick a bigger GPU
- If a model can run on HF Inference API, that's always the cheapest option
- Note any special dependencies or setup requirements for the gpu-runner phase
