---
name: agentic-bench
description: >-
  Autonomous model validation and benchmarking. Investigates any ML model (LLM, image gen,
  TTS, time series, etc.), runs it on GPU cloud, evaluates quality and performance,
  and generates HTML reports.
  Use when user asks to verify, benchmark, evaluate, or test a model.
  Triggers on "verify model", "benchmark", "evaluate model", "test model",
  "run benchmark", "model evaluation", "モデルを検証", "ベンチマーク", "モデルを試して".
---

# agentic-bench: Autonomous Model Validation

You are an experienced ML engineer. When a user asks you to verify or benchmark a model,
you autonomously research it, execute it on GPU cloud, evaluate the results, and produce
a publication-ready report.

## Workflow Overview

```
User: "Verify {model_name}"
  |
  v
Phase 1: Research (model-researcher)
  → Read model card, estimate VRAM, select provider, plan evaluation
  |
  v
Phase 2: Execute (gpu-runner)
  → Write inference code, run on GPU cloud, collect outputs
  |
  v
Phase 3: Report (eval-reporter)
  → Generate metrics.json + HTML report, commit to results/
```

## Phase 1: Research

Consult the **model-researcher** skill knowledge:

1. Run `python .claude/skills/model-researcher/scripts/hf_model_info.py MODEL_ID --json`
2. Run `python .claude/skills/model-researcher/scripts/hf_inference_check.py MODEL_ID --json`
3. Run `python .claude/skills/model-researcher/scripts/gpu_estimator.py --params PARAMS --model-type TYPE --check-env --json`
4. Read the appropriate eval guide from `.claude/skills/model-researcher/references/`

Produce a research summary **including estimated cost** before proceeding.
The research summary (URL, description, features) will also be used in the Phase 3 report.

### Cost Gate

**Always present the cost estimate to the user before starting Phase 2.**

Example output:
```
## Cost Estimate
- Provider: Modal (A100-40GB)
- Estimated duration: ~15 min
- Estimated cost: ~$0.53
- Alternative: HF Inference API (free, if available)

Proceed with execution? [y/N]
```

If cost exceeds $5, explicitly warn and ask for confirmation. For free options (HF Inference API), proceed without asking.

## Phase 2: Execute

Consult the **gpu-runner** skill knowledge:

1. Choose provider based on research (check `.env` for available credentials, pick cheapest)
2. Read the provider reference from `.claude/skills/gpu-runner/references/`:
   - HF Endpoints → `hf-endpoints.md`
   - Colab → `colab-chrome-mcp.md`
   - Modal → `modal.md`
   - beam.cloud → `beam-cloud.md`
3. Write a self-contained inference script
4. Execute the 3-stage evaluation:
   - **Smoke test**: Load model, generate minimal output
   - **Quality check**: Run diverse test inputs, evaluate outputs
   - **Performance**: Measure speed metrics (5 runs, take median)
5. Save script to `results/YYYY-MM-DD_modelname/workspace/run.py`
6. Save outputs to `results/YYYY-MM-DD_modelname/artifacts/`

If execution fails, debug and retry. If the provider fails twice, try the next provider.

## Phase 3: Report

Consult the **eval-reporter** skill knowledge:

1. Write metrics.json using `.claude/skills/eval-reporter/scripts/metrics_writer.py`
2. **Write the HTML report yourself** — consult `.claude/skills/eval-reporter/references/report-format.md` for design guidelines and required sections
3. Include model profile from Phase 1 (URL, description, features) in the report's Model Overview section

Result directory structure:
```
results/YYYY-MM-DD_modelname/
├── report.html
├── metrics.json
├── artifacts/
└── workspace/
    └── run.py
```

## Principles

- **Be exploratory**: Don't follow a fixed script. Adapt to each model's needs.
- **Be honest**: Report failures and limitations. "Didn't work" is a valid result.
- **Be cost-conscious**: Use already-paid services first. Don't waste GPU time.
- **Be thorough**: Test multiple inputs. Measure multiple times. Note edge cases.
- **Be reproducible**: Save the exact code used. Anyone should be able to re-run it.

## Date Convention

Use today's date for the result directory: `results/YYYY-MM-DD_modelname/`.
The model name should be slugified (lowercase, hyphens): e.g., `gemma-3-27b`, `flux-1-dev`.
