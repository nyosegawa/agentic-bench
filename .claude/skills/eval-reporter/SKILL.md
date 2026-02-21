---
name: eval-reporter
description: >-
  Generate HTML reports and structured metrics from model evaluation results.
  Creates publication-ready reports with embedded outputs (images, audio, charts)
  and metrics.json for cross-model comparison.
  Use when generating reports, writing metrics, creating evaluation summaries,
  or formatting benchmark results.
  Triggers on "generate report", "write metrics", "create report", "evaluation summary",
  "benchmark results", "format results".
---

# Eval Reporter

You are generating evaluation reports from model benchmark results.

## Your Goal

Given evaluation outputs and metrics:
1. Write structured `metrics.json` with all measurements
2. Generate a beautiful HTML report with embedded outputs
3. Save everything to `results/YYYY-MM-DD_modelname/`

## Workflow

### Step 1: Collect Data

Gather from the gpu-runner phase:
- Model metadata (name, type, params, provider, GPU)
- Stage results (smoke test pass/fail, quality outputs, performance metrics)
- Generated artifacts (images, audio, text files)
- Timing and cost information

### Step 2: Write metrics.json

Run the metrics writer:
```bash
python .claude/skills/eval-reporter/scripts/metrics_writer.py \
  --output results/YYYY-MM-DD_modelname/metrics.json \
  --model MODEL_ID \
  --model-type MODEL_TYPE \
  --provider PROVIDER \
  --gpu GPU_NAME \
  --json-data '{"stages": {...}}'
```

Or construct the JSON directly following `references/report-format.md`.

### Step 3: Generate HTML Report

Run the report generator:
```bash
python .claude/skills/eval-reporter/scripts/report_generator.py \
  --metrics results/YYYY-MM-DD_modelname/metrics.json \
  --artifacts-dir results/YYYY-MM-DD_modelname/artifacts/ \
  --output results/YYYY-MM-DD_modelname/report.html
```

Or write the HTML directly using the template in `assets/report_template.html`.

### Step 4: Organize Result Directory

```
results/YYYY-MM-DD_modelname/
├── report.html          # Human-readable report (GitHub Pages ready)
├── metrics.json         # Structured data for comparison
├── artifacts/           # Generated outputs (images, audio, text)
│   ├── output_001.png
│   ├── output_002.wav
│   └── ...
└── workspace/           # Reproduction scripts (committed)
    └── run.py           # The inference script used
```

### Step 5: Write Evaluation Commentary

In the HTML report, include your assessment:
- **Smoke test**: Did the model load and produce output?
- **Quality**: Your observations on output quality (be specific and honest)
- **Performance**: How do the numbers compare to similar models?
- **Cost**: Was this cost-effective? What would production use look like?
- **Verdict**: Overall recommendation and notable strengths/weaknesses

## Important

- Consult `references/report-format.md` for the exact metrics.json schema
- Images should be referenced by relative path in HTML (not base64, to keep file size small)
- Audio files: use `<audio controls>` tags in HTML
- Charts: use inline Chart.js for performance visualizations
- Always include the run.py script for reproducibility
- Be honest in assessments — document failures and limitations clearly
