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
2. Write a beautiful HTML report **directly** (no template — you design the page)
3. Save everything to `results/YYYY-MM-DD_modelname/`

## Workflow

### Step 1: Collect Data

Gather from the previous phases:
- **Model profile** from research phase: name, URL, description, architecture, params, license, notable features
- **Stage results**: smoke test pass/fail, quality outputs, performance metrics
- **Generated artifacts**: images, audio, text files in `artifacts/`
- **Timing and cost** information
- **Device info**: GPU, VRAM, framework versions

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

### Step 3: Write HTML Report

**Write the HTML yourself.** Do not use a template. You are an ML engineer writing a report
for a technical audience. Design the page to be informative, honest, and beautiful.

Consult `references/report-format.md` for design guidelines (CSS palette, component examples).

**Required sections:**

1. **Header** — Model name (linked to HuggingFace page), type badge, date, provider/GPU
2. **Model Overview** — Architecture, parameter count, license, key features, 1-2 sentence description from model card
3. **Execution Environment** — GPU, VRAM, framework versions, provider
4. **Smoke Test** — Pass/fail, load time, any issues
5. **Quality Results** — Test inputs paired with outputs:
   - TTS: show input text above each `<audio>` player
   - Image gen: show prompt above each `<img>`
   - LLM: show prompt and response together
   - Always show what went IN and what came OUT
6. **Performance** — Key metrics table, comparison to published benchmarks if known
7. **Conclusion** — Your honest assessment: strengths, weaknesses, comparison to claimed performance, practical recommendations
8. **Reproduction** — Link to `workspace/run.py`, note the provider and cost

**Design principles:**
- Dark theme preferred (see references/report-format.md for CSS palette)
- Responsive layout, readable at any width
- Use semantic HTML — `<table>`, `<audio>`, `<img>`, not div soup
- Reference artifacts by relative path (not base64)

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

### Step 5: Update Index Page

Regenerate the top-level index.html (report listing for GitHub Pages):
```bash
python .claude/skills/eval-reporter/scripts/generate_index.py
```

## Important

- Consult `references/report-format.md` for the metrics.json schema and CSS design reference
- Images should be referenced by relative path in HTML (not base64, to keep file size small)
- Audio files: use `<audio controls>` tags in HTML
- Always include the run.py script for reproducibility
- Be honest in assessments — document failures and limitations clearly
- The report should be self-contained and understandable without reading metrics.json
