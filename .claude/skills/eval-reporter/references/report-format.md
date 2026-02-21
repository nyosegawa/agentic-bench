# Report Format Specification

## metrics.json Schema

```json
{
  "run_id": "2026-02-21T10:30:00+00:00",
  "model": "google/gemma-3-27b",
  "model_type": "llm",
  "provider": "colab",
  "gpu": "A100-40GB",
  "stages": {
    "smoke": {
      "status": "pass",
      "load_time_seconds": 12.3,
      "notes": null
    },
    "quality": {
      "outputs": ["artifacts/output_001.txt"],
      "notes": "Good instruction following, weak at math"
    },
    "performance": {
      "tokens_per_second": 42.5,
      "latency_p50_ms": 23.4,
      "latency_p99_ms": 89.2,
      "sec_per_image": null,
      "rtf": null,
      "mae": null,
      "rmse": null,
      "num_runs": 5
    }
  },
  "cost_usd": 0.0,
  "duration_seconds": 180
}
```

## Field Definitions

### Top-level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| run_id | string (ISO 8601) | Yes | Timestamp of the run |
| model | string | Yes | Model identifier |
| model_type | string | Yes | One of: llm, vlm, image-gen, tts, stt, audio, embedding, timeseries, code-gen, unknown |
| provider | string | Yes | One of: hf_inference, hf_endpoints, colab, modal, beam, api, local |
| gpu | string | No | GPU used (e.g., "A100-40GB", "T4") |
| stages | object | Yes | Stage results (see below) |
| cost_usd | number | No | Estimated cost in USD |
| duration_seconds | number | No | Total wall-clock time |

### stages.smoke

| Field | Type | Description |
|-------|------|-------------|
| status | "pass" / "fail" / "skip" | Did the model load and produce output? |
| load_time_seconds | number | Time to load the model |
| notes | string | Any notes about the smoke test |

### stages.quality

| Field | Type | Description |
|-------|------|-------------|
| outputs | string[] | Paths to output files (relative to result dir) |
| notes | string | Agent's qualitative assessment |

### stages.performance

Use the metric appropriate for the model type:

| Field | Model Types | Description |
|-------|------------|-------------|
| tokens_per_second | llm, vlm, code-gen | Median tokens/sec |
| latency_p50_ms | llm, vlm | 50th percentile latency |
| latency_p99_ms | llm, vlm | 99th percentile latency |
| sec_per_image | image-gen | Median seconds per image |
| rtf | tts, stt | Real-Time Factor |
| mae | timeseries | Mean Absolute Error |
| rmse | timeseries | Root Mean Square Error |
| num_runs | all | Number of measurement runs |

---

## HTML Report Design Reference

You write the HTML report yourself — no template. Use the following as a **reference** for
consistent visual style, not as a strict requirement.

### CSS Color Palette (Dark Theme)

```css
:root {
  --bg: #0d1117;
  --fg: #e6edf3;
  --muted: #8b949e;
  --border: #30363d;
  --green: #3fb950;
  --red: #f85149;
  --blue: #58a6ff;
  --card: #161b22;
}
```

### Common Components

**Badge:**
```html
<span style="display:inline-block; padding:0.2rem 0.6rem; border-radius:1rem;
  font-size:0.8rem; font-weight:600; background:rgba(63,185,80,0.2); color:#3fb950;">
  PASS
</span>
```

**Card:**
```html
<div style="background:#161b22; border:1px solid #30363d; border-radius:0.5rem;
  padding:1.2rem; margin:1rem 0;">
  Content here
</div>
```

**Audio player with input text:**
```html
<div style="margin-bottom:1.5rem;">
  <div style="color:#8b949e; font-size:0.85rem; margin-bottom:0.3rem;">Input</div>
  <p style="margin-bottom:0.5rem;">"The quick brown fox jumps over the lazy dog."</p>
  <audio controls style="width:100%;"><source src="artifacts/simple.wav"></audio>
</div>
```

**Image with prompt:**
```html
<div style="margin-bottom:1.5rem;">
  <div style="color:#8b949e; font-size:0.85rem;">Prompt</div>
  <p>"A serene landscape at sunset"</p>
  <img src="artifacts/output_001.png" style="max-width:100%; border-radius:0.5rem;">
</div>
```

### Required Sections

1. **Header** — Model name as `<h1>`, linked to HuggingFace. Model type, provider, GPU, date as metadata.
2. **Model Overview** — Brief description, architecture, params, license, notable features.
3. **Execution Environment** — GPU, VRAM, torch version, provider.
4. **Smoke Test** — Status, load time.
5. **Quality Results** — Input/output pairs. Always show what went in and what came out.
6. **Performance** — Metrics table. Compare to published benchmarks when known.
7. **Conclusion** — Honest assessment. Strengths, weaknesses, practical recommendations.
8. **Reproduction** — Link to `workspace/run.py`, provider, cost.
