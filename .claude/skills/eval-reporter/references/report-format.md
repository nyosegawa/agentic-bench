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
| provider | string | Yes | One of: hf_inference, colab, modal, beam, api, local |
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

## HTML Report Structure

1. **Header**: Model name, type, date, provider/GPU
2. **Summary**: Pass/fail badge, key metrics, cost
3. **Smoke Test**: Load time, status
4. **Quality**: Embedded outputs (images, audio players, text), agent commentary
5. **Performance**: Metrics table, Chart.js visualization
6. **Reproduction**: Link to workspace/run.py
7. **Footer**: Run metadata
