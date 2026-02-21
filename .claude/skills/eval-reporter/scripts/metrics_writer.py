#!/usr/bin/env python3
"""Write structured metrics.json for benchmark results.

Usage:
    python metrics_writer.py --output metrics.json \
      --model MODEL_ID --model-type llm --provider colab --gpu A100-40GB
    python metrics_writer.py --output metrics.json --from-json '{"model": "...", ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Required fields in metrics.json
REQUIRED_FIELDS = {"run_id", "model", "model_type", "provider", "stages"}

VALID_MODEL_TYPES = {
    "llm",
    "vlm",
    "image-gen",
    "tts",
    "stt",
    "audio",
    "embedding",
    "timeseries",
    "code-gen",
    "video-gen",
    "object-detection",
    "3d-gen",
    "unknown",
}

VALID_PROVIDERS = {"hf_inference", "colab", "modal", "beam", "api", "local"}

VALID_STAGE_STATUSES = {"pass", "fail", "skip"}


def validate_metrics(data: dict) -> list[str]:
    """Validate metrics data against the schema. Returns list of errors."""
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "model_type" in data and data["model_type"] not in VALID_MODEL_TYPES:
        errors.append(f"Invalid model_type: {data['model_type']}")

    if "provider" in data and data["provider"] not in VALID_PROVIDERS:
        errors.append(f"Invalid provider: {data['provider']}")

    stages = data.get("stages", {})
    if "smoke" in stages:
        status = stages["smoke"].get("status")
        if status and status not in VALID_STAGE_STATUSES:
            errors.append(f"Invalid smoke test status: {status}")

    return errors


def build_metrics(
    model: str,
    model_type: str,
    provider: str,
    gpu: str | None = None,
    stages: dict | None = None,
    cost_usd: float | None = None,
    duration_seconds: float | None = None,
) -> dict:
    """Build a metrics dict with defaults."""
    return {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_type": model_type,
        "provider": provider,
        "gpu": gpu,
        "stages": stages or {},
        "cost_usd": cost_usd,
        "duration_seconds": duration_seconds,
    }


def write_metrics(data: dict, output_path: Path) -> None:
    """Validate and write metrics to file."""
    errors = validate_metrics(data)
    if errors:
        for error in errors:
            print(f"Validation error: {error}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"Written: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write metrics.json for benchmark results")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--model", help="Model ID")
    parser.add_argument("--model-type", help="Model type (llm, image-gen, tts, etc.)")
    parser.add_argument("--provider", help="Provider (colab, modal, beam, etc.)")
    parser.add_argument("--gpu", help="GPU used")
    parser.add_argument("--from-json", help="Full metrics as JSON string")
    parser.add_argument("--json-data", help="Additional JSON data to merge")
    args = parser.parse_args()

    if args.from_json:
        data = json.loads(args.from_json)
    else:
        if not all([args.model, args.model_type, args.provider]):
            parser.error("--model, --model-type, and --provider are required (or use --from-json)")

        extra = json.loads(args.json_data) if args.json_data else {}
        data = build_metrics(
            model=args.model,
            model_type=args.model_type,
            provider=args.provider,
            gpu=args.gpu,
            stages=extra.get("stages"),
            cost_usd=extra.get("cost_usd"),
            duration_seconds=extra.get("duration_seconds"),
        )

    write_metrics(data, Path(args.output))


if __name__ == "__main__":
    main()
