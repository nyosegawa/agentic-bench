#!/usr/bin/env python3
"""Estimate VRAM requirements and recommend GPU/provider.

Usage:
    python gpu_estimator.py --params 27000000000
    python gpu_estimator.py --params 7B --quant int8
    python gpu_estimator.py --params 70B --quant int4 --json

Param count accepts raw numbers or shorthand (7B, 13B, 70B).
"""

from __future__ import annotations

import argparse
import json
import sys


def parse_param_count(value: str) -> int:
    """Parse parameter count from string like '7B', '13B', '70B', or raw number."""
    value = value.strip().upper()
    multipliers = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000}
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * mult)
    return int(value)


# Bytes per parameter for each quantization level
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Available GPUs sorted by VRAM
GPUS = [
    {"name": "T4", "vram_gb": 16, "providers": ["colab"]},
    {"name": "L4", "vram_gb": 24, "providers": ["colab", "modal"]},
    {"name": "A100-40GB", "vram_gb": 40, "providers": ["colab", "modal", "beam"]},
    {"name": "A100-80GB", "vram_gb": 80, "providers": ["modal", "beam"]},
    {"name": "H100", "vram_gb": 80, "providers": ["modal", "beam"]},
]

# Provider priority (lower = preferred)
PROVIDER_PRIORITY = {
    "hf_inference": 0,
    "colab": 1,
    "modal": 2,
    "beam": 3,
}

# GPU pricing per hour (USD) by provider — updated 2025-02
GPU_PRICING: dict[str, dict[str, float]] = {
    "colab": {"T4": 1.17, "L4": 2.50, "A100-40GB": 6.20},
    "modal": {"T4": 0.59, "L4": 0.80, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95},
    "beam": {"T4": 0.54, "A100-40GB": 2.75, "H100": 3.50},
}

# Flat monthly subscriptions (included in cost context)
SUBSCRIPTION_COSTS: dict[str, dict[str, float]] = {
    "hf_inference": {"monthly_usd": 9.0, "note": "HF Pro — included inference credits"},
    "colab": {"monthly_usd": 9.99, "note": "Colab Pro — included compute units"},
}

# Estimated benchmark duration in minutes by model type
BENCH_DURATION_MINUTES: dict[str, float] = {
    "llm": 15.0,
    "vlm": 15.0,
    "code-gen": 15.0,
    "embedding": 5.0,
    "image-gen": 20.0,
    "tts": 10.0,
    "stt": 10.0,
    "audio": 10.0,
    "timeseries": 10.0,
    "video-gen": 30.0,
    "object-detection": 10.0,
    "3d-gen": 20.0,
    "unknown": 15.0,
}


def estimate_vram(param_count: int, quant: str = "fp16") -> float:
    """Estimate VRAM in GB. Adds ~20% overhead for activations/framework."""
    bytes_per_param = BYTES_PER_PARAM.get(quant, 2.0)
    model_size_gb = (param_count * bytes_per_param) / (1024**3)
    # Add overhead for KV cache, activations, framework
    return model_size_gb * 1.2


def recommend_gpu(vram_required_gb: float) -> list[dict]:
    """Return list of suitable GPUs, sorted by provider priority."""
    suitable = []
    for gpu in GPUS:
        if gpu["vram_gb"] >= vram_required_gb:
            for provider in gpu["providers"]:
                suitable.append(
                    {
                        "gpu": gpu["name"],
                        "vram_gb": gpu["vram_gb"],
                        "provider": provider,
                        "priority": PROVIDER_PRIORITY[provider],
                    }
                )
    suitable.sort(key=lambda x: (x["priority"], x["vram_gb"]))
    return suitable


def estimate_cost(
    gpu: str,
    provider: str,
    model_type: str = "unknown",
    duration_minutes: float | None = None,
) -> dict:
    """Estimate cost for a benchmark run.

    Returns cost breakdown: per-hour rate, estimated duration, total cost.
    """
    if duration_minutes is None:
        duration_minutes = BENCH_DURATION_MINUTES.get(model_type, 15.0)

    duration_hours = duration_minutes / 60.0

    provider_pricing = GPU_PRICING.get(provider, {})
    hourly_rate = provider_pricing.get(gpu)

    if provider == "hf_inference":
        return {
            "provider": provider,
            "gpu": "cloud",
            "hourly_rate_usd": 0.0,
            "estimated_duration_min": duration_minutes,
            "estimated_cost_usd": 0.0,
            "note": "Free with HF Pro ($9/mo subscription)",
        }

    if hourly_rate is None:
        return {
            "provider": provider,
            "gpu": gpu,
            "hourly_rate_usd": None,
            "estimated_duration_min": duration_minutes,
            "estimated_cost_usd": None,
            "note": f"No pricing data for {gpu} on {provider}",
        }

    estimated_cost = round(hourly_rate * duration_hours, 2)
    return {
        "provider": provider,
        "gpu": gpu,
        "hourly_rate_usd": hourly_rate,
        "estimated_duration_min": duration_minutes,
        "estimated_cost_usd": estimated_cost,
    }


def estimate(param_count: int, quant: str = "fp16", model_type: str = "unknown") -> dict:
    """Full estimation: VRAM + GPU + provider + cost recommendations."""
    vram_gb = estimate_vram(param_count, quant)
    recommendations = recommend_gpu(vram_gb)

    # Check if small enough for HF Inference API (rough heuristic: <15B fp16)
    hf_viable = param_count < 15_000_000_000 and quant in ("fp16", "bf16")

    # Add cost estimates to each recommendation
    for rec in recommendations:
        cost = estimate_cost(rec["gpu"], rec["provider"], model_type)
        rec["estimated_cost_usd"] = cost["estimated_cost_usd"]
        rec["hourly_rate_usd"] = cost["hourly_rate_usd"]

    result = {
        "param_count": param_count,
        "quantization": quant,
        "estimated_vram_gb": round(vram_gb, 1),
        "hf_inference_viable": hf_viable,
        "model_type": model_type,
        "recommendations": recommendations[:5],  # Top 5
    }

    if hf_viable:
        result["recommendations"].insert(
            0,
            {
                "gpu": "cloud",
                "vram_gb": 0,
                "provider": "hf_inference",
                "priority": 0,
                "estimated_cost_usd": 0.0,
                "hourly_rate_usd": 0.0,
            },
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate VRAM and recommend GPU/provider")
    parser.add_argument(
        "--params", required=True, help="Parameter count (e.g., 7B, 13B, 70B, or raw number)"
    )
    parser.add_argument(
        "--quant",
        default="fp16",
        choices=list(BYTES_PER_PARAM.keys()),
        help="Quantization level (default: fp16)",
    )
    parser.add_argument(
        "--model-type",
        default="unknown",
        help="Model type for duration estimate (llm, vlm, image-gen, tts, etc.)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        param_count = parse_param_count(args.params)
    except ValueError:
        print(f"Error: Cannot parse param count: {args.params}", file=sys.stderr)
        sys.exit(1)

    result = estimate(param_count, args.quant, args.model_type)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Parameters: {param_count:,}")
        print(f"Quantization: {args.quant}")
        print(f"Model type: {args.model_type}")
        print(f"Estimated VRAM: {result['estimated_vram_gb']} GB")
        print()
        if result["hf_inference_viable"]:
            print("  HF Inference API is viable (recommended, free with HF Pro)")
        print("Recommended GPUs (with estimated cost):")
        for rec in result["recommendations"]:
            if rec["provider"] == "hf_inference":
                continue
            cost = rec.get("estimated_cost_usd")
            cost_str = f"~${cost:.2f}" if cost is not None else "N/A"
            rate = rec.get("hourly_rate_usd")
            rate_str = f"${rate:.2f}/hr" if rate is not None else "?"
            print(
                f"  {rec['gpu']} ({rec['vram_gb']}GB) on {rec['provider']}"
                f"  — {rate_str}, est. {cost_str}"
            )


if __name__ == "__main__":
    main()
