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


def estimate(param_count: int, quant: str = "fp16") -> dict:
    """Full estimation: VRAM + GPU + provider recommendations."""
    vram_gb = estimate_vram(param_count, quant)
    recommendations = recommend_gpu(vram_gb)

    # Check if small enough for HF Inference API (rough heuristic: <15B fp16)
    hf_viable = param_count < 15_000_000_000 and quant in ("fp16", "bf16")

    result = {
        "param_count": param_count,
        "quantization": quant,
        "estimated_vram_gb": round(vram_gb, 1),
        "hf_inference_viable": hf_viable,
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
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        param_count = parse_param_count(args.params)
    except ValueError:
        print(f"Error: Cannot parse param count: {args.params}", file=sys.stderr)
        sys.exit(1)

    result = estimate(param_count, args.quant)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Parameters: {param_count:,}")
        print(f"Quantization: {args.quant}")
        print(f"Estimated VRAM: {result['estimated_vram_gb']} GB")
        print()
        if result["hf_inference_viable"]:
            print("  HF Inference API is viable (recommended, free with HF Pro)")
        print("Recommended GPUs:")
        for rec in result["recommendations"]:
            if rec["provider"] == "hf_inference":
                continue
            print(f"  {rec['gpu']} ({rec['vram_gb']}GB) on {rec['provider']}")


if __name__ == "__main__":
    main()
