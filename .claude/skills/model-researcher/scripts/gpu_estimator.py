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
import os
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
    {"name": "T4", "vram_gb": 16, "providers": ["hf_endpoints", "colab", "modal", "beam", "vast"]},
    {"name": "L4", "vram_gb": 24, "providers": ["hf_endpoints", "colab", "modal", "runpod"]},
    {"name": "A10G", "vram_gb": 24, "providers": ["hf_endpoints", "beam"]},
    {"name": "RTX4090", "vram_gb": 24, "providers": ["vast", "runpod"]},
    {"name": "A100-40GB", "vram_gb": 40, "providers": ["colab", "modal", "beam", "vast"]},
    {"name": "L40S", "vram_gb": 48, "providers": ["hf_endpoints"]},
    {
        "name": "A100-80GB",
        "vram_gb": 80,
        "providers": ["hf_endpoints", "modal", "beam", "vast", "runpod"],
    },
    {"name": "H100", "vram_gb": 80, "providers": ["modal", "beam", "vast", "runpod"]},
]

# Provider env var requirements — used to check which providers are available
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "hf_inference": ["HF_TOKEN"],
    "hf_endpoints": ["HF_TOKEN"],
    "colab": [],  # Chrome MCP, no token needed
    "modal": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
    "beam": ["BEAM_TOKEN"],
    "vast": ["VAST_API_KEY"],
    "runpod": ["RUNPOD_API_KEY"],
}

# Provider cost priority (lower = cheaper, used as tiebreaker)
PROVIDER_COST_RANK = {
    "hf_inference": 0,
    "hf_endpoints": 1,
    "colab": 2,
    "modal": 3,
    "beam": 4,
    "vast": 5,
    "runpod": 6,
}

# GPU pricing per hour (USD) by provider — updated 2026-02
GPU_PRICING: dict[str, dict[str, float]] = {
    "hf_endpoints": {
        "T4": 0.50,
        "L4": 0.80,
        "A10G": 1.00,
        "L40S": 1.80,
        "A100-80GB": 2.50,
    },
    "colab": {"T4": 1.17, "L4": 2.50, "A100-40GB": 6.20},
    "modal": {
        "T4": 0.59,
        "L4": 0.80,
        "A100-40GB": 2.10,
        "A100-80GB": 2.50,
        "H100": 3.95,
    },
    "beam": {"T4": 0.54, "A100-40GB": 2.75, "H100": 3.50},
    "vast": {
        "T4": 0.10,
        "RTX4090": 0.15,
        "A100-40GB": 0.75,
        "A100-80GB": 1.10,
        "H100": 2.00,
    },
    "runpod": {
        "L4": 0.44,
        "RTX4090": 0.34,
        "A100-80GB": 1.19,
        "H100": 2.69,
    },
}

# Monthly subscriptions / free credits
# prepaid=True means the user already pays a subscription or has free credits,
# so the effective additional cost for a short benchmark is ~$0.
SUBSCRIPTION_COSTS: dict[str, dict] = {
    "hf_inference": {
        "monthly_usd": 9.0,
        "prepaid": True,
        "note": "HF Pro — included inference credits",
    },
    "hf_endpoints": {
        "monthly_usd": 0.0,
        "prepaid": False,
        "note": "Pay-as-you-go, HF_TOKEN only",
    },
    "colab": {
        "monthly_usd": 9.99,
        "prepaid": True,
        "note": "Colab Pro — included compute units, effectively free",
    },
    "modal": {
        "monthly_usd": 0.0,
        "prepaid": True,
        "note": "$30/mo free compute credits",
    },
    "beam": {
        "monthly_usd": 0.0,
        "prepaid": False,
        "note": "Pay-as-you-go",
    },
    "vast": {
        "monthly_usd": 0.0,
        "prepaid": False,
        "note": "Marketplace pricing, prepaid credits",
    },
    "runpod": {
        "monthly_usd": 0.0,
        "prepaid": False,
        "note": "Pay-as-you-go, per-second billing",
    },
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


def check_available_providers() -> dict[str, bool]:
    """Check which providers have required env vars set.

    Returns dict of provider -> available (True/False).
    Loads .env from repo root if python-dotenv is available.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    available = {}
    for provider, env_vars in PROVIDER_ENV_VARS.items():
        if not env_vars:
            # No token needed (e.g., colab via Chrome MCP)
            available[provider] = True
        else:
            available[provider] = all(os.environ.get(v) for v in env_vars)
    return available


def estimate_vram(param_count: int, quant: str = "fp16") -> float:
    """Estimate VRAM in GB. Adds ~20% overhead for activations/framework."""
    bytes_per_param = BYTES_PER_PARAM.get(quant, 2.0)
    model_size_gb = (param_count * bytes_per_param) / (1024**3)
    # Add overhead for KV cache, activations, framework
    return model_size_gb * 1.2


def recommend_gpu(
    vram_required_gb: float,
    filter_available: bool = False,
) -> list[dict]:
    """Return GPUs sorted by hourly cost (cheapest first), then provider rank.

    Args:
        vram_required_gb: Minimum VRAM in GB.
        filter_available: If True, only include providers whose env vars are set.
    """
    if filter_available:
        available = check_available_providers()
    else:
        available = None

    suitable = []
    for gpu in GPUS:
        if gpu["vram_gb"] >= vram_required_gb:
            for provider in gpu["providers"]:
                # Skip providers without required tokens
                if available is not None and not available.get(provider, False):
                    continue

                hourly = GPU_PRICING.get(provider, {}).get(gpu["name"])
                sub = SUBSCRIPTION_COSTS.get(provider, {})
                is_prepaid = sub.get("prepaid", False)

                rec: dict = {
                    "gpu": gpu["name"],
                    "vram_gb": gpu["vram_gb"],
                    "provider": provider,
                    "cost_rank": PROVIDER_COST_RANK.get(provider, 99),
                    "hourly_rate_usd": hourly,
                    "prepaid": is_prepaid,
                    "available": available.get(provider, True) if available else None,
                }

                # Add subscription / free tier info
                note = sub.get("note", "")
                if note:
                    rec["subscription_note"] = note

                suitable.append(rec)

    # Sort priority:
    #   1. Prepaid providers first (subscription/free credits → effective cost ~$0)
    #   2. Then by hourly rate (cheapest first)
    #   3. Then by provider rank as tiebreaker
    #   4. None rates (unknown pricing) go last
    suitable.sort(
        key=lambda x: (
            0 if x["prepaid"] else 1,
            x["hourly_rate_usd"] if x["hourly_rate_usd"] is not None else 999,
            x["cost_rank"],
        )
    )
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


def estimate(
    param_count: int,
    quant: str = "fp16",
    model_type: str = "unknown",
    filter_available: bool = False,
) -> dict:
    """Full estimation: VRAM + GPU + provider + cost recommendations.

    Args:
        param_count: Number of model parameters.
        quant: Quantization level.
        model_type: Model type for duration estimate.
        filter_available: If True, only show providers with env vars set.
    """
    vram_gb = estimate_vram(param_count, quant)
    recommendations = recommend_gpu(vram_gb, filter_available=filter_available)

    # Check provider availability for the summary
    provider_status = check_available_providers() if filter_available else None

    # Check if small enough for HF Inference API (rough heuristic: <15B fp16)
    hf_viable = param_count < 15_000_000_000 and quant in ("fp16", "bf16")

    # If filtering, also check if HF_TOKEN is available for hf_inference
    if filter_available and provider_status:
        hf_viable = hf_viable and provider_status.get("hf_inference", False)

    # Add cost estimates to each recommendation
    for rec in recommendations:
        cost = estimate_cost(rec["gpu"], rec["provider"], model_type)
        rec["estimated_cost_usd"] = cost["estimated_cost_usd"]
        # hourly_rate_usd already set by recommend_gpu, keep estimate_cost note if present
        if cost.get("note"):
            rec["note"] = cost["note"]

    result = {
        "param_count": param_count,
        "quantization": quant,
        "estimated_vram_gb": round(vram_gb, 1),
        "hf_inference_viable": hf_viable,
        "model_type": model_type,
        "recommendations": recommendations[:5],  # Top 5
    }

    if provider_status is not None:
        result["provider_availability"] = provider_status

    if hf_viable:
        result["recommendations"].insert(
            0,
            {
                "gpu": "cloud",
                "vram_gb": 0,
                "provider": "hf_inference",
                "cost_rank": 0,
                "estimated_cost_usd": 0.0,
                "hourly_rate_usd": 0.0,
                "subscription_note": "HF Pro — included inference credits",
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
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check .env for provider tokens, filter unavailable providers",
    )
    args = parser.parse_args()

    try:
        param_count = parse_param_count(args.params)
    except ValueError:
        print(f"Error: Cannot parse param count: {args.params}", file=sys.stderr)
        sys.exit(1)

    result = estimate(param_count, args.quant, args.model_type, filter_available=args.check_env)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Parameters: {param_count:,}")
        print(f"Quantization: {args.quant}")
        print(f"Model type: {args.model_type}")
        print(f"Estimated VRAM: {result['estimated_vram_gb']} GB")
        print()

        # Show provider availability if --check-env
        if "provider_availability" in result:
            print("Provider availability:")
            for provider, available in result["provider_availability"].items():
                status = "✓" if available else "✗"
                env_vars = PROVIDER_ENV_VARS.get(provider, [])
                env_str = ", ".join(env_vars) if env_vars else "(no token needed)"
                sub = SUBSCRIPTION_COSTS.get(provider, {})
                note = f" — {sub['note']}" if sub.get("note") else ""
                print(f"  {status} {provider} [{env_str}]{note}")
            print()

        if result["hf_inference_viable"]:
            print("  ★ HF Inference API is viable (recommended, free with HF Pro)")
        print("Recommended GPUs (prepaid first, then cheapest):")
        for rec in result["recommendations"]:
            if rec["provider"] == "hf_inference":
                continue
            cost = rec.get("estimated_cost_usd")
            cost_str = f"~${cost:.2f}" if cost is not None else "N/A"
            rate = rec.get("hourly_rate_usd")
            rate_str = f"${rate:.2f}/hr" if rate is not None else "?"
            prepaid_tag = " [prepaid]" if rec.get("prepaid") else ""
            sub_note = f"  ({rec['subscription_note']})" if rec.get("subscription_note") else ""
            print(
                f"  {rec['gpu']} ({rec['vram_gb']}GB) on {rec['provider']}"
                f"  — {rate_str}, est. {cost_str}{prepaid_tag}{sub_note}"
            )


if __name__ == "__main__":
    main()
