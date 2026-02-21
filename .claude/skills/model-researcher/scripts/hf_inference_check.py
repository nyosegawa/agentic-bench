#!/usr/bin/env python3
"""Check if a model is available on HuggingFace Inference API.

Usage:
    python hf_inference_check.py MODEL_ID
    python hf_inference_check.py MODEL_ID --json
    python hf_inference_check.py MODEL_ID --provider hf-inference

Example:
    python hf_inference_check.py google/gemma-3-27b-it
    python hf_inference_check.py black-forest-labs/FLUX.1-schnell --json
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

load_dotenv()


def check_inference(model_id: str, token: str | None = None) -> dict:
    """Check inference availability and provider mapping for a model."""
    try:
        info = model_info(
            model_id,
            token=token,
            expand=["inference", "inferenceProviderMapping"],
        )
    except GatedRepoError:
        return {
            "model_id": model_id,
            "accessible": False,
            "gated": True,
            "inference": None,
            "providers": {},
            "error": "Gated model — accept license at huggingface.co first",
        }
    except RepositoryNotFoundError:
        return {
            "model_id": model_id,
            "accessible": False,
            "gated": None,
            "inference": None,
            "providers": {},
            "error": "Model not found",
        }

    providers = {}
    if info.inference_provider_mapping:
        for provider_name, mapping in info.inference_provider_mapping.items():
            providers[provider_name] = {
                "status": mapping.status,
                "task": mapping.task,
                "provider_id": mapping.provider_id,
            }

    return {
        "model_id": model_id,
        "accessible": True,
        "gated": info.gated if hasattr(info, "gated") else None,
        "inference": info.inference,  # "warm" or None
        "pipeline_tag": info.pipeline_tag,
        "providers": providers,
    }


def is_serverless_available(model_id: str, provider: str = "hf-inference") -> bool:
    """Quick check: is model available on a specific provider?"""
    result = check_inference(model_id)
    if not result["accessible"] or not result["providers"]:
        return False
    mapping = result["providers"].get(provider)
    return mapping is not None and mapping.get("status") == "live"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check HF Inference API availability")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--provider", help="Check specific provider (e.g., hf-inference)")
    args = parser.parse_args()

    result = check_inference(args.model_id)

    if args.provider:
        available = is_serverless_available(args.model_id, args.provider)
        if args.json:
            print(json.dumps({"available": available, "provider": args.provider}))
        else:
            status = "available" if available else "NOT available"
            print(f"{args.model_id} is {status} on {args.provider}")
        sys.exit(0 if available else 1)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

        inference = result["inference"] or "not available"
        print(f"Model: {result['model_id']}")
        print(f"Task: {result['pipeline_tag']}")
        print(f"Inference: {inference}")
        print(f"Gated: {result['gated']}")

        if result["providers"]:
            print(f"\nProviders ({len(result['providers'])}):")
            for name, info in result["providers"].items():
                print(f"  {name}: {info['status']} ({info['task']})")
        else:
            print("\nNo inference providers available — must run locally or on GPU cloud")


if __name__ == "__main__":
    main()
