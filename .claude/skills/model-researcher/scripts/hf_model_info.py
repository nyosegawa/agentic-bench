#!/usr/bin/env python3
"""Fetch model metadata from HuggingFace Hub.

Usage:
    python hf_model_info.py MODEL_ID [--json]

Example:
    python hf_model_info.py google/gemma-3-27b
    python hf_model_info.py black-forest-labs/FLUX.1-dev --json
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard

load_dotenv()


def fetch_model_info(model_id: str) -> dict:
    """Fetch model metadata from HuggingFace Hub."""
    api = HfApi()

    try:
        model_info = api.model_info(model_id)
    except Exception as e:
        return {"error": f"Failed to fetch model info: {e}", "model_id": model_id}

    # Extract parameter count from safetensors metadata
    param_count = None
    if model_info.safetensors:
        params = model_info.safetensors.parameters
        if params:
            param_count = sum(params.values())

    # Extract tags for model type classification
    tags = model_info.tags or []
    pipeline_tag = model_info.pipeline_tag or "unknown"

    # Try to get model card text
    card_text = None
    try:
        card = ModelCard.load(model_id)
        card_text = card.text[:2000] if card.text else None
    except Exception:
        pass

    return {
        "model_id": model_id,
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "library_name": model_info.library_name,
        "param_count": param_count,
        "license": model_info.card_data.get("license") if model_info.card_data else None,
        "downloads": model_info.downloads,
        "likes": model_info.likes,
        "created_at": model_info.created_at.isoformat() if model_info.created_at else None,
        "card_text_preview": card_text,
    }


def classify_model_type(info: dict) -> str:
    """Classify model type from pipeline tag and tags."""
    pipeline = info.get("pipeline_tag", "")
    tags = info.get("tags", [])

    mapping = {
        # LLM
        "text-generation": "llm",
        "text2text-generation": "llm",
        "conversational": "llm",
        # Image generation
        "text-to-image": "image-gen",
        "image-to-image": "image-gen",
        # Audio
        "text-to-speech": "tts",
        "text-to-audio": "audio",
        "automatic-speech-recognition": "stt",
        "audio-to-audio": "audio",
        "audio-classification": "audio",
        # Vision-Language
        "image-text-to-text": "vlm",
        "visual-question-answering": "vlm",
        "document-question-answering": "vlm",
        # Embedding
        "feature-extraction": "embedding",
        "sentence-similarity": "embedding",
        # Time series
        "time-series-forecasting": "timeseries",
        # Video
        "text-to-video": "video-gen",
        "image-to-video": "video-gen",
        # Object detection / segmentation
        "object-detection": "object-detection",
        "image-segmentation": "object-detection",
        "zero-shot-object-detection": "object-detection",
        # 3D
        "image-to-3d": "3d-gen",
        "text-to-3d": "3d-gen",
    }

    if pipeline in mapping:
        return mapping[pipeline]

    # Fallback: check tags (more specific types first to avoid false matches)
    tag_set = {t.lower() for t in tags}
    if tag_set & {"llm", "causal-lm", "text-generation"}:
        return "llm"
    if tag_set & {"code", "code-generation", "coder"}:
        return "code-gen"
    if tag_set & {"video-generation", "text-to-video"}:
        return "video-gen"
    if tag_set & {"diffusion", "stable-diffusion", "text-to-image"}:
        return "image-gen"
    if tag_set & {"tts", "text-to-speech"}:
        return "tts"
    if tag_set & {"object-detection", "yolo", "detr"}:
        return "object-detection"
    if tag_set & {"3d", "mesh-generation", "point-cloud"}:
        return "3d-gen"

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HuggingFace model metadata")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g., google/gemma-3-27b)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    info = fetch_model_info(args.model_id)

    if "error" in info:
        print(json.dumps(info, indent=2) if args.json else f"Error: {info['error']}")
        sys.exit(1)

    model_type = classify_model_type(info)
    info["model_type"] = model_type

    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print(f"Model: {info['model_id']}")
        print(f"Type: {model_type} (pipeline: {info['pipeline_tag']})")
        print(f"Library: {info['library_name']}")
        print(f"Parameters: {info['param_count']:,}" if info["param_count"] else "Parameters: N/A")
        print(f"License: {info['license'] or 'N/A'}")
        print(f"Downloads: {info['downloads']:,}" if info["downloads"] else "Downloads: N/A")


if __name__ == "__main__":
    main()
