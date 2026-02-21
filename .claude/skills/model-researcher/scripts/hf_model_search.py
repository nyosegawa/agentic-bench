#!/usr/bin/env python3
"""Search and list models from HuggingFace Hub.

Usage:
    python hf_model_search.py --task text-generation --limit 10
    python hf_model_search.py --task text-to-image --inference-only
    python hf_model_search.py --search "llama" --sort downloads
    python hf_model_search.py --trending --limit 20

Example:
    python hf_model_search.py --task text-generation --sort downloads --limit 5
    python hf_model_search.py --task text-to-image --inference-only --json
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

VALID_SORTS = ["downloads", "likes", "trendingScore", "createdAt", "lastModified"]

TASK_ALIASES = {
    "llm": "text-generation",
    "image-gen": "text-to-image",
    "tts": "text-to-speech",
    "stt": "automatic-speech-recognition",
    "vlm": "image-text-to-text",
    "embedding": "feature-extraction",
    "code": "text-generation",
}


def search_models(
    task: str | None = None,
    search: str | None = None,
    sort: str = "downloads",
    limit: int = 10,
    inference_only: bool = False,
    library: str | None = None,
    author: str | None = None,
    non_gated: bool = False,
) -> list[dict]:
    """Search HuggingFace Hub for models."""
    api = HfApi()

    # Resolve task aliases
    if task and task in TASK_ALIASES:
        task = TASK_ALIASES[task]

    kwargs: dict = {"sort": sort, "limit": limit}
    if task:
        kwargs["pipeline_tag"] = task
    if search:
        kwargs["search"] = search
    if inference_only:
        kwargs["inference"] = "warm"
    if library:
        kwargs["library"] = library
    if author:
        kwargs["author"] = author
    if non_gated:
        kwargs["gated"] = False

    models = list(api.list_models(**kwargs))

    results = []
    for m in models:
        results.append(
            {
                "id": m.id,
                "pipeline_tag": m.pipeline_tag,
                "library": m.library_name,
                "downloads": m.downloads,
                "likes": m.likes,
                "gated": m.gated if hasattr(m, "gated") else None,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search HuggingFace models")
    parser.add_argument("--task", help="Pipeline task (or alias: llm, image-gen, tts, stt, vlm)")
    parser.add_argument("--search", help="Search query keyword")
    parser.add_argument(
        "--sort", default="downloads", choices=VALID_SORTS, help="Sort order (default: downloads)"
    )
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--inference-only", action="store_true", help="Only models with API")
    parser.add_argument("--library", help="Filter by library (transformers, diffusers, etc.)")
    parser.add_argument("--author", help="Filter by author/org")
    parser.add_argument("--non-gated", action="store_true", help="Exclude gated models")
    parser.add_argument("--trending", action="store_true", help="Show trending models")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.trending:
        args.sort = "trendingScore"

    if not args.task and not args.search and not args.trending:
        parser.error("Provide --task, --search, or --trending")

    results = search_models(
        task=args.task,
        search=args.search,
        sort=args.sort,
        limit=args.limit,
        inference_only=args.inference_only,
        library=args.library,
        author=args.author,
        non_gated=args.non_gated,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        if not results:
            print("No models found.")
            sys.exit(0)
        for i, m in enumerate(results, 1):
            dl = f"{m['downloads']:,}" if m["downloads"] else "N/A"
            lib = m["library"] or "?"
            gated = " [GATED]" if m["gated"] else ""
            print(f"{i:3d}. {m['id']}")
            print(f"     {m['pipeline_tag'] or '?'} | {lib} | {dl} downloads{gated}")


if __name__ == "__main__":
    main()
