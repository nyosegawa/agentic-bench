#!/usr/bin/env python3
"""Generate HTML benchmark report from metrics.json and artifacts.

Usage:
    python report_generator.py --metrics metrics.json --output report.html
    python report_generator.py --metrics metrics.json \
      --artifacts-dir artifacts/ --output report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

SKILL_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = SKILL_DIR / "assets"


def load_metrics(metrics_path: Path) -> dict:
    """Load and return metrics.json."""
    return json.loads(metrics_path.read_text())


def discover_artifacts(artifacts_dir: Path) -> dict[str, list[Path]]:
    """Discover artifacts grouped by type."""
    groups: dict[str, list[Path]] = {"images": [], "audio": [], "text": [], "other": []}

    if not artifacts_dir.exists():
        return groups

    for f in sorted(artifacts_dir.iterdir()):
        if f.suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            groups["images"].append(f)
        elif f.suffix in (".wav", ".mp3", ".ogg", ".flac"):
            groups["audio"].append(f)
        elif f.suffix in (".txt", ".json", ".md"):
            groups["text"].append(f)
        else:
            groups["other"].append(f)

    return groups


def generate_report(
    metrics: dict,
    artifacts: dict[str, list[Path]],
    output_path: Path,
) -> None:
    """Generate HTML report from metrics and artifacts."""
    env = Environment(
        loader=FileSystemLoader(str(ASSETS_DIR)),
        autoescape=True,
    )
    template = env.get_template("report_template.html")

    # Make artifact paths relative to the report
    report_dir = output_path.parent
    relative_artifacts = {}
    for group, files in artifacts.items():
        relative_artifacts[group] = []
        for f in files:
            try:
                rel = f.relative_to(report_dir)
            except ValueError:
                rel = f
            relative_artifacts[group].append({"path": str(rel), "name": f.name})

    # Read text file contents for inline display
    text_contents = []
    for f in artifacts.get("text", []):
        content = f.read_text(errors="replace")[:5000]  # Limit to 5KB
        text_contents.append({"name": f.name, "content": content})

    html = template.render(
        metrics=metrics,
        artifacts=relative_artifacts,
        text_contents=text_contents,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Report generated: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json")
    parser.add_argument("--artifacts-dir", help="Path to artifacts directory")
    parser.add_argument("--output", required=True, help="Output HTML file path")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics))
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path()
    artifacts = discover_artifacts(artifacts_dir)

    generate_report(metrics, artifacts, Path(args.output))


if __name__ == "__main__":
    main()
