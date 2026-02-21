#!/usr/bin/env python3
"""Generate index.html listing all benchmark reports.

Scans results/*/metrics.json and creates a landing page linking to each report.

Usage:
    python generate_index.py
    python generate_index.py --output /path/to/index.html
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_all_metrics() -> list[dict]:
    """Load all metrics.json files from results/."""
    entries = []
    if not RESULTS_DIR.exists():
        return entries

    for metrics_path in sorted(RESULTS_DIR.glob("*/metrics.json"), reverse=True):
        try:
            data = json.loads(metrics_path.read_text())
            data["_dir"] = metrics_path.parent.name
            data["_has_report"] = (metrics_path.parent / "report.html").exists()
            entries.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return entries


def model_type_badge(model_type: str) -> str:
    """Generate badge HTML for model type."""
    colors = {
        "llm": "#58a6ff",
        "vlm": "#d2a8ff",
        "tts": "#3fb950",
        "stt": "#3fb950",
        "image-gen": "#f778ba",
        "code-gen": "#58a6ff",
        "embedding": "#8b949e",
        "audio": "#3fb950",
        "video-gen": "#f778ba",
        "object-detection": "#d29922",
        "3d-gen": "#f778ba",
        "timeseries": "#d29922",
    }
    color = colors.get(model_type, "#8b949e")
    return (
        f'<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:1rem;'
        f'font-size:0.75rem;font-weight:600;background:{color}22;color:{color};">'
        f"{model_type}</span>"
    )


def smoke_badge(status: str) -> str:
    """Generate pass/fail badge."""
    if status == "pass":
        return (
            '<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:1rem;'
            "font-size:0.75rem;font-weight:600;background:rgba(63,185,80,0.2);"
            'color:#3fb950;">PASS</span>'
        )
    return (
        '<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:1rem;'
        "font-size:0.75rem;font-weight:600;background:rgba(248,81,73,0.2);"
        'color:#f85149;">FAIL</span>'
    )


def key_metric(entry: dict) -> str:
    """Extract the key performance metric for display."""
    perf = entry.get("stages", {}).get("performance", {})
    if perf.get("tokens_per_second"):
        return f"{perf['tokens_per_second']:.1f} tok/s"
    if perf.get("rtf"):
        return f"RTF {perf['rtf']:.2f}"
    if perf.get("sec_per_image"):
        return f"{perf['sec_per_image']:.1f}s/img"
    if perf.get("mae"):
        return f"MAE {perf['mae']:.4f}"
    return "—"


def generate_html(entries: list[dict]) -> str:
    """Generate the index HTML."""
    rows = []
    for e in entries:
        date = e.get("run_id", "")[:10]
        model = e.get("model", "unknown")
        model_type = e.get("model_type", "unknown")
        provider = e.get("provider", "—")
        gpu = e.get("gpu", "—")
        smoke = e.get("stages", {}).get("smoke", {}).get("status", "—")
        cost = e.get("cost_usd")
        cost_str = f"${cost:.2f}" if cost is not None else "—"
        report_link = (
            f'<a href="results/{e["_dir"]}/report.html" style="color:#58a6ff;">{model}</a>'
            if e.get("_has_report")
            else model
        )

        rows.append(
            f"<tr>"
            f"<td>{date}</td>"
            f"<td>{report_link}</td>"
            f"<td>{model_type_badge(model_type)}</td>"
            f"<td>{smoke_badge(smoke)}</td>"
            f"<td>{key_metric(e)}</td>"
            f"<td>{provider} / {gpu}</td>"
            f"<td>{cost_str}</td>"
            f"</tr>"
        )

    rows_html = "\n      ".join(rows) if rows else "<tr><td colspan='7'>No results yet</td></tr>"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>agentic-bench — Model Benchmark Reports</title>
<style>
  :root {{ --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --border: #30363d;
    --card: #161b22; --blue: #58a6ff; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--fg); line-height: 1.6; padding: 2rem;
    max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }}
  a {{ color: var(--blue); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--border); border-radius: 0.5rem; overflow: hidden; }}
  th {{ text-align: left; padding: 0.6rem 1rem; color: var(--muted); font-weight: 500;
    font-size: 0.8rem; text-transform: uppercase; border-bottom: 1px solid var(--border);
    background: var(--bg); }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(88,166,255,0.04); }}
  footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.8rem; }}
</style>
</head>
<body>

<h1>agentic-bench</h1>
<p class="subtitle">
  Autonomous ML model validation reports —
  <a href="https://github.com/nyosegawa/agentic-bench">GitHub</a>
</p>

<table>
  <thead>
    <tr>
      <th>Date</th>
      <th>Model</th>
      <th>Type</th>
      <th>Status</th>
      <th>Key Metric</th>
      <th>Provider / GPU</th>
      <th>Cost</th>
    </tr>
  </thead>
  <tbody>
      {rows_html}
  </tbody>
</table>

<footer>
  <p>Generated {now} — {len(entries)} benchmark(s)</p>
</footer>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate index.html for benchmark reports")
    parser.add_argument("--output", default=str(REPO_ROOT / "index.html"), help="Output path")
    args = parser.parse_args()

    entries = load_all_metrics()
    html = generate_html(entries)

    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"Generated {output_path} with {len(entries)} report(s)")


if __name__ == "__main__":
    main()
