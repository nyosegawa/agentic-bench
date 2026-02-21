"""GLM-OCR Benchmark on Modal (T4 GPU)

Tests:
1. Smoke test: Simple text recognition
2. English text: Document paragraph
3. Japanese text: Japanese document
4. Math formula: LaTeX formula image
5. Table: Structured table image
6. Information extraction: Structured JSON extraction
7. Performance: 5-run speed measurement
"""

import modal
import json
import base64
import io
import time

app = modal.App("agentic-bench-glm-ocr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("fonts-dejavu-core", "fonts-noto-cjk", "git")
    .pip_install(
        "torch",
        "torchvision",
        "accelerate",
        "Pillow",
        "huggingface_hub",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_ID = "zai-org/GLM-OCR"


def create_test_images():
    """Generate test images with PIL for OCR evaluation."""
    from PIL import Image, ImageDraw, ImageFont
    import os

    images = {}

    # --- 1. Smoke test: simple English text ---
    img = Image.new("RGB", (600, 100), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((20, 30), "Hello World! OCR test 2026.", fill="black", font=font)
    images["smoke_test"] = {
        "image": img,
        "prompt": "Text Recognition:",
        "expected_contains": ["Hello", "World", "OCR", "2026"],
        "description": "Simple English text line",
    }

    # --- 2. English paragraph ---
    img = Image.new("RGB", (700, 200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()
    lines = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "GLM-OCR achieves state-of-the-art document understanding.",
    ]
    for i, line in enumerate(lines):
        draw.text((20, 20 + i * 40), line, fill="black", font=font)
    images["english_paragraph"] = {
        "image": img,
        "prompt": "Text Recognition:",
        "expected_contains": ["quick brown fox", "machine learning", "artificial intelligence"],
        "description": "Multi-line English paragraph",
    }

    # --- 3. Japanese text ---
    img = Image.new("RGB", (700, 160), "white")
    draw = ImageDraw.Draw(img)
    # Try common CJK fonts
    cjk_font = None
    for fp in [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]:
        try:
            cjk_font = ImageFont.truetype(fp, 22)
            break
        except (IOError, OSError):
            continue
    if cjk_font is None:
        cjk_font = font  # fallback
    ja_lines = [
        "人工知能は現代社会を変革しています。",
        "OCR技術は文書のデジタル化に不可欠です。",
    ]
    for i, line in enumerate(ja_lines):
        draw.text((20, 20 + i * 50), line, fill="black", font=cjk_font)
    images["japanese_text"] = {
        "image": img,
        "prompt": "Text Recognition:",
        "expected_contains": ["人工知能", "OCR", "デジタル"],
        "description": "Japanese text recognition",
    }

    # --- 4. Math formula (rendered as text) ---
    img = Image.new("RGB", (700, 180), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_math = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except (IOError, OSError):
        font_math = ImageFont.load_default()
    math_lines = [
        "E = mc²",
        "f(x) = ax² + bx + c",
        "∫₀¹ x² dx = 1/3",
    ]
    for i, line in enumerate(math_lines):
        draw.text((20, 20 + i * 50), line, fill="black", font=font_math)
    images["math_formula"] = {
        "image": img,
        "prompt": "Formula Recognition:",
        "expected_contains": ["E", "mc", "f(x)", "ax"],
        "description": "Mathematical formula recognition",
    }

    # --- 5. Table ---
    img = Image.new("RGB", (600, 250), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_tbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (IOError, OSError):
        font_tbl = ImageFont.load_default()
    # Draw table grid
    for y in [20, 60, 100, 140, 180]:
        draw.line([(20, y), (560, y)], fill="black", width=2)
    for x in [20, 160, 320, 440, 560]:
        draw.line([(x, 20), (x, 180)], fill="black", width=2)
    # Header
    headers = ["Name", "Age", "City", "Score"]
    x_positions = [30, 170, 330, 450]
    for x, h in zip(x_positions, headers):
        draw.text((x, 30), h, fill="black", font=font_tbl)
    # Rows
    rows = [
        ["Alice", "28", "Tokyo", "92.5"],
        ["Bob", "34", "London", "87.3"],
        ["Charlie", "22", "New York", "95.1"],
    ]
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            draw.text((x_positions[ci], 70 + ri * 40), cell, fill="black", font=font_tbl)
    images["table"] = {
        "image": img,
        "prompt": "Table Recognition:",
        "expected_contains": ["Alice", "Bob", "Charlie", "Tokyo", "London"],
        "description": "Structured table recognition",
    }

    # --- 6. Information extraction ---
    img = Image.new("RGB", (500, 250), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_ie = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except (IOError, OSError):
        font_ie = ImageFont.load_default()
    ie_lines = [
        "INVOICE #12345",
        "Date: 2026-02-21",
        "Customer: Taro Yamada",
        "Item: GPU Server (x2)",
        "Total: $4,500.00",
    ]
    for i, line in enumerate(ie_lines):
        draw.text((20, 20 + i * 40), line, fill="black", font=font_ie)
    ie_prompt = """Information Extraction:
```json
{
    "invoice_number": "",
    "date": "",
    "customer": "",
    "item": "",
    "total": ""
}
```"""
    images["info_extraction"] = {
        "image": img,
        "prompt": ie_prompt,
        "expected_contains": ["12345", "2026", "Yamada", "4,500"],
        "description": "Structured information extraction",
    }

    return images


@app.function(
    gpu="T4",
    image=image,
    timeout=900,
    secrets=[modal.Secret.from_dotenv(path="/Users/sakasegawa/src/github.com/nyosegawa/agentic-bench")],
)
def run_glm_ocr_benchmark() -> dict:
    """Run full GLM-OCR benchmark."""
    import torch
    import traceback
    import os
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    results = {
        "model_id": MODEL_ID,
        "gpu": "T4",
        "tests": [],
        "performance": {},
        "errors": [],
    }

    # --- Load model ---
    print("Loading model...")
    print(f"HF_TOKEN set: {'HF_TOKEN' in os.environ}")
    load_start = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        load_time = time.perf_counter() - load_start
        results["load_time_seconds"] = round(load_time, 2)
        print(f"Model loaded in {load_time:.1f}s")

        # GPU info
        if torch.cuda.is_available():
            results["gpu_name"] = torch.cuda.get_device_name(0)
            results["gpu_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception as e:
        tb = traceback.format_exc()
        results["errors"].append(f"Model load failed: {str(e)}\n{tb}")
        print(f"FULL ERROR:\n{tb}")
        return results

    # --- Create test images ---
    print("Creating test images...")
    test_images = create_test_images()

    # --- Helper function ---
    def run_ocr(img, prompt, max_tokens=2048):
        import tempfile, os

        # Save PIL image to temp file (model card pattern uses file URLs)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": tmp_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)

            start = time.perf_counter()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            elapsed = time.perf_counter() - start

            output_text = processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            tokens_generated = generated_ids.shape[1] - inputs["input_ids"].shape[1]

            return {
                "output": output_text,
                "elapsed_seconds": round(elapsed, 3),
                "tokens_generated": int(tokens_generated),
                "tokens_per_second": round(tokens_generated / elapsed, 1) if elapsed > 0 else 0,
            }
        finally:
            os.unlink(tmp_path)

    # --- Run tests ---
    for test_name, test_data in test_images.items():
        print(f"\nRunning test: {test_name}...")
        try:
            result = run_ocr(test_data["image"], test_data["prompt"])

            # Check expected content
            output_lower = result["output"].lower()
            matches = []
            for expected in test_data["expected_contains"]:
                found = expected.lower() in output_lower
                matches.append({"term": expected, "found": found})

            match_rate = sum(1 for m in matches if m["found"]) / len(matches)

            test_result = {
                "name": test_name,
                "description": test_data["description"],
                "prompt": test_data["prompt"][:100],
                "output": result["output"],
                "elapsed_seconds": result["elapsed_seconds"],
                "tokens_generated": result["tokens_generated"],
                "tokens_per_second": result["tokens_per_second"],
                "expected_matches": matches,
                "match_rate": round(match_rate, 2),
                "status": "pass" if match_rate >= 0.5 else "fail",
            }
            results["tests"].append(test_result)
            print(f"  Result: {test_result['status']} (match_rate={match_rate:.0%})")
            print(f"  Output: {result['output'][:200]}")

        except Exception as e:
            results["tests"].append({
                "name": test_name,
                "description": test_data["description"],
                "status": "error",
                "error": str(e),
            })
            results["errors"].append(f"Test {test_name} failed: {str(e)}")
            print(f"  ERROR: {e}")

    # --- Performance benchmark ---
    print("\nRunning performance benchmark (5 runs)...")
    try:
        # Use the smoke test image for consistent measurement
        perf_img = test_images["smoke_test"]["image"]
        perf_prompt = "Text Recognition:"

        # Warmup
        run_ocr(perf_img, perf_prompt, max_tokens=128)

        perf_times = []
        perf_tok_per_sec = []
        for i in range(5):
            r = run_ocr(perf_img, perf_prompt, max_tokens=128)
            perf_times.append(r["elapsed_seconds"])
            perf_tok_per_sec.append(r["tokens_per_second"])
            print(f"  Run {i+1}: {r['elapsed_seconds']:.3f}s, {r['tokens_per_second']:.1f} tok/s")

        perf_times_sorted = sorted(perf_times)
        perf_tps_sorted = sorted(perf_tok_per_sec)

        results["performance"] = {
            "runs": 5,
            "latency_median_seconds": round(perf_times_sorted[2], 3),
            "latency_p99_seconds": round(perf_times_sorted[4], 3),
            "latency_min_seconds": round(perf_times_sorted[0], 3),
            "tokens_per_second_median": round(perf_tps_sorted[2], 1),
            "tokens_per_second_max": round(perf_tps_sorted[4], 1),
            "all_latencies": perf_times,
            "all_tps": perf_tok_per_sec,
        }
    except Exception as e:
        results["errors"].append(f"Performance benchmark failed: {str(e)}")
        print(f"  Performance ERROR: {e}")

    # --- Save test images as base64 for report ---
    image_data = {}
    for test_name, test_data in test_images.items():
        buf = io.BytesIO()
        test_data["image"].save(buf, format="PNG")
        image_data[test_name] = base64.b64encode(buf.getvalue()).decode("utf-8")
    results["test_image_data"] = image_data

    # --- VRAM usage ---
    if torch.cuda.is_available():
        results["gpu_memory_used_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    print("\n=== Benchmark complete ===")
    return results


@app.local_entrypoint()
def main():
    print("Starting GLM-OCR benchmark on Modal T4...")
    result = run_glm_ocr_benchmark.remote()

    # Save results
    output_path = "results/2026-02-21_glm-ocr/artifacts/benchmark_results.json"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\nModel: {result['model_id']}")
    print(f"GPU: {result.get('gpu_name', 'unknown')}")
    print(f"Load time: {result.get('load_time_seconds', 'N/A')}s")
    print(f"VRAM used: {result.get('gpu_memory_used_gb', 'N/A')} GB")

    if result.get("tests"):
        passed = sum(1 for t in result["tests"] if t.get("status") == "pass")
        total = len(result["tests"])
        print(f"\nTests: {passed}/{total} passed")
        for t in result["tests"]:
            status_icon = "✓" if t.get("status") == "pass" else ("✗" if t.get("status") == "fail" else "⚠")
            print(f"  {status_icon} {t['name']}: {t.get('status', 'unknown')}")

    if result.get("performance"):
        perf = result["performance"]
        print(f"\nPerformance (median of {perf['runs']} runs):")
        print(f"  Latency: {perf['latency_median_seconds']}s")
        print(f"  Throughput: {perf['tokens_per_second_median']} tok/s")

    if result.get("errors"):
        print(f"\nErrors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"  - {e}")
