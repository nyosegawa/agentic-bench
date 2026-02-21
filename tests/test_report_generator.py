"""Tests for report_generator.py — HTML report generation."""

import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / ".claude/skills/eval-reporter/scripts")
)

from report_generator import discover_artifacts, generate_report, load_metrics


class TestDiscoverArtifacts:
    def test_empty_dir(self, tmp_path):
        result = discover_artifacts(tmp_path)
        assert result == {"images": [], "audio": [], "text": [], "other": []}

    def test_nonexistent_dir(self, tmp_path):
        result = discover_artifacts(tmp_path / "nonexistent")
        assert result == {"images": [], "audio": [], "text": [], "other": []}

    def test_image_files(self, tmp_path):
        (tmp_path / "photo.png").write_bytes(b"fake png")
        (tmp_path / "render.jpg").write_bytes(b"fake jpg")
        result = discover_artifacts(tmp_path)
        assert len(result["images"]) == 2

    def test_audio_files(self, tmp_path):
        (tmp_path / "speech.wav").write_bytes(b"fake wav")
        (tmp_path / "music.mp3").write_bytes(b"fake mp3")
        result = discover_artifacts(tmp_path)
        assert len(result["audio"]) == 2

    def test_text_files(self, tmp_path):
        (tmp_path / "output.txt").write_text("hello")
        (tmp_path / "data.json").write_text("{}")
        result = discover_artifacts(tmp_path)
        assert len(result["text"]) == 2

    def test_mixed_files(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"png")
        (tmp_path / "audio.wav").write_bytes(b"wav")
        (tmp_path / "text.txt").write_text("txt")
        (tmp_path / "model.bin").write_bytes(b"bin")
        result = discover_artifacts(tmp_path)
        assert len(result["images"]) == 1
        assert len(result["audio"]) == 1
        assert len(result["text"]) == 1
        assert len(result["other"]) == 1


class TestLoadMetrics:
    def test_load_valid(self, tmp_path):
        data = {"model": "test", "stages": {}}
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(data))
        result = load_metrics(path)
        assert result["model"] == "test"


class TestGenerateReport:
    def test_basic_report(self, tmp_path):
        metrics = {
            "run_id": "2026-02-21T10:00:00+00:00",
            "model": "test/model-7b",
            "model_type": "llm",
            "provider": "colab",
            "gpu": "T4",
            "stages": {
                "smoke": {"status": "pass", "load_time_seconds": 5.0},
                "quality": {"outputs": [], "notes": "Good quality output"},
                "performance": {
                    "tokens_per_second": 35.0,
                    "latency_p50_ms": 28.5,
                    "latency_p99_ms": 95.0,
                    "num_runs": 5,
                },
            },
            "cost_usd": 0.0,
            "duration_seconds": 120,
        }
        artifacts = {"images": [], "audio": [], "text": [], "other": []}
        output = tmp_path / "report.html"

        generate_report(metrics, artifacts, output)

        assert output.exists()
        html = output.read_text()
        assert "test/model-7b" in html
        assert "PASS" in html
        assert "35.0" in html

    def test_report_with_artifacts(self, tmp_path):
        metrics = {
            "run_id": "2026-02-21T10:00:00+00:00",
            "model": "test/image-model",
            "model_type": "image-gen",
            "provider": "modal",
            "gpu": "A100-40GB",
            "stages": {
                "smoke": {"status": "pass"},
                "quality": {"outputs": ["artifacts/img.png"], "notes": "Sharp images"},
                "performance": {"sec_per_image": 4.2, "num_runs": 3},
            },
            "cost_usd": 0.15,
            "duration_seconds": 60,
        }
        # Create a fake artifact
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "img.png").write_bytes(b"fake png")

        artifacts = discover_artifacts(artifacts_dir)
        output = tmp_path / "report.html"

        generate_report(metrics, artifacts, output)

        html = output.read_text()
        assert "test/image-model" in html
        assert "4.2" in html
