"""Tests for metrics_writer.py — metrics validation and building."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / ".claude/skills/eval-reporter/scripts")
)

from metrics_writer import build_metrics, validate_metrics, write_metrics


class TestValidateMetrics:
    def test_valid_minimal(self):
        data = {
            "run_id": "2026-02-21T10:00:00+00:00",
            "model": "test/model",
            "model_type": "llm",
            "provider": "colab",
            "stages": {},
        }
        errors = validate_metrics(data)
        assert errors == []

    def test_missing_required_fields(self):
        data = {"model": "test/model"}
        errors = validate_metrics(data)
        assert len(errors) >= 3  # Missing run_id, model_type, provider, stages

    def test_invalid_model_type(self):
        data = {
            "run_id": "2026-02-21T10:00:00+00:00",
            "model": "test/model",
            "model_type": "invalid_type",
            "provider": "colab",
            "stages": {},
        }
        errors = validate_metrics(data)
        assert any("model_type" in e for e in errors)

    def test_invalid_provider(self):
        data = {
            "run_id": "2026-02-21T10:00:00+00:00",
            "model": "test/model",
            "model_type": "llm",
            "provider": "invalid_provider",
            "stages": {},
        }
        errors = validate_metrics(data)
        assert any("provider" in e for e in errors)

    def test_invalid_smoke_status(self):
        data = {
            "run_id": "2026-02-21T10:00:00+00:00",
            "model": "test/model",
            "model_type": "llm",
            "provider": "colab",
            "stages": {"smoke": {"status": "maybe"}},
        }
        errors = validate_metrics(data)
        assert any("smoke" in e for e in errors)

    def test_valid_all_model_types(self):
        for model_type in ["llm", "vlm", "image-gen", "tts", "stt", "embedding", "timeseries"]:
            data = {
                "run_id": "2026-02-21T10:00:00+00:00",
                "model": "test/model",
                "model_type": model_type,
                "provider": "colab",
                "stages": {},
            }
            assert validate_metrics(data) == []

    def test_valid_all_providers(self):
        for provider in ["hf_inference", "colab", "modal", "beam", "api", "local"]:
            data = {
                "run_id": "2026-02-21T10:00:00+00:00",
                "model": "test/model",
                "model_type": "llm",
                "provider": provider,
                "stages": {},
            }
            assert validate_metrics(data) == []


class TestBuildMetrics:
    def test_basic_build(self):
        result = build_metrics(model="test/model", model_type="llm", provider="colab")
        assert result["model"] == "test/model"
        assert result["model_type"] == "llm"
        assert result["provider"] == "colab"
        assert "run_id" in result
        assert result["stages"] == {}

    def test_with_stages(self):
        stages = {"smoke": {"status": "pass", "load_time_seconds": 5.0}}
        result = build_metrics(
            model="test/model", model_type="llm", provider="modal", stages=stages
        )
        assert result["stages"]["smoke"]["status"] == "pass"

    def test_with_optional_fields(self):
        result = build_metrics(
            model="test/model",
            model_type="image-gen",
            provider="beam",
            gpu="A100-40GB",
            cost_usd=0.50,
            duration_seconds=120.0,
        )
        assert result["gpu"] == "A100-40GB"
        assert result["cost_usd"] == 0.50
        assert result["duration_seconds"] == 120.0


class TestWriteMetrics:
    def test_write_valid(self, tmp_path):
        data = build_metrics(model="test/model", model_type="llm", provider="colab")
        output = tmp_path / "metrics.json"
        write_metrics(data, output)
        assert output.exists()
        loaded = json.loads(output.read_text())
        assert loaded["model"] == "test/model"

    def test_write_creates_dirs(self, tmp_path):
        data = build_metrics(model="test/model", model_type="llm", provider="colab")
        output = tmp_path / "deep" / "nested" / "metrics.json"
        write_metrics(data, output)
        assert output.exists()

    def test_write_invalid_fails(self, tmp_path):
        data = {"model": "test/model"}  # Missing required fields
        output = tmp_path / "metrics.json"
        with pytest.raises(SystemExit):
            write_metrics(data, output)
