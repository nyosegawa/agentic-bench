"""Tests for gpu_estimator.py — VRAM estimation and provider recommendation."""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / ".claude/skills/model-researcher/scripts")
)

from gpu_estimator import estimate, estimate_vram, parse_param_count, recommend_gpu


class TestParseParamCount:
    def test_billions(self):
        assert parse_param_count("7B") == 7_000_000_000
        assert parse_param_count("27b") == 27_000_000_000
        assert parse_param_count("0.5B") == 500_000_000

    def test_millions(self):
        assert parse_param_count("125M") == 125_000_000
        assert parse_param_count("350m") == 350_000_000

    def test_thousands(self):
        assert parse_param_count("100K") == 100_000

    def test_raw_number(self):
        assert parse_param_count("7000000000") == 7_000_000_000
        assert parse_param_count("125000000") == 125_000_000

    def test_with_whitespace(self):
        assert parse_param_count("  7B  ") == 7_000_000_000


class TestEstimateVram:
    def test_7b_fp16(self):
        vram = estimate_vram(7_000_000_000, "fp16")
        # 7B * 2 bytes / 1GB * 1.2 overhead ≈ 15.6 GB
        assert 15 < vram < 17

    def test_7b_int8(self):
        vram = estimate_vram(7_000_000_000, "int8")
        # 7B * 1 byte / 1GB * 1.2 overhead ≈ 7.8 GB
        assert 7 < vram < 9

    def test_7b_int4(self):
        vram = estimate_vram(7_000_000_000, "int4")
        # 7B * 0.5 bytes / 1GB * 1.2 overhead ≈ 3.9 GB
        assert 3 < vram < 5

    def test_70b_fp16(self):
        vram = estimate_vram(70_000_000_000, "fp16")
        # ~156 GB — needs multi-GPU or quantization
        assert vram > 100


class TestRecommendGpu:
    def test_small_model_fits_t4(self):
        recs = recommend_gpu(10.0)  # 10GB needed
        gpu_names = [r["gpu"] for r in recs]
        assert "T4" in gpu_names

    def test_medium_model_needs_a100(self):
        recs = recommend_gpu(30.0)  # 30GB needed
        gpu_names = [r["gpu"] for r in recs]
        assert "T4" not in gpu_names
        assert "A100-40GB" in gpu_names

    def test_large_model_needs_80gb(self):
        recs = recommend_gpu(60.0)  # 60GB needed
        gpu_names = [r["gpu"] for r in recs]
        assert "T4" not in gpu_names
        assert "A100-40GB" not in gpu_names
        assert "A100-80GB" in gpu_names or "H100" in gpu_names

    def test_colab_preferred_over_modal(self):
        recs = recommend_gpu(10.0)
        # First recommendation with T4 should be colab
        t4_recs = [r for r in recs if r["gpu"] == "T4"]
        assert t4_recs[0]["provider"] == "colab"

    def test_no_recommendation_for_huge(self):
        recs = recommend_gpu(200.0)  # 200GB — nothing fits
        assert len(recs) == 0


class TestEstimate:
    def test_small_model_hf_viable(self):
        result = estimate(7_000_000_000, "fp16")
        assert result["hf_inference_viable"] is True
        providers = [r["provider"] for r in result["recommendations"]]
        assert "hf_inference" in providers

    def test_large_model_not_hf_viable(self):
        result = estimate(70_000_000_000, "fp16")
        assert result["hf_inference_viable"] is False

    def test_quantized_large_model(self):
        result = estimate(70_000_000_000, "int4")
        # 70B int4 ≈ 39GB — should fit A100-40GB
        assert result["estimated_vram_gb"] < 45
        gpu_names = [r["gpu"] for r in result["recommendations"]]
        assert "A100-40GB" in gpu_names
