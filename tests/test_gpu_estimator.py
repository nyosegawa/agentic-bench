"""Tests for gpu_estimator.py — VRAM estimation and provider recommendation."""

import os
import sys
from pathlib import Path
from unittest import mock

# Add scripts to path
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / ".claude/skills/model-researcher/scripts")
)

from gpu_estimator import (
    BENCH_DURATION_MINUTES,
    GPU_PRICING,
    check_available_providers,
    estimate,
    estimate_cost,
    estimate_vram,
    parse_param_count,
    recommend_gpu,
)


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

    def test_prepaid_providers_first(self):
        recs = recommend_gpu(10.0)
        # Prepaid providers (colab, modal) should come before pay-as-you-go
        prepaid_indices = [i for i, r in enumerate(recs) if r["prepaid"]]
        payg_indices = [i for i, r in enumerate(recs) if not r["prepaid"]]
        if prepaid_indices and payg_indices:
            assert max(prepaid_indices) < min(payg_indices)

    def test_colab_before_hf_endpoints(self):
        recs = recommend_gpu(10.0)
        # Colab (prepaid) should appear before hf_endpoints (pay-as-you-go) for same GPU
        t4_recs = [r for r in recs if r["gpu"] == "T4"]
        colab_idx = next(i for i, r in enumerate(t4_recs) if r["provider"] == "colab")
        hfe_idx = next(i for i, r in enumerate(t4_recs) if r["provider"] == "hf_endpoints")
        assert colab_idx < hfe_idx

    def test_within_prepaid_sorted_by_rate(self):
        recs = recommend_gpu(10.0)
        prepaid_rates = [
            r["hourly_rate_usd"] for r in recs if r["prepaid"] and r["hourly_rate_usd"] is not None
        ]
        assert prepaid_rates == sorted(prepaid_rates)

    def test_within_payg_sorted_by_rate(self):
        recs = recommend_gpu(10.0)
        payg_rates = [
            r["hourly_rate_usd"]
            for r in recs
            if not r["prepaid"] and r["hourly_rate_usd"] is not None
        ]
        assert payg_rates == sorted(payg_rates)

    def test_rtx4090_in_recommendations(self):
        recs = recommend_gpu(20.0)  # 20GB needed — RTX4090 (24GB) fits
        gpu_names = [r["gpu"] for r in recs]
        assert "RTX4090" in gpu_names

    def test_vast_cheapest_for_small_models(self):
        recs = recommend_gpu(10.0)
        # Vast T4 at $0.10/hr should be among the cheapest pay-as-you-go options
        payg_recs = [r for r in recs if not r["prepaid"]]
        if payg_recs:
            cheapest = payg_recs[0]
            assert cheapest["provider"] == "vast"

    def test_vast_and_runpod_in_providers(self):
        recs = recommend_gpu(20.0)
        providers = {r["provider"] for r in recs}
        assert "vast" in providers
        assert "runpod" in providers

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

    def test_estimate_includes_cost(self):
        result = estimate(7_000_000_000, "fp16", "llm")
        assert result["model_type"] == "llm"
        for rec in result["recommendations"]:
            assert "estimated_cost_usd" in rec
            assert "hourly_rate_usd" in rec

    def test_hf_inference_cost_is_zero(self):
        result = estimate(7_000_000_000, "fp16", "llm")
        hf_rec = [r for r in result["recommendations"] if r["provider"] == "hf_inference"]
        assert len(hf_rec) == 1
        assert hf_rec[0]["estimated_cost_usd"] == 0.0


class TestCheckAvailableProviders:
    def test_all_tokens_set(self):
        env = {
            "HF_TOKEN": "hf_xxx",
            "MODAL_TOKEN_ID": "ak-xxx",
            "MODAL_TOKEN_SECRET": "as-xxx",
            "BEAM_TOKEN": "bt-xxx",
            "VAST_API_KEY": "vast-xxx",
            "RUNPOD_API_KEY": "rpa-xxx",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            available = check_available_providers()
        assert available["hf_inference"] is True
        assert available["hf_endpoints"] is True
        assert available["colab"] is True  # no token needed
        assert available["modal"] is True
        assert available["beam"] is True
        assert available["vast"] is True
        assert available["runpod"] is True

    def test_only_hf_token(self):
        env = {"HF_TOKEN": "hf_xxx"}
        # Clear Modal/Beam/Vast/RunPod tokens
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.dict(
                os.environ,
                {
                    "MODAL_TOKEN_ID": "",
                    "MODAL_TOKEN_SECRET": "",
                    "BEAM_TOKEN": "",
                    "VAST_API_KEY": "",
                    "RUNPOD_API_KEY": "",
                },
            ),
        ):
            available = check_available_providers()
        assert available["hf_inference"] is True
        assert available["hf_endpoints"] is True
        assert available["colab"] is True
        assert available["modal"] is False
        assert available["beam"] is False
        assert available["vast"] is False
        assert available["runpod"] is False

    def test_no_tokens(self):
        with mock.patch.dict(
            os.environ,
            {
                "HF_TOKEN": "",
                "MODAL_TOKEN_ID": "",
                "MODAL_TOKEN_SECRET": "",
                "BEAM_TOKEN": "",
                "VAST_API_KEY": "",
                "RUNPOD_API_KEY": "",
            },
        ):
            available = check_available_providers()
        assert available["hf_inference"] is False
        assert available["hf_endpoints"] is False
        assert available["colab"] is True  # always available
        assert available["modal"] is False
        assert available["beam"] is False
        assert available["vast"] is False
        assert available["runpod"] is False


class TestRecommendGpuFilterAvailable:
    def test_filter_removes_unavailable(self):
        env = {"HF_TOKEN": "hf_xxx"}
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.dict(
                os.environ,
                {
                    "MODAL_TOKEN_ID": "",
                    "MODAL_TOKEN_SECRET": "",
                    "BEAM_TOKEN": "",
                    "VAST_API_KEY": "",
                    "RUNPOD_API_KEY": "",
                },
            ),
        ):
            recs = recommend_gpu(10.0, filter_available=True)
        providers = {r["provider"] for r in recs}
        # Only hf_endpoints and colab should remain (modal/beam/vast/runpod filtered out)
        assert "modal" not in providers
        assert "beam" not in providers
        assert "vast" not in providers
        assert "runpod" not in providers
        assert "hf_endpoints" in providers
        assert "colab" in providers

    def test_no_filter_includes_all(self):
        recs = recommend_gpu(10.0, filter_available=False)
        providers = {r["provider"] for r in recs}
        assert "hf_endpoints" in providers
        assert "colab" in providers
        assert "modal" in providers
        assert "beam" in providers
        assert "vast" in providers


class TestEstimateWithEnvCheck:
    def test_filter_available_adds_provider_status(self):
        env = {"HF_TOKEN": "hf_xxx"}
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.dict(
                os.environ,
                {
                    "MODAL_TOKEN_ID": "",
                    "MODAL_TOKEN_SECRET": "",
                    "BEAM_TOKEN": "",
                    "VAST_API_KEY": "",
                    "RUNPOD_API_KEY": "",
                },
            ),
        ):
            result = estimate(7_000_000_000, "fp16", "llm", filter_available=True)
        assert "provider_availability" in result
        assert result["provider_availability"]["hf_inference"] is True
        assert result["provider_availability"]["modal"] is False
        assert result["provider_availability"]["vast"] is False
        assert result["provider_availability"]["runpod"] is False

    def test_no_filter_no_provider_status(self):
        result = estimate(7_000_000_000, "fp16", "llm", filter_available=False)
        assert "provider_availability" not in result


class TestEstimateCost:
    def test_modal_a100_llm(self):
        cost = estimate_cost("A100-40GB", "modal", "llm")
        assert cost["provider"] == "modal"
        assert cost["hourly_rate_usd"] == GPU_PRICING["modal"]["A100-40GB"]
        assert cost["estimated_duration_min"] == BENCH_DURATION_MINUTES["llm"]
        # 15 min at $2.10/hr = $0.525
        assert 0.50 < cost["estimated_cost_usd"] < 0.60

    def test_beam_h100_image_gen(self):
        cost = estimate_cost("H100", "beam", "image-gen")
        assert cost["estimated_duration_min"] == BENCH_DURATION_MINUTES["image-gen"]
        # 20 min at $3.50/hr ≈ $1.17
        assert 1.0 < cost["estimated_cost_usd"] < 1.5

    def test_hf_inference_free(self):
        cost = estimate_cost("cloud", "hf_inference", "llm")
        assert cost["estimated_cost_usd"] == 0.0
        assert "Free" in cost.get("note", "")

    def test_unknown_gpu_returns_none(self):
        cost = estimate_cost("RTX4090", "modal", "llm")
        assert cost["estimated_cost_usd"] is None
        assert cost["hourly_rate_usd"] is None

    def test_custom_duration(self):
        cost = estimate_cost("A100-40GB", "modal", "llm", duration_minutes=30.0)
        assert cost["estimated_duration_min"] == 30.0
        # 30 min at $2.10/hr = $1.05
        assert 1.0 < cost["estimated_cost_usd"] < 1.10

    def test_embedding_short_duration(self):
        cost = estimate_cost("T4", "colab", "embedding")
        assert cost["estimated_duration_min"] == BENCH_DURATION_MINUTES["embedding"]
        # 5 min at $1.17/hr ≈ $0.10
        assert cost["estimated_cost_usd"] < 0.15

    def test_vast_rtx4090_tts(self):
        cost = estimate_cost("RTX4090", "vast", "tts")
        assert cost["provider"] == "vast"
        assert cost["hourly_rate_usd"] == GPU_PRICING["vast"]["RTX4090"]
        assert cost["estimated_duration_min"] == BENCH_DURATION_MINUTES["tts"]
        # 10 min at $0.15/hr = $0.025
        assert cost["estimated_cost_usd"] < 0.05

    def test_vast_h100_llm(self):
        cost = estimate_cost("H100", "vast", "llm")
        assert cost["hourly_rate_usd"] == GPU_PRICING["vast"]["H100"]
        # 15 min at $2.00/hr = $0.50
        assert 0.45 < cost["estimated_cost_usd"] < 0.55

    def test_runpod_rtx4090_tts(self):
        cost = estimate_cost("RTX4090", "runpod", "tts")
        assert cost["provider"] == "runpod"
        assert cost["hourly_rate_usd"] == GPU_PRICING["runpod"]["RTX4090"]
        # 10 min at $0.34/hr ≈ $0.057
        assert cost["estimated_cost_usd"] < 0.10

    def test_runpod_a100_80gb_llm(self):
        cost = estimate_cost("A100-80GB", "runpod", "llm")
        assert cost["hourly_rate_usd"] == GPU_PRICING["runpod"]["A100-80GB"]
        # 15 min at $1.19/hr ≈ $0.30
        assert 0.25 < cost["estimated_cost_usd"] < 0.35
