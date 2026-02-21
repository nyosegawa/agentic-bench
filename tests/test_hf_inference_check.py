"""Tests for hf_inference_check.py — inference availability checking."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".claude/skills/model-researcher/scripts")

from hf_inference_check import check_inference, is_serverless_available


class TestCheckInference:
    """Tests for check_inference()."""

    @patch("hf_inference_check.model_info")
    def test_accessible_model_with_providers(self, mock_info):
        mapping = MagicMock()
        mapping.status = "live"
        mapping.task = "text-generation"
        mapping.provider_id = "meta-llama/Llama-3-8B"

        info = SimpleNamespace(
            inference="warm",
            pipeline_tag="text-generation",
            gated=False,
            inference_provider_mapping={"hf-inference": mapping},
        )
        mock_info.return_value = info

        result = check_inference("meta-llama/Llama-3-8B")
        assert result["accessible"] is True
        assert result["inference"] == "warm"
        assert "hf-inference" in result["providers"]
        assert result["providers"]["hf-inference"]["status"] == "live"

    @patch("hf_inference_check.model_info")
    def test_accessible_model_no_providers(self, mock_info):
        info = SimpleNamespace(
            inference=None,
            pipeline_tag="text-generation",
            gated=False,
            inference_provider_mapping=None,
        )
        mock_info.return_value = info

        result = check_inference("some/local-only-model")
        assert result["accessible"] is True
        assert result["providers"] == {}
        assert result["inference"] is None

    @patch("hf_inference_check.model_info")
    def test_gated_model(self, mock_info):
        from huggingface_hub.utils import GatedRepoError

        mock_info.side_effect = GatedRepoError("Gated")

        result = check_inference("meta-llama/Llama-Guard")
        assert result["accessible"] is False
        assert result["gated"] is True
        assert "error" in result

    @patch("hf_inference_check.model_info")
    def test_not_found(self, mock_info):
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_info.side_effect = RepositoryNotFoundError("Not found")

        result = check_inference("nonexistent/model-xyz")
        assert result["accessible"] is False
        assert "error" in result

    @patch("hf_inference_check.model_info")
    def test_multiple_providers(self, mock_info):
        def make_mapping(status, task, pid):
            m = MagicMock()
            m.status = status
            m.task = task
            m.provider_id = pid
            return m

        info = SimpleNamespace(
            inference="warm",
            pipeline_tag="text-generation",
            gated=False,
            inference_provider_mapping={
                "hf-inference": make_mapping("live", "text-generation", "p1"),
                "together": make_mapping("live", "text-generation", "p2"),
                "fireworks": make_mapping("staging", "text-generation", "p3"),
            },
        )
        mock_info.return_value = info

        result = check_inference("model-id")
        assert len(result["providers"]) == 3
        assert result["providers"]["fireworks"]["status"] == "staging"


class TestIsServerlessAvailable:
    """Tests for is_serverless_available()."""

    @patch("hf_inference_check.check_inference")
    def test_available(self, mock_check):
        mock_check.return_value = {
            "accessible": True,
            "providers": {"hf-inference": {"status": "live"}},
        }
        assert is_serverless_available("model-id") is True

    @patch("hf_inference_check.check_inference")
    def test_not_available_wrong_status(self, mock_check):
        mock_check.return_value = {
            "accessible": True,
            "providers": {"hf-inference": {"status": "staging"}},
        }
        assert is_serverless_available("model-id") is False

    @patch("hf_inference_check.check_inference")
    def test_not_available_no_provider(self, mock_check):
        mock_check.return_value = {
            "accessible": True,
            "providers": {},
        }
        assert is_serverless_available("model-id") is False

    @patch("hf_inference_check.check_inference")
    def test_not_accessible(self, mock_check):
        mock_check.return_value = {
            "accessible": False,
            "providers": {},
        }
        assert is_serverless_available("model-id") is False
