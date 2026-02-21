"""Tests for hf_model_search.py — HuggingFace model search."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".claude/skills/model-researcher/scripts")

from hf_model_search import TASK_ALIASES, search_models


class TestTaskAliases:
    """Verify task alias mapping."""

    def test_llm_alias(self):
        assert TASK_ALIASES["llm"] == "text-generation"

    def test_vlm_alias(self):
        assert TASK_ALIASES["vlm"] == "image-text-to-text"

    def test_embedding_alias(self):
        assert TASK_ALIASES["embedding"] == "feature-extraction"

    def test_image_gen_alias(self):
        assert TASK_ALIASES["image-gen"] == "text-to-image"

    def test_tts_alias(self):
        assert TASK_ALIASES["tts"] == "text-to-speech"

    def test_stt_alias(self):
        assert TASK_ALIASES["stt"] == "automatic-speech-recognition"

    def test_code_alias(self):
        assert TASK_ALIASES["code"] == "text-generation"


class TestSearchModels:
    """Tests for search_models()."""

    @patch("hf_model_search.HfApi")
    def test_basic_search(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        model = SimpleNamespace(
            id="google/gemma-3-27b-it",
            pipeline_tag="text-generation",
            library_name="transformers",
            downloads=100000,
            likes=500,
            gated=False,
        )
        mock_api.list_models.return_value = [model]

        results = search_models(task="llm", limit=5)
        assert len(results) == 1
        assert results[0]["id"] == "google/gemma-3-27b-it"
        assert results[0]["pipeline_tag"] == "text-generation"

        # Verify alias resolution: "llm" -> "text-generation"
        call_kwargs = mock_api.list_models.call_args[1]
        assert call_kwargs["pipeline_tag"] == "text-generation"

    @patch("hf_model_search.HfApi")
    def test_search_with_keyword(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.list_models.return_value = []

        search_models(search="qwen", limit=10)

        call_kwargs = mock_api.list_models.call_args[1]
        assert call_kwargs["search"] == "qwen"

    @patch("hf_model_search.HfApi")
    def test_inference_only_filter(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.list_models.return_value = []

        search_models(task="llm", inference_only=True)

        call_kwargs = mock_api.list_models.call_args[1]
        assert call_kwargs["inference"] == "warm"

    @patch("hf_model_search.HfApi")
    def test_author_filter(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.list_models.return_value = []

        search_models(task="llm", author="google")

        call_kwargs = mock_api.list_models.call_args[1]
        assert call_kwargs["author"] == "google"

    @patch("hf_model_search.HfApi")
    def test_non_gated_filter(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.list_models.return_value = []

        search_models(task="llm", non_gated=True)

        call_kwargs = mock_api.list_models.call_args[1]
        assert call_kwargs["gated"] is False

    @patch("hf_model_search.HfApi")
    def test_empty_results(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.list_models.return_value = []

        results = search_models(search="nonexistent_xyz_model")
        assert results == []

    @patch("hf_model_search.HfApi")
    def test_result_fields(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        model = SimpleNamespace(
            id="BAAI/bge-m3",
            pipeline_tag="feature-extraction",
            library_name="sentence-transformers",
            downloads=500000,
            likes=200,
            gated="auto",
        )
        mock_api.list_models.return_value = [model]

        results = search_models(task="embedding")
        r = results[0]
        assert r["id"] == "BAAI/bge-m3"
        assert r["library"] == "sentence-transformers"
        assert r["downloads"] == 500000
        assert r["likes"] == 200
        assert r["gated"] == "auto"

    @patch("hf_model_search.HfApi")
    def test_library_filter(self, mock_api_cls):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.list_models.return_value = []

        search_models(task="llm", library="transformers")

        call_kwargs = mock_api.list_models.call_args[1]
        assert call_kwargs["library"] == "transformers"
