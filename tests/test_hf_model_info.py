"""Tests for hf_model_info.py — model type classification (no API calls)."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / ".claude/skills/model-researcher/scripts")
)

from hf_model_info import classify_model_type


class TestClassifyModelType:
    def test_text_generation(self):
        assert classify_model_type({"pipeline_tag": "text-generation", "tags": []}) == "llm"

    def test_text2text(self):
        assert classify_model_type({"pipeline_tag": "text2text-generation", "tags": []}) == "llm"

    def test_conversational(self):
        assert classify_model_type({"pipeline_tag": "conversational", "tags": []}) == "llm"

    def test_text_to_image(self):
        info = {"pipeline_tag": "text-to-image", "tags": []}
        assert classify_model_type(info) == "image-gen"

    def test_tts(self):
        assert classify_model_type({"pipeline_tag": "text-to-speech", "tags": []}) == "tts"

    def test_stt(self):
        info = {"pipeline_tag": "automatic-speech-recognition", "tags": []}
        assert classify_model_type(info) == "stt"

    def test_vlm(self):
        info = {"pipeline_tag": "image-text-to-text", "tags": []}
        assert classify_model_type(info) == "vlm"

    def test_embedding(self):
        info = {"pipeline_tag": "feature-extraction", "tags": []}
        assert classify_model_type(info) == "embedding"

    def test_timeseries(self):
        info = {"pipeline_tag": "time-series-forecasting", "tags": []}
        assert classify_model_type(info) == "timeseries"

    def test_unknown_pipeline(self):
        info = {"pipeline_tag": "something-new", "tags": []}
        assert classify_model_type(info) == "unknown"

    def test_fallback_to_tags_llm(self):
        info = {"pipeline_tag": "", "tags": ["llm", "pytorch"]}
        assert classify_model_type(info) == "llm"

    def test_fallback_to_tags_diffusion(self):
        info = {"pipeline_tag": "", "tags": ["diffusion", "image"]}
        assert classify_model_type(info) == "image-gen"

    def test_fallback_to_tags_tts(self):
        info = {"pipeline_tag": "", "tags": ["tts", "audio"]}
        assert classify_model_type(info) == "tts"

    def test_empty_everything(self):
        info = {"pipeline_tag": "", "tags": []}
        assert classify_model_type(info) == "unknown"
