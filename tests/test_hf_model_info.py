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

    # New model types: video-gen, object-detection, 3d-gen, audio, code-gen fallback

    def test_text_to_video(self):
        info = {"pipeline_tag": "text-to-video", "tags": []}
        assert classify_model_type(info) == "video-gen"

    def test_image_to_video(self):
        info = {"pipeline_tag": "image-to-video", "tags": []}
        assert classify_model_type(info) == "video-gen"

    def test_object_detection(self):
        info = {"pipeline_tag": "object-detection", "tags": []}
        assert classify_model_type(info) == "object-detection"

    def test_image_segmentation(self):
        info = {"pipeline_tag": "image-segmentation", "tags": []}
        assert classify_model_type(info) == "object-detection"

    def test_zero_shot_object_detection(self):
        info = {"pipeline_tag": "zero-shot-object-detection", "tags": []}
        assert classify_model_type(info) == "object-detection"

    def test_image_to_3d(self):
        info = {"pipeline_tag": "image-to-3d", "tags": []}
        assert classify_model_type(info) == "3d-gen"

    def test_text_to_3d(self):
        info = {"pipeline_tag": "text-to-3d", "tags": []}
        assert classify_model_type(info) == "3d-gen"

    def test_text_to_audio(self):
        info = {"pipeline_tag": "text-to-audio", "tags": []}
        assert classify_model_type(info) == "audio"

    def test_audio_to_audio(self):
        info = {"pipeline_tag": "audio-to-audio", "tags": []}
        assert classify_model_type(info) == "audio"

    def test_document_qa_is_vlm(self):
        info = {"pipeline_tag": "document-question-answering", "tags": []}
        assert classify_model_type(info) == "vlm"

    # Tag fallback tests for new types

    def test_fallback_to_tags_code_gen(self):
        info = {"pipeline_tag": "", "tags": ["code", "pytorch"]}
        assert classify_model_type(info) == "code-gen"

    def test_fallback_to_tags_video_gen(self):
        info = {"pipeline_tag": "", "tags": ["video-generation", "diffusion"]}
        assert classify_model_type(info) == "video-gen"

    def test_fallback_to_tags_object_detection(self):
        info = {"pipeline_tag": "", "tags": ["yolo", "detection"]}
        assert classify_model_type(info) == "object-detection"

    def test_fallback_to_tags_3d_gen(self):
        info = {"pipeline_tag": "", "tags": ["3d", "mesh-generation"]}
        assert classify_model_type(info) == "3d-gen"
