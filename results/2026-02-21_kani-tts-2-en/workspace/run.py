"""
KaniTTS2-en Benchmark on Modal (L4 GPU)
Model: nineninesix/kani-tts-2-en
"""

import json
import modal

app = modal.App("agentic-bench-kani-tts-2-en")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("uv")
    .run_commands(
        # Step 1: Core ML stack
        "uv pip install --system "
        "torch torchaudio "
        "numpy soundfile scipy "
        "transformers==4.56.0 accelerate "
        "librosa omegaconf hydra-core==1.3.2 "
        "einops sentencepiece protobuf "
        "scikit-learn numba wrapt cloudpickle "
        "lightning==2.4.0 torchmetrics peft "
        "torchcodec matplotlib pandas editdistance "
        "braceexpand text-unidecode ruamel.yaml onnx "
        "seaborn nltk attrdict pypinyin pypinyin-dict "
        "lhotse marshmallow pydub pyloudnorm resampy sox "
        "datasets inflect sacremoses num2words "
        "pyannote.core pyannote.metrics jiwer fiddle "
        "tensorboard wget kornia mediapy jieba janome cdifflib "
        "webdataset texterrors whisper-normalizer nemo_text_processing kaldi-python-io "
        "wandb optuna "
    )
    .run_commands(
        # Step 2: kani-tts-2 and nemo-toolkit without deps
        # to bypass nemo-toolkit's transformers<=4.52.0 constraint
        "uv pip install --system --no-deps kani-tts-2 'nemo-toolkit[tts]==2.4.0'"
    )
)

MODEL_ID = "nineninesix/kani-tts-2-en"

TEST_SENTENCES = {
    "simple": "The quick brown fox jumps over the lazy dog.",
    "numbers": "On January 15th, 2026, the temperature was minus 3 degrees Celsius.",
    "emotion": "Oh wow, that is absolutely incredible! I cannot believe it!",
    "long": (
        "Artificial intelligence is transforming the way we interact with technology. "
        "From voice assistants to autonomous vehicles, the applications are vast "
        "and continue to grow every day. Researchers around the world are working "
        "to make these systems more capable, more efficient, and more accessible "
        "to everyone."
    ),
}


@app.function(gpu="L4", image=image, timeout=600)
def run_benchmark() -> dict:
    import io
    import os
    import tempfile
    import time

    import numpy as np
    import soundfile as sf
    import torch

    TMPDIR = tempfile.mkdtemp()

    def save_and_read(model, audio, filename):
        """Save audio to temp file, read back as bytes and get duration."""
        path = os.path.join(TMPDIR, filename)
        model.save_audio(audio, path)
        with open(path, "rb") as f:
            audio_bytes = f.read()
        with sf.SoundFile(path) as f:
            sample_rate = f.samplerate
            duration = len(f) / f.samplerate
        return audio_bytes, sample_rate, duration

    results = {
        "model_id": MODEL_ID,
        "smoke_test": {},
        "quality_check": {},
        "performance": {},
        "audio_files": {},
        "device_info": {},
    }

    results["device_info"] = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(
            getattr(torch.cuda.get_device_properties(0), "total_memory",
                    getattr(torch.cuda.get_device_properties(0), "total_mem", 0)) / 1e9, 2
        ) if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }
    print(f"Device: {results['device_info']}")

    # --- Stage 1: Smoke Test ---
    print("\n=== Stage 1: Smoke Test ===")
    try:
        from kani_tts import KaniTTS

        load_start = time.perf_counter()
        model = KaniTTS(MODEL_ID)
        load_time = time.perf_counter() - load_start
        print(f"Model loaded in {load_time:.2f}s")

        # Check available methods for language tag
        print(f"KaniTTS methods: {[m for m in dir(model) if not m.startswith('_')]}")
        if hasattr(model, 'show_language_tags'):
            model.show_language_tags()

        smoke_start = time.perf_counter()
        audio, text = model("Hello, this is a test of the speech synthesis system.", language_tag="en_us")
        smoke_time = time.perf_counter() - smoke_start

        smoke_audio_bytes, sample_rate, audio_duration = save_and_read(
            model, audio, "smoke_test.wav"
        )

        results["smoke_test"] = {
            "status": "pass",
            "load_time_seconds": round(load_time, 3),
            "generation_time_seconds": round(smoke_time, 3),
            "audio_duration_seconds": round(audio_duration, 3),
            "sample_rate": sample_rate,
            "audio_size_bytes": len(smoke_audio_bytes),
            "text_returned": text,
        }
        results["audio_files"]["smoke_test.wav"] = smoke_audio_bytes
        print(f"Smoke test PASSED: {smoke_time:.2f}s, audio={audio_duration:.2f}s, sr={sample_rate}")

    except Exception as e:
        results["smoke_test"] = {"status": "fail", "error": str(e)}
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return results

    # --- Stage 2: Quality Check ---
    print("\n=== Stage 2: Quality Check ===")
    for name, sentence in TEST_SENTENCES.items():
        try:
            gen_start = time.perf_counter()
            audio, text = model(sentence)
            gen_time = time.perf_counter() - gen_start

            audio_bytes, _, audio_duration = save_and_read(
                model, audio, f"quality_{name}.wav"
            )

            rtf = gen_time / audio_duration if audio_duration > 0 else float("inf")

            results["quality_check"][name] = {
                "status": "pass",
                "input_text": sentence,
                "text_returned": text,
                "generation_time_seconds": round(gen_time, 3),
                "audio_duration_seconds": round(audio_duration, 3),
                "rtf": round(rtf, 4),
                "audio_size_bytes": len(audio_bytes),
            }
            results["audio_files"][f"quality_{name}.wav"] = audio_bytes
            print(f"  [{name}] OK: gen={gen_time:.2f}s, dur={audio_duration:.2f}s, RTF={rtf:.3f}")

        except Exception as e:
            results["quality_check"][name] = {
                "status": "fail",
                "input_text": sentence,
                "error": str(e),
            }
            print(f"  [{name}] FAILED: {e}")

    # --- Stage 3: Performance Measurement ---
    print("\n=== Stage 3: Performance ===")
    perf_text = "This is a sample sentence for measuring speech synthesis speed."
    rtfs = []
    gen_times = []

    for i in range(5):
        try:
            start = time.perf_counter()
            audio, _ = model(perf_text)
            elapsed = time.perf_counter() - start

            _, _, dur = save_and_read(model, audio, f"perf_{i}.wav")

            rtf = elapsed / dur if dur > 0 else float("inf")
            rtfs.append(rtf)
            gen_times.append(elapsed)
            print(f"  Run {i+1}: gen={elapsed:.3f}s, dur={dur:.3f}s, RTF={rtf:.4f}")
        except Exception as e:
            print(f"  Run {i+1} FAILED: {e}")

    if rtfs:
        rtfs_sorted = sorted(rtfs)
        gen_sorted = sorted(gen_times)
        results["performance"] = {
            "test_text": perf_text,
            "num_runs": len(rtfs),
            "rtf_median": round(rtfs_sorted[len(rtfs_sorted) // 2], 4),
            "rtf_min": round(min(rtfs), 4),
            "rtf_max": round(max(rtfs), 4),
            "rtf_all": [round(r, 4) for r in rtfs],
            "gen_time_median_seconds": round(gen_sorted[len(gen_sorted) // 2], 3),
            "gen_time_all_seconds": [round(g, 3) for g in gen_times],
        }

    print("\n=== Benchmark Complete ===")
    return results


@app.local_entrypoint()
def main():
    import os

    result = run_benchmark.remote()

    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    audio_files = result.pop("audio_files", {})
    for filename, audio_bytes in audio_files.items():
        filepath = os.path.join(artifacts_dir, filename)
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        print(f"Saved: {filepath}")

    results_path = os.path.join(os.path.dirname(__file__), "..", "raw_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(json.dumps(result, indent=2))
