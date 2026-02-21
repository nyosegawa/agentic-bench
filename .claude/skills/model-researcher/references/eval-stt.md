# STT (Speech Recognition) Evaluation Guide

## Target Models

Whisper (large-v3, turbo), Canary, wav2vec2, Conformer, faster-whisper.

## Setup

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch, time, librosa

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
```

For faster-whisper (CTranslate2 backend):
```python
from faster_whisper import WhisperModel
model = WhisperModel(MODEL_ID, device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav")
```

## Test Audio Preparation

Agent should prepare or locate test audio:
- Short clip (5-10 sec): basic smoke test
- Medium clip (30-60 sec): quality evaluation
- Long clip (5+ min): throughput and chunking test
- Multi-language: English + Japanese at minimum

If no test audio exists, generate synthetic audio via TTS or use
publicly available speech datasets (LibriSpeech, Common Voice).

## Smoke Test

```python
audio, sr = librosa.load("test.wav", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(model.device)
output = model.generate(**inputs)
text = processor.batch_decode(output, skip_special_tokens=True)[0]
assert len(text) > 0, "Transcription returned empty"
```

## Quality Evaluation

### WER (Word Error Rate)

If ground truth transcription is available:
```python
import jiwer
wer = jiwer.wer(reference_text, hypothesis_text)
cer = jiwer.cer(reference_text, hypothesis_text)
```

If no ground truth: display transcription for human review in the report.

### Test Cases
1. Clear speech, native speaker (should achieve low WER)
2. Accented or non-native speech
3. Noisy environment (background noise)
4. Japanese speech (multi-language capability check)
5. Technical vocabulary / domain-specific terms

## Performance Metrics

```python
audio_duration = len(audio) / sr
start = time.perf_counter()
output = model.generate(**inputs)
elapsed = time.perf_counter() - start
rtf = elapsed / audio_duration  # <1.0 means real-time capable
```

Key metrics:
- **RTF** (Real-Time Factor): <1.0 = faster than real-time
- **Latency**: time to first output
- **WER/CER**: if ground truth available

## Common Issues

- **Sample rate mismatch**: Most models expect 16kHz mono. Always resample.
- **Long audio**: Models may have max length (30s for Whisper). Use chunking.
- **Language detection**: Whisper auto-detects. Force language with `language="ja"` if needed.
- **faster-whisper**: Much faster but different API. May need CTranslate2 install.
