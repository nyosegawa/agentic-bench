# TTS (Text-to-Speech) Evaluation Guide

## Smoke Test
- Load model with appropriate library (transformers, TTS, bark, etc.)
- Generate audio for a short sentence: "Hello, this is a test of the speech synthesis system."
- Save as WAV file and verify it's playable (non-zero file, correct sample rate)

## Quality Check — Test Sentences

Run at least 3 sentences:

1. **Simple**: "The quick brown fox jumps over the lazy dog."
2. **Numbers/dates**: "On January 15th, 2026, the temperature was minus 3 degrees."
3. **Emotion/expression**: "Oh wow, that's absolutely incredible! I can't believe it!"
4. **Japanese** (if supported): "本日は晴天なり。東京の気温は15度です。"

Save all outputs as WAV/MP3 for human review. Agent cannot listen to audio directly.

Embed in HTML report:
```html
<audio controls>
  <source src="artifacts/output_001.wav" type="audio/wav">
</audio>
```

## Performance Measurement

```python
import time
import wave

text = "This is a sample sentence for measuring speech synthesis speed."

times = []
for _ in range(3):
    start = time.perf_counter()
    audio = model.synthesize(text)  # Adjust to actual API
    elapsed = time.perf_counter() - start

    # Calculate audio duration
    duration_sec = len(audio_samples) / sample_rate
    rtf = elapsed / duration_sec  # Real-Time Factor
    times.append(rtf)

median_rtf = sorted(times)[len(times) // 2]
```

Key metrics:
- **RTF** (Real-Time Factor): processing_time / audio_duration. RTF < 1 means faster than real-time.
- **Sample rate** of output audio (e.g., 22050 Hz, 44100 Hz)
- **Audio duration** generated per second of processing

## Common Issues
- No sound: Check output array isn't all zeros; verify sample rate
- Robotic voice: May need specific speaker embeddings or voice presets
- Slow: Some models (e.g., Bark) are slow by design; document expected RTF
- Dependencies: Many TTS models need `phonemizer`, `espeak-ng` system package
