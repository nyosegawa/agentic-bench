# Audio / Music Generation Evaluation Guide

## Target Models

MusicGen, AudioCraft, Stable Audio, Riffusion, AudioLDM2.

## Setup

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch, time, soundfile as sf

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to("cuda")
```

For AudioLDM2:
```python
from diffusers import AudioLDM2Pipeline
pipe = AudioLDM2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
audio = pipe("lo-fi hip hop beat", num_inference_steps=50).audios[0]
```

## Smoke Test

```python
inputs = processor(text=["cheerful acoustic guitar melody"], padding=True, return_tensors="pt")
inputs = inputs.to("cuda")
audio_values = model.generate(**inputs, max_new_tokens=256)
sf.write("output.wav", audio_values[0, 0].cpu().numpy(),
         model.config.audio_encoder.sampling_rate)
# Verify file is non-empty and playable
```

## Quality Evaluation (Human Listening)

Generate audio for diverse prompts and embed in HTML report:
1. Instrumental music (specific genre: jazz, rock, classical)
2. Sound effects (rain, thunder, footsteps)
3. Ambient/background audio
4. Music with specific mood (happy, sad, energetic)
5. Complex prompt (multiple instruments, specific tempo)

### HTML Embedding
```html
<audio controls src="artifacts/output_01.wav"></audio>
<p>Prompt: "cheerful acoustic guitar melody"</p>
```

## Performance Metrics

```python
start = time.perf_counter()
audio_values = model.generate(**inputs, max_new_tokens=512)
elapsed = time.perf_counter() - start
audio_duration = len(audio_values[0, 0]) / model.config.audio_encoder.sampling_rate
rtf = elapsed / audio_duration
```

Key metrics:
- **RTF**: generation_time / audio_duration
- **sec/generation**: wall-clock per clip
- **Audio duration**: how long the output is

## Common Issues

- **Generation is slow**: Music models can take minutes per clip. Budget time.
- **Sampling rate**: Varies by model (16kHz, 24kHz, 32kHz). Check config.
- **Duration control**: Some models use `max_new_tokens`, others use seconds.
- **Stereo vs mono**: Some models output stereo. Handle both in report.
