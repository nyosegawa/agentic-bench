# Video Generation Evaluation Guide

## Target Models

Wan2.1, HunyuanVideo, CogVideoX, AnimateDiff, LTX-Video.

## ⚠️ Cost Warning

Video generation is the **most expensive** model type:
- Requires 24-80GB+ VRAM
- 1 generation can take 5-30 minutes
- Budget entire benchmark at ~30 min, $2-5+

Always confirm cost estimate with user before proceeding.

## Setup

```python
# Most video models use diffusers
from diffusers import DiffusionPipeline  # or model-specific pipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda")
```

**Important**: Each video model often requires a specific pipeline class.
Check the model card:
- CogVideoX: `CogVideoXPipeline`
- AnimateDiff: `AnimateDiffPipeline`
- Wan2.1: may need custom pipeline

## Smoke Test

```python
video = pipe("A cat walking on grass", num_frames=16, num_inference_steps=25)
from diffusers.utils import export_to_video
export_to_video(video.frames[0], "output.mp4", fps=8)
# Verify mp4 is playable and non-empty
```

## Quality Evaluation (Human Viewing)

Generate videos for diverse prompts and embed in HTML report:
1. Simple motion (person walking, animal moving)
2. Camera movement (panning, zooming)
3. Scene transition or temporal consistency
4. Complex scene (multiple objects interacting)
5. Text prompt adherence (specific colors, objects, actions)

### HTML Embedding
```html
<video controls width="512" src="artifacts/output_01.mp4"></video>
<p>Prompt: "A cat walking on grass"</p>
```

### What to Look For
- **Temporal consistency**: Do objects maintain shape/identity across frames?
- **Motion quality**: Is movement smooth or jittery?
- **Prompt adherence**: Does the video match the text description?
- **Artifacts**: Flickering, morphing faces, impossible physics

## Performance Metrics

```python
import time
start = time.perf_counter()
video = pipe(prompt, num_frames=num_frames, num_inference_steps=steps)
elapsed = time.perf_counter() - start
sec_per_frame = elapsed / num_frames
```

Key metrics:
- **sec/frame**: generation time per frame
- **Total generation time**: wall-clock for full video
- **VRAM peak**: monitor with `torch.cuda.max_memory_allocated()`

## Common Issues

- **OOM**: Video models are huge. May need A100-80GB or H100.
- **Slow generation**: 10+ minutes per clip is normal. Set long timeouts.
- **Frame count**: Some models only support fixed frame counts (e.g., 16, 24).
- **Resolution**: Many models output 512x512 or 768x512. Check capabilities.
- **Export format**: Use `export_to_video` from diffusers or ffmpeg.
