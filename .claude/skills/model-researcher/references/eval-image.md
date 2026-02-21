# Image Generation Evaluation Guide

## Smoke Test
- Load model with `diffusers` (appropriate pipeline class)
- Generate one image with a simple prompt: "A red cat sitting on a chair"
- Verify output is a valid PIL Image / PNG file

## Quality Check — Test Prompts

Run at least 3 prompts covering different capabilities:

1. **Photorealistic**: "Professional portrait photo of a woman, natural lighting, 4K"
2. **Artistic**: "Oil painting of a sunset over mountains, impressionist style"
3. **Text rendering**: "A storefront sign that reads OPEN 24 HOURS"
4. **Complex scene**: "An astronaut riding a horse on Mars, with Earth visible in the sky"

Evaluate each image for (Claude can see images):
- Overall quality and coherence
- Prompt faithfulness
- Detail level and artifacts
- Text rendering accuracy (if applicable)
- Hand/finger quality (common failure mode)

## Performance Measurement

```python
import time

# Warmup
pipe("test", num_inference_steps=1)

# Measure (use model's default step count)
times = []
for _ in range(3):
    start = time.perf_counter()
    image = pipe(prompt, num_inference_steps=DEFAULT_STEPS).images[0]
    elapsed = time.perf_counter() - start
    times.append(elapsed)

sec_per_image = sorted(times)[len(times) // 2]  # median
```

Key metrics:
- **sec/image** (median of 3 runs at default steps)
- **resolution** of generated images
- **num_inference_steps** used

## Common Issues
- OOM: Use `pipe.enable_model_cpu_offload()` or reduce resolution
- Black images: Check safety checker; some models have NSFW filters
- Low quality: Ensure correct scheduler; some models need specific schedulers
- Slow: Use `torch.compile(pipe.unet)` if supported
