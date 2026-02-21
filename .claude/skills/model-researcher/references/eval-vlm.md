# VLM (Vision-Language Model) Evaluation Guide

## Representative Models
- Qwen2.5-VL (3B-72B), InternVL2.5/3, Gemma 3, SmolVLM2
- Llama 3.2 Vision, PaliGemma 2, Kimi-VL, moondream2
- Specialized: RolmOCR (document OCR), ColPali (document retrieval)

## Framework
All major VLMs use `transformers` with `AutoProcessor`:
```python
from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
```

For chat-style VLMs (Qwen2.5-VL, InternVL):
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
```

## Smoke Test
- Load model + processor
- Pass a simple image (e.g., photo of a cat) with prompt "What is in this image?"
- Verify text output is coherent and relevant

## Quality Check — Test Inputs

Run at least 4 diverse tests:

1. **Object description**: Photo of a room. "Describe everything you see."
   - Check: Accurate object identification, no hallucinated objects
2. **OCR / text reading**: Image with text (sign, document).
   "Read all the text in this image."
   - Check: Accurate text extraction
3. **Reasoning**: Chart or diagram. "What trend does this chart show?"
   - Check: Correct interpretation of data
4. **Spatial understanding**: Complex scene. "What is to the left of the person?"
   - Check: Correct spatial relationships

### Critical: Hallucination Testing
VLMs frequently hallucinate. Test specifically:
- Show an image of 3 dogs, ask "How many cats are in this image?"
  - Correct answer: "There are no cats" (not "I see 3 cats")
- Ask about objects NOT in the image
- Ask about text NOT present in the image

## Performance Measurement

```python
import time
from PIL import Image

image = Image.open("test.jpg")
inputs = processor(text="Describe this image.", images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Warmup
model.generate(**inputs, max_new_tokens=10)

times = []
for _ in range(5):
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=128)
    elapsed = time.perf_counter() - start
    tokens = output.shape[1] - inputs["input_ids"].shape[1]
    times.append(tokens / elapsed)

tokens_per_sec = sorted(times)[len(times) // 2]
```

Key metrics: tokens/sec, latency p50/p99

## Standard Benchmarks (for reference)
- **MMMU**: Multi-discipline understanding (primary)
- **MMBench-EN**: General VQA
- **DocVQA / OCRBench**: Document understanding
- **TextVQA**: Scene text reading

## Common Issues
- **Hallucination**: The #1 problem. Always test for it explicitly.
- **Image resolution**: Most VLMs tile images. Very large images consume many tokens.
- **Chat template**: Many VLMs require specific chat templates; check model card.
- **Multi-image**: Not all VLMs support multiple images. Check before testing.
- **Video input**: Qwen2.5-VL, InternVL3, SmolVLM2 support video; others may not.
