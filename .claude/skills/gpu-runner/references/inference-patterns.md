# Inference Patterns by Model Type

Quick-reference code snippets for writing inference scripts per model type.
The agent should adapt these patterns based on each model's README/model card.

## LLM (text-generation)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
start = time.perf_counter()
output = model.generate(**inputs, max_new_tokens=256)
elapsed = time.perf_counter() - start

text = tokenizer.decode(output[0], skip_special_tokens=True)
new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
print(f"tokens/sec: {new_tokens / elapsed:.1f}")
```

**Alternatives**: vLLM (high-throughput serving), llama.cpp (GGUF quantized), TGI (HF serving).

## VLM (vision-language)

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch, time

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

image = Image.open("test.png")
messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": "Describe this image."}
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

start = time.perf_counter()
output = model.generate(**inputs, max_new_tokens=256)
elapsed = time.perf_counter() - start

result = processor.decode(output[0], skip_special_tokens=True)
```

**Note**: Chat template varies by model (Qwen2.5-VL, InternVL, LLaVA each differ).
Always check the model card for the correct chat format.

## Code Generation

Same as LLM, but test Fill-in-the-Middle (FIM) if supported:

```python
# FIM tokens vary by model:
# - StarCoder/DeepSeek: <fim_prefix>, <fim_suffix>, <fim_middle>
# - Qwen2.5-Coder: <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>
fim_prompt = f"{fim_prefix}def fibonacci(n):\n{fim_suffix}\n    return fib(n-1) + fib(n-2){fim_middle}"
```

**Verify**: Execute generated code in a subprocess to check correctness.

## Image Generation (diffusers)

```python
from diffusers import DiffusionPipeline
import torch, time

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda")

start = time.perf_counter()
image = pipe("A cat wearing sunglasses", num_inference_steps=30).images[0]
elapsed = time.perf_counter() - start

image.save("output.png")
print(f"sec/image: {elapsed:.1f}")
```

**Alternatives**: ComfyUI (node-based), Automatic1111 WebUI. For FLUX models,
check if `FluxPipeline` is needed instead of generic `DiffusionPipeline`.

## TTS (text-to-speech)

Varies heavily by model. Common patterns:

```python
# Pattern A: transformers pipeline
from transformers import pipeline
tts = pipeline("text-to-speech", model=MODEL_ID, device="cuda")
output = tts("Hello world")
import soundfile as sf
sf.write("output.wav", output["audio"][0], output["sampling_rate"])

# Pattern B: model-specific (e.g., Bark)
from transformers import AutoProcessor, AutoModel
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to("cuda")
inputs = processor("Hello world", return_tensors="pt").to("cuda")
output = model.generate(**inputs)

# Pattern C: dedicated library (e.g., Coqui TTS, Fish Speech)
# Always check the model card for the recommended approach.
```

**Key metric**: RTF (Real-Time Factor) = generation_time / audio_duration.

## STT (speech recognition)

```python
# Whisper-family (most common)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch, time

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)

# Load audio (16kHz mono)
import librosa
audio, sr = librosa.load("test.wav", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(model.device)

start = time.perf_counter()
output = model.generate(**inputs, max_new_tokens=256)
elapsed = time.perf_counter() - start

text = processor.batch_decode(output, skip_special_tokens=True)[0]
```

**Alternatives**: faster-whisper (CTranslate2 backend, 4x speed), whisper.cpp.

## Embedding

```python
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer(MODEL_ID, device="cuda")

sentences = ["query: What is ML?", "passage: Machine learning is..."]
start = time.perf_counter()
embeddings = model.encode(sentences, normalize_embeddings=True)
elapsed = time.perf_counter() - start

similarity = embeddings[0] @ embeddings[1]
print(f"Similarity: {similarity:.4f}, embeddings/sec: {len(sentences)/elapsed:.1f}")
```

**Note**: Some models require instruction prefixes (e.g., `query:` / `passage:`).
Check model card. Dimension may be configurable via `truncate_dim`.

## Time Series

```python
# Chronos (Amazon) — most common foundation model
from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
# context: 1D tensor of historical values
forecast = pipeline.predict(context, prediction_length=24, num_samples=20)
# forecast shape: (num_samples, prediction_length)
median = forecast.median(dim=0).values

# TimesFM (Google) — different API
from timesfm import TimesFm
model = TimesFm.load_from_checkpoint(repo_id=MODEL_ID)
forecast = model.forecast(input_ts, freq=[0])
```

**Note**: Time series models have very diverse APIs. Always read the model card.

## Audio / Music Generation

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch, time, soundfile as sf

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to("cuda")

inputs = processor(text=["happy rock song"], padding=True, return_tensors="pt").to("cuda")
start = time.perf_counter()
audio = model.generate(**inputs, max_new_tokens=512)
elapsed = time.perf_counter() - start

sf.write("output.wav", audio[0, 0].cpu().numpy(), model.config.audio_encoder.sampling_rate)
```

## Video Generation

```python
# diffusers-based (Wan2.1, CogVideoX, etc.)
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
# Most video models need specific pipeline class — check model card
video = pipe("A cat playing piano", num_frames=16, num_inference_steps=25).frames[0]

# Save as mp4
from diffusers.utils import export_to_video
export_to_video(video, "output.mp4", fps=8)
```

**Warning**: Video generation is very VRAM-intensive (24-80GB+) and slow.
Budget 30+ minutes per benchmark run.

## Object Detection / Segmentation

```python
# YOLO (ultralytics)
from ultralytics import YOLO
model = YOLO(MODEL_ID)  # e.g., "yolov11n.pt"
results = model("test.jpg")
results[0].save("output.jpg")  # saves with bounding boxes

# DETR / Grounding DINO (transformers)
from transformers import AutoProcessor, AutoModelForObjectDetection
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to("cuda")
inputs = processor(images=image, return_tensors="pt").to("cuda")
outputs = model(**inputs)

# SAM 2 (segment-anything)
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID)
```

**Key metric**: mAP (mean Average Precision), FPS for real-time applications.

## 3D Generation

```python
# TripoSR / InstantMesh — image-to-3D
import torch
from PIL import Image

# Most 3D models have custom pipelines — always check the model card
model = torch.hub.load(MODEL_ID, "model").to("cuda")
image = Image.open("input.png")
mesh = model(image)
mesh.export("output.glb")  # or .obj
```

**Note**: 3D generation models have the most diverse APIs.
Always read the model README carefully. Consider embedding a three.js viewer
in the HTML report for interactive 3D visualization.

---

## HF Inference API (all types)

For models available via HF Inference API, use `InferenceClient`:

```python
from huggingface_hub import InferenceClient
client = InferenceClient(token=os.environ["HF_TOKEN"])

# The client auto-detects task from the model's pipeline_tag
client.text_generation("Hello", model=MODEL_ID)
client.text_to_image("A cat", model=MODEL_ID)
client.text_to_speech("Hello", model=MODEL_ID)
client.automatic_speech_recognition("audio.wav", model=MODEL_ID)
client.feature_extraction("Hello world", model=MODEL_ID)
client.object_detection("image.jpg", model=MODEL_ID)
client.image_to_text("image.jpg", model=MODEL_ID)  # VLM
```

This is always the simplest option when available. Check availability with
`hf_inference_check.py` first.

---

## General Principles

1. **Always read the model card first** — it has the correct import and usage
2. **Use bfloat16/float16** — never load in fp32 on GPU
3. **Set device_map="auto"** — for models > 1 GPU worth of VRAM
4. **Time everything** — wrap inference in `time.perf_counter()` pairs
5. **Save all outputs** — even errors are valuable for the report
6. **Handle dependencies** — each model may need specific packages (librosa,
   soundfile, ultralytics, etc.). Install them in the execution environment.
