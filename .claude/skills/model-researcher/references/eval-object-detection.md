# Object Detection / Segmentation Evaluation Guide

## Target Models

YOLOv11, SAM 2, DETR, Grounding DINO, RT-DETR.

## Setup

### YOLO (ultralytics)
```python
from ultralytics import YOLO
model = YOLO(MODEL_ID)  # e.g., "yolov11n.pt" or HF model ID
```

### DETR / Grounding DINO (transformers)
```python
from transformers import AutoProcessor, AutoModelForObjectDetection
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to("cuda")
```

### SAM 2 (segment-anything)
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID)
```

## Test Image Preparation

Agent should prepare test images:
- Simple scene: 1-3 objects, clear background
- Crowded scene: many objects, overlapping
- Small objects: test detection sensitivity
- Domain-specific: if the model targets a specific domain

Use copyright-free images (Unsplash, generated images, or standard datasets).

## Smoke Test

```python
# YOLO
results = model("test.jpg")
assert len(results[0].boxes) > 0, "No detections"
results[0].save("output.jpg")

# DETR
from PIL import Image
image = Image.open("test.jpg")
inputs = processor(images=image, return_tensors="pt").to("cuda")
outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.5
)
assert len(results[0]["labels"]) > 0
```

## Quality Evaluation

### Visualization (Critical)
**Always** save images with bounding boxes/masks overlaid:

```python
# YOLO — automatic
results[0].save("output_detected.jpg")

# DETR — manual overlay
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)
ax.imshow(image)
for score, label, box in zip(
    results[0]["scores"], results[0]["labels"], results[0]["boxes"]
):
    x0, y0, x1, y1 = box.tolist()
    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2,
                              edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x0, y0, f"{model.config.id2label[label.item()]}: {score:.2f}",
            color='red', fontsize=8)
plt.savefig("output_detected.png", bbox_inches='tight')
```

### Test Cases
1. Common objects (people, cars, animals)
2. Small objects (far away, partially hidden)
3. Overlapping objects (crowded scenes)
4. Zero-shot (for Grounding DINO: text-specified target objects)
5. Segmentation quality (for SAM: mask precision)

## Performance Metrics

```python
import time
start = time.perf_counter()
results = model("test.jpg")
elapsed = time.perf_counter() - start
fps = 1.0 / elapsed
```

Key metrics:
- **FPS**: frames per second (critical for real-time use cases)
- **mAP**: mean Average Precision (if using COCO-style evaluation)
- **Detection count**: number of objects found per image
- **Confidence threshold**: report how results change at different thresholds

## Common Issues

- **ultralytics version**: YOLO versions change rapidly. Ensure latest ultralytics.
- **Model size variants**: YOLOv11n (nano) vs YOLOv11x (extra-large) — big perf difference.
- **Threshold tuning**: Default threshold may miss objects or produce false positives.
- **COCO categories**: Standard models detect 80 COCO categories. Domain models differ.
