# 3D Generation Evaluation Guide

## Target Models

TripoSR, InstantMesh, Point-E, Shap-E, LGM.

## Setup

Each 3D model has a unique API. Always check the model card.

### TripoSR
```python
import torch
from tsr.system import TSR
model = TSR.from_pretrained("stabilityai/TripoSR", device="cuda")
```

### Shap-E (OpenAI)
```python
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
model = load_model("text300M", device="cuda")
diffusion = diffusion_from_config(load_config("diffusion"))
```

## Smoke Test

```python
from PIL import Image

# Image-to-3D (most common)
image = Image.open("input.png")
mesh = model(image)  # API varies by model
mesh.export("output.glb")  # or .obj

import os
assert os.path.getsize("output.glb") > 1000, "Output mesh is too small"
```

## Quality Evaluation (Visual Inspection)

### HTML Report with 3D Viewer

Embed an interactive 3D viewer using three.js / model-viewer:

```html
<script type="module"
  src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js">
</script>

<model-viewer
  src="artifacts/output.glb"
  alt="Generated 3D model"
  auto-rotate camera-controls
  style="width: 100%; height: 400px;">
</model-viewer>
```

If three.js embedding is not feasible, render multiple viewpoints:
```python
# Render from 4 angles and save as images
for angle in [0, 90, 180, 270]:
    rendered = render_mesh(mesh, azimuth=angle)
    rendered.save(f"view_{angle}.png")
```

### Test Cases
1. Simple object (single item, clear shape)
2. Complex object (multiple parts, fine details)
3. Text-to-3D prompt (if supported): "a red sports car"
4. Image-to-3D: photo of a real object
5. Geometry quality: check for holes, inverted normals, non-manifold edges

## Performance Metrics

```python
import time
start = time.perf_counter()
mesh = model(input_image)
elapsed = time.perf_counter() - start
```

Key metrics:
- **Generation time**: seconds per mesh
- **Vertex count**: mesh complexity
- **File size**: .glb/.obj output size

## Common Issues

- **Diverse APIs**: No standard interface. Each model is unique.
- **Dependencies**: trimesh, pygltflib, pytorch3d may be needed.
- **Export format**: .glb (binary glTF) is most portable for HTML embedding.
- **Quality assessment**: Primarily visual. No widely-accepted automatic metric.
- **Background removal**: Image-to-3D models often need clean background input.
  Use rembg or SAM to preprocess if needed.
