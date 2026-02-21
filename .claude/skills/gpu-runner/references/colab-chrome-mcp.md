# Google Colab via Chrome MCP

## Overview

Control Google Colab notebooks through Chrome MCP browser automation.
Colab Pro provides T4 (16GB), V100 (16GB), A100 (40GB) GPUs.

## Setup

1. Open Chrome with Chrome MCP extension connected
2. Navigate to `https://colab.research.google.com/`
3. Create a new notebook or open an existing one

## GPU Runtime Selection

1. Click "Runtime" menu > "Change runtime type"
2. Select GPU type:
   - **T4**: Default, good for models up to ~7B
   - **A100**: For models 7B-30B (may not always be available)
3. Click "Save"
4. Verify GPU: Run cell with `!nvidia-smi`

## Notebook Operations via Chrome MCP

### Create a new cell
- Click "+ Code" button, or use keyboard shortcut

### Write code in a cell
- Click on the cell to select it
- Type or paste code

### Run a cell
- Click the play button on the cell, or Shift+Enter
- Wait for the cell execution indicator to complete (spinner stops)

### Read cell output
- Read the output area below the executed cell

## Typical Workflow

```
1. Navigate to colab.research.google.com
2. Create new notebook
3. Set GPU runtime (A100 if available, otherwise T4)
4. Cell 1: Install dependencies
   !pip install transformers accelerate torch
5. Cell 2: Load model
   from transformers import AutoModelForCausalLM, AutoTokenizer
   ...
6. Cell 3: Run inference
   ...
7. Cell 4: Measure performance
   ...
8. Download results from Colab
```

## Downloading Results

### Files generated in Colab
```python
# In Colab cell
from google.colab import files
files.download('output.png')
```

### Or save to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy files to Drive
```

## Common Issues

- **GPU unavailable**: Try again later, or switch to Modal
- **Session timeout**: Colab Pro gives longer sessions (~24h) but still times out
- **Disk space**: `/content/` has ~100GB, use it for model cache
- **Memory**: If OOM, restart runtime and try with quantization
- **Slow install**: Use `%pip install` instead of `!pip install` for better caching
