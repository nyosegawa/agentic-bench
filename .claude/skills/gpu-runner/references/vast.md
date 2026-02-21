# Vast.ai GPU Execution Guide

## Overview

Vast.ai is a GPU marketplace — hosts list GPUs at competitive prices.
Cheapest option for A100/H100. Docker-based with SSH access.
Per-second billing. No free tier.

## Authentication

Environment variable (loaded from `.env`):
```
VAST_API_KEY=your_api_key
```

Get your key from https://cloud.vast.ai/cli/

## Python SDK

```bash
pip install vastai-sdk
```

## Execution Pattern (Instance-Based)

Vast.ai uses Docker containers with SSH. The workflow is:
1. Search for a GPU offer
2. Create instance with onstart script
3. Wait for completion
4. Download results via SCP
5. Destroy instance immediately (storage charges continue otherwise!)

```python
import json
import time
from vastai_sdk import VastAI

vast = VastAI(api_key="YOUR_API_KEY")

# 1. Search for offers
offers = vast.search_offers(
    query="gpu_name=A100_SXM4 reliability>0.95 num_gpus=1 rentable=True rented=False"
)
offers_data = json.loads(offers) if isinstance(offers, str) else offers
offer_id = offers_data[0]["id"]

# 2. Create instance
result = vast.create_instance(
    ID=offer_id,
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    disk=30,
    runtype="ssh",
    onstart="pip install transformers accelerate && python /workspace/run.py",
    env="-e HF_TOKEN=xxx",
)
instance_id = result.get("new_contract")

# 3. Wait for job completion (poll logs)
while True:
    time.sleep(30)
    logs = vast.logs(INSTANCE_ID=instance_id)
    if "BENCHMARK_COMPLETE" in str(logs):
        break

# 4. Download results
vast.copy(
    src=f"{instance_id}:/workspace/results/",
    dst="./results/",
    identity="~/.ssh/id_rsa",
)

# 5. ALWAYS destroy when done — storage charges continue on stopped instances
vast.destroy_instance(id=instance_id)
```

## GPU Search Query Syntax

```python
# Filter examples
"gpu_name=RTX_4090 reliability>0.99 num_gpus>=1"
"gpu_name=A100_SXM4 cuda_vers>=12.0 rented=False"
"gpu_ram>=80 dph_total<2.0"  # 80GB+ VRAM, under $2/hr
```

## Available GPUs (typical marketplace prices)

| GPU | VRAM | Approx. Price/hr |
|-----|------|-------------------|
| T4 | 16GB | $0.05–0.15 |
| RTX 4090 | 24GB | $0.10–0.25 |
| A100-40GB | 40GB | $0.52–1.00 |
| A100-80GB | 80GB | $0.80–1.50 |
| H100 SXM | 80GB | $1.49–2.50 |

## Common Issues

- **Storage billing**: Charges continue on stopped instances. Always `destroy_instance()`.
- **Host reliability**: Filter with `reliability>0.95`. Low-reliability hosts may drop.
- **Cold start**: 1–5 min for Docker image pull. Use small base images.
- **SSH key**: Must have SSH key set up. Use `vast.create_ssh_key()` if needed.
- **Disk sizing**: Cannot be resized after creation. Size correctly upfront.
