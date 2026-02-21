# RunPod GPU Execution Guide

## Overview

RunPod provides Pods (persistent GPU VMs) and Serverless endpoints.
Good balance of price, reliability, and developer experience.
Per-second billing. No persistent free tier.

## Authentication

Environment variable (loaded from `.env`):
```
RUNPOD_API_KEY=rpa_xxxxxxxx
```

Get your key from https://www.runpod.io/console/user/settings (API Keys section).

## Python SDK

```bash
pip install runpod
```

## Execution Pattern A: Pod (Recommended for Benchmarks)

Pods are persistent GPU VMs with SSH/Jupyter access.

```python
import runpod
import os
import time

runpod.api_key = os.environ["RUNPOD_API_KEY"]

# 1. Create a pod
pod = runpod.create_pod(
    name="agentic-bench-run",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_type_id="NVIDIA GeForce RTX 4090",
    gpu_count=1,
    volume_in_gb=20,
    container_disk_in_gb=20,
    env={
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    },
)
pod_id = pod["id"]
print(f"Pod created: {pod_id}")

# 2. Wait for pod to be ready
while True:
    status = runpod.get_pod(pod_id)
    if status.get("desiredStatus") == "RUNNING" and status.get("runtime"):
        break
    time.sleep(10)

# 3. Execute commands via SSH or use runpodctl for file transfer
# SSH: ssh root@{pod_ip} -p {pod_port} -i ~/.ssh/id_rsa
# Files: runpodctl send/receive for peer-to-peer transfer

# 4. Terminate pod when done
runpod.terminate_pod(pod_id)
```

## Execution Pattern B: Serverless (For Repeated Inference)

Serverless auto-scales workers. Requires a pre-built Docker image.

```python
import runpod

runpod.api_key = "YOUR_API_KEY"

# Call an existing serverless endpoint
endpoint = runpod.Endpoint("ENDPOINT_ID")

# Synchronous call (waits for result)
result = endpoint.run_sync(
    {"prompt": "Hello, world!"},
    timeout=120,
)

# Or asynchronous
run = endpoint.run({"prompt": "Hello, world!"})
status = run.status()  # "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED"
output = run.output()  # blocks until complete
```

## GPU Types (Pod Pricing)

| GPU | VRAM | Price/hr |
|-----|------|----------|
| RTX 4090 | 24GB | $0.34 |
| L4 | 24GB | $0.44 |
| A100 PCIe 80GB | 80GB | $1.19 |
| A100 SXM 80GB | 80GB | $1.39 |
| H100 SXM 80GB | 80GB | $2.69 |
| H200 141GB | 141GB | $3.59 |

Community Cloud is cheaper; Secure Cloud costs 20-30% more.

## GPU Type IDs for create_pod()

```python
"NVIDIA GeForce RTX 4090"
"NVIDIA L4"
"NVIDIA A100-SXM4-80GB"
"NVIDIA A100 80GB PCIe"
"NVIDIA H100 80GB HBM3"
```

## Common Issues

- **Pod not starting**: Check GPU availability. Try a different GPU type or region.
- **SSH access**: Pods expose SSH on a random port. Check `pod["runtime"]["ports"]`.
- **File transfer**: Use `runpodctl send`/`receive` (pre-installed on pods) for easy transfer.
- **Templates**: Use RunPod's PyTorch template for pre-installed ML dependencies.
- **Volume persistence**: Pod volumes persist across restarts but not across termination.
