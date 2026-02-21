# RunPod Deep Dive Research (2026-02)

**Official Documentation**: https://docs.runpod.io/
**Pricing Page**: https://www.runpod.io/pricing
**Python SDK (GitHub)**: https://github.com/runpod/runpod-python

## 1. Pricing Model

### Billing Structure
- **Per-second billing** (partial seconds rounded up to next full second)
- No ingress/egress fees
- Spend limit default: $80/hour across all resources

### Pod Pricing (On-Demand, per hour)

| GPU | VRAM | On-Demand | 6-month commit | 1-year commit |
|-----|------|-----------|----------------|---------------|
| RTX A5000 | 24 GB | $0.16/hr | $0.21/hr | $0.20/hr |
| RTX 3090 | 24 GB | $0.22/hr | $0.36/hr | $0.34/hr |
| RTX 4090 | 24 GB | $0.34/hr | $0.51/hr | $0.50/hr |
| L4 | 24 GB | $0.44/hr | $0.33/hr | $0.32/hr |
| RTX 5090 | 32 GB | $0.69/hr | $0.77/hr | $0.76/hr |
| A40 | 48 GB | $0.35/hr | $0.28/hr | $0.20/hr |
| RTX A6000 | 48 GB | $0.33/hr | $0.42/hr | $0.40/hr |
| RTX 6000 Ada | 48 GB | $0.74/hr | $0.66/hr | $0.63/hr |
| L40 | 48 GB | $0.69/hr | $0.84/hr | $0.81/hr |
| L40S | 48 GB | $0.79/hr | $0.73/hr | $0.71/hr |
| A100 PCIe | 80 GB | $1.19/hr | $1.18/hr | $1.14/hr |
| A100 SXM | 80 GB | $1.39/hr | $1.27/hr | $1.22/hr |
| H100 PCIe | 80 GB | $1.99/hr | $2.08/hr | $2.03/hr |
| H100 NVL | 94 GB | $2.59/hr | $2.67/hr | $2.61/hr |
| H100 SXM | 80 GB | $2.69/hr | - | - |
| RTX Pro 6000 | 96 GB | $1.69/hr | $1.64/hr | $1.61/hr |
| H200 | 141 GB | $3.59/hr | $3.12/hr | $3.05/hr |
| B200 | 180 GB | $5.98/hr | $4.34/hr | $4.24/hr |

Note: T4 is NOT listed in current pod pricing (likely phased out for pods, available only in serverless).

### Serverless Pricing (per-second billing)

| GPU | VRAM | Flex Worker (/sec) | Active Worker (/sec) | Flex ~$/hr | Active ~$/hr |
|-----|------|--------------------|----------------------|------------|--------------|
| A4000/A4500/RTX 4000 | 16 GB | $0.00016 | $0.00011 | $0.58 | $0.40 |
| L4/A5000/3090 | 24 GB | $0.00019 | $0.00013 | $0.68 | $0.47 |
| 4090 PRO | 24 GB | $0.00031 | $0.00021 | $1.12 | $0.76 |
| A6000/A40 | 48 GB | $0.00034 | $0.00024 | $1.22 | $0.86 |
| L40/L40S/6000 Ada | 48 GB | $0.00053 | $0.00037 | $1.91 | $1.33 |
| A100 | 80 GB | $0.00076 | $0.00060 | $2.74 | $2.16 |
| H100 PRO | 80 GB | $0.00116 | $0.00093 | $4.18 | $3.35 |
| H200 PRO | 141 GB | $0.00155 | $0.00124 | $5.58 | $4.46 |
| B200 | 180 GB | $0.00240 | $0.00190 | $8.64 | $6.84 |

- **Flex Workers**: Scale up/down with traffic, idle after jobs complete
- **Active Workers**: Always-on, 20-30% cheaper than flex
- **FlashBoot**: Cold start optimization included at no extra cost

### Storage Pricing
- Container Disk: $0.10/GB/month
- Volume Disk (Running): $0.10/GB/month
- Volume Disk (Idle): $0.20/GB/month
- Network Storage (<1TB): $0.07/GB/month
- Network Storage (>1TB): $0.05/GB/month
- High-Performance Network (<1TB): $0.14/GB/month
- No ingress/egress fees

---

## 2. Authentication

### API Key
- Created in RunPod dashboard under Settings > API Keys
- Environment variable: `RUNPOD_API_KEY`
- Key prefix: `rpa_`
- Scoped keys available: All, Restricted, Read Only
- Fine-grained scoping by endpoint

### Usage in Python SDK
```python
import runpod
import os

# Recommended: environment variable
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# Or per-endpoint override
endpoint = runpod.Endpoint("ENDPOINT_ID", api_key="specific_key")
```

### Usage in REST API
```bash
curl https://rest.runpod.io/v1/pods \
  --header 'Authorization: Bearer YOUR_API_KEY'
```

### Usage in CLI
```bash
runpodctl config --apiKey=YOUR_API_KEY
```

---

## 3. Python SDK and CLI

### Installation
```bash
pip install runpod       # stable from PyPI
uv add runpod            # via uv
# or development version:
pip install git+https://github.com/runpod/runpod-python.git
```
Requires Python 3.8+

### Pod Management (Python SDK)
```python
import runpod
runpod.api_key = "your_api_key"

# Create a pod
pod = runpod.create_pod(
    "my-inference-pod",           # name
    "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu",  # image
    "NVIDIA GeForce RTX 4090"     # GPU type string
)

# List all pods
pods = runpod.get_pods()

# Get specific pod
pod_info = runpod.get_pod(pod.id)

# Stop (preserves state, billing pauses)
runpod.stop_pod(pod.id)

# Resume
runpod.resume_pod(pod.id)

# Terminate (destroy permanently)
runpod.terminate_pod(pod.id)
```

### Serverless Endpoint Interaction (Python SDK)
```python
import runpod
import os

runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Synchronous (blocks until result, default 90s timeout)
result = endpoint.run_sync(
    {"prompt": "Hello, world!"},
    timeout=60,
)

# Asynchronous (returns immediately, poll for result)
run_request = endpoint.run({"prompt": "Hello, World!"})
status = run_request.status()        # "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED"
output = run_request.output(timeout=60)

# Streaming
for chunk in run_request.stream():
    print(chunk)

# Cancel
run_request.cancel()

# Health check
health = endpoint.health()

# Purge all queued jobs
endpoint.purge_queue()
```

### Async with asyncio
```python
import asyncio
import aiohttp
import runpod
from runpod import AsyncioEndpoint

async def main():
    async with aiohttp.ClientSession() as session:
        endpoint = AsyncioEndpoint("YOUR_ENDPOINT_ID", session)
        job = await endpoint.run({"prompt": "Hello"})
        while True:
            status = await job.status()
            if status == "COMPLETED":
                output = await job.output()
                break
            elif status == "FAILED":
                break
            await asyncio.sleep(3)

asyncio.run(main())
```

### File Transfer (CLI)
```bash
# Install CLI
wget -qO- cli.runpod.net | sudo bash
# or
brew install runpod/runpodctl/runpodctl

# Configure
runpodctl config --apiKey=YOUR_API_KEY

# Pod management
runpodctl get pod                     # list all pods
runpodctl get pod {podId}             # get specific pod
runpodctl start pod {podId}           # start pod
runpodctl stop pod {podId}            # stop pod

# File transfer (peer-to-peer, no API key needed)
# On source machine:
runpodctl send data.txt               # outputs a receive code
# On destination (e.g., inside a pod):
runpodctl receive {code}
```

Note: `runpodctl` is pre-installed on all RunPod pods with a pod-scoped API key.

---

## 4. Serverless vs Pods

### Pods
- **What**: Persistent GPU instances (virtual machines with Docker containers)
- **Billing**: Per-second while running; stopped pods only pay for storage
- **Use when**:
  - Training / fine-tuning models
  - Interactive development (Jupyter, SSH, VS Code)
  - Long-running simulations
  - Debugging and experimentation
  - Need persistent filesystem state
- **Features**: SSH access, Jupyter Lab, persistent storage, full root access

### Serverless
- **What**: Auto-scaling endpoint workers that process jobs via a queue
- **Billing**: Per-second for execution time + start time + idle time (default 5s idle)
- **Use when**:
  - Production inference APIs
  - Bursty/unpredictable traffic (scale to zero when idle)
  - Cost optimization (pay only during active use)
  - Building products/apps that serve users
- **Features**: Auto-scaling, FlashBoot (1-2s cold start), queue-based job management
- **Cost advantage**: Up to 80% savings vs always-on pods for inference workloads

### Hybrid
Smart approach: Use pods for development/training, serverless for production inference.

---

## 5. API Endpoints

### REST API (New, 2025+)
Base URL: `https://rest.runpod.io/v1`
Interactive docs: `https://rest.runpod.io/v1/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /pods | Create a new pod |
| GET | /pods | List all pods |
| POST | /pods/{pod-id}/stop | Stop a pod |
| POST | /endpoints | Create serverless endpoint |
| GET | /endpoints | List endpoints |

### GraphQL API (Legacy but still functional)
Endpoint: `https://api.runpod.io/graphql`
Spec: `https://graphql-spec.runpod.io/`

### Serverless Job REST API
Base URL: `https://api.runpod.ai/v2/{ENDPOINT_ID}`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /run | Submit async job (returns job ID) |
| POST | /runsync | Submit sync job (blocks until done, 90s default) |
| GET | /status/{JOB_ID} | Check job status |
| GET | /stream/{JOB_ID} | Stream incremental results |
| POST | /cancel/{JOB_ID} | Cancel a job |
| POST | /retry/{JOB_ID} | Retry a failed job |
| POST | /purge-queue | Remove all pending jobs |
| GET | /health | Endpoint health/worker stats |

Auth header for all: `Authorization: YOUR_RUNPOD_API_KEY`

Example:
```bash
# Submit async job
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "authorization: $RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{"input": {"prompt": "Hello, world!"}}'

# Check status
curl https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID \
  -H "authorization: $RUNPOD_API_KEY"
```

---

## 6. Free Tier / Credits

- **No persistent free tier** (unlike Modal's $30/month)
- Sign-up bonus: Spend first $10, receive random bonus between $5-$500 (non-transferable, expires 90 days)
- Referral program: Both parties get $5-$500 after referee spends $10
- Startup program: Apply for credits based on projected GPU usage (requires upfront investment commitment)

---

## 7. Unique Features vs Modal / beam.cloud

| Feature | RunPod | Modal | beam.cloud |
|---------|--------|-------|------------|
| Billing | Per-second | Per-second | Per-second |
| Free tier | $5-500 one-time bonus | $30/month recurring | ~10-15hr one-time |
| GPU variety | 20+ GPU types | ~8 types | ~6 types |
| Pods (persistent VMs) | Yes | No (serverless only) | No (serverless only) |
| Serverless | Yes | Yes | Yes |
| Cold start | FlashBoot 1-2s | ~5-10s | Sub-second (checkpoint restore) |
| Community cloud | Yes (cheaper, shared infra) | No | No |
| Secure cloud | Yes (dedicated infra) | Default | Default |
| Pre-built templates | Yes (Template Gallery) | No (build from code) | No |
| Docker-first | Yes | No (Python-native) | Python SDK |
| SSH to pods | Yes | No | No |
| Jupyter Lab | Built-in on pods | No | No |
| Open source runtime | No | No | Yes |
| Multi-GPU clusters | Yes (up to 64 GPUs) | Limited | Limited |
| File transfer CLI | Yes (runpodctl) | modal volume | No built-in |
| DX quality | Good (Docker-centric) | Best (Python-native) | Good |
| Self-hosted option | No | No | Yes |

### RunPod's key differentiators:
1. **Broadest GPU selection** (20+ types including B200, RTX 5090)
2. **Dual model**: Both persistent pods and serverless (others are serverless-only)
3. **Community cloud**: Cheaper pricing via shared infrastructure
4. **Pre-built template gallery**: One-click deploy for common ML stacks
5. **FlashBoot**: 1-2 second cold starts for serverless
6. **SSH + Jupyter**: Full interactive access on pods
7. **Multi-GPU clusters**: Up to 64 GPUs for distributed training

---

## 8. Common Pattern for ML Inference (One-Off Job)

### Option A: Serverless Endpoint (Recommended for repeated inference)

**Step 1**: Create a worker handler (`handler.py`):
```python
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

def handler(job):
    global model, tokenizer
    if model is None:
        load_model()

    input_data = job["input"]
    prompt = input_data["prompt"]
    max_tokens = input_data.get("max_tokens", 512)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": result}

runpod.serverless.start({"handler": handler})
```

**Step 2**: Dockerfile:
```dockerfile
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
WORKDIR /app
RUN pip install transformers accelerate
COPY handler.py /app/
CMD ["python3", "-u", "handler.py"]
```

**Step 3**: Build, push, deploy:
```bash
docker build --platform linux/amd64 -t myuser/my-inference:latest .
docker push myuser/my-inference:latest
# Then create endpoint in RunPod console or via API
```

**Step 4**: Submit inference job:
```python
import runpod
import os

runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

result = endpoint.run_sync(
    {"prompt": "Explain quantum computing in one paragraph.", "max_tokens": 256},
    timeout=120,
)
print(result)
```

### Option B: Pod-Based (For one-off or exploratory work)

```python
import runpod
import os

runpod.api_key = os.getenv("RUNPOD_API_KEY")

# Create pod with PyTorch template
pod = runpod.create_pod(
    "one-off-inference",
    "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu",
    "NVIDIA A100 80GB PCIe"
)
print(f"Pod created: {pod.id}")

# SSH in, run your inference script, download results
# Then destroy:
runpod.terminate_pod(pod.id)
```

### Option C: REST API One-Off Job

```bash
# Create and run in one curl call (serverless)
JOB=$(curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "authorization: $RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{"input": {"prompt": "Hello world"}}')

echo $JOB | jq '.output'
```

---

## 9. Templates

RunPod provides a Template Gallery with pre-built Docker images:

### Official Templates (via `runpod/containers` GitHub repo)
- **PyTorch 2.8 + CUDA 12.8**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu`
- **PyTorch 2.4 + CUDA 12.4**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Various other ML framework templates

### Template Features
- Pre-installed: PyTorch, CUDA, cuDNN, Python
- Built-in services: Jupyter Lab, SSH server
- All available on Docker Hub under `runpod/` namespace
- Source Dockerfiles: https://github.com/runpod/containers

### Custom Templates
You can create custom templates from any Docker image and share them in the Template Gallery.

