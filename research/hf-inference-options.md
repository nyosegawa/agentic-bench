# HuggingFace 推論オプション調査 (2026-02)

## 概要

HuggingFace は4つの推論実行手段を提供する。任意モデルのデプロイ可否と料金体系が異なる。

---

## 1. Inference API (サーバーレス)

HF がホストする共有推論エンドポイント。対応モデルのみ。

- **任意モデル**: ❌ 対応モデルのみ（HF が選定したカタログ）
- **コスト**: 無料 (Pro $9/月で20倍のクレジット)
- **セットアップ**: `InferenceClient` 1行
- **適用**: LLM, 画像生成, VLM, Embedding は対応多い。TTS, 動画, 検出はほぼ非対応

### 種別ごとの対応状況 (2026-02 調査)

| 種別 | 対応モデル数 | 主要モデル |
|------|------------|-----------|
| LLM | 多い | Qwen2.5, Llama 3.1, GPT-OSS |
| 画像生成 | 多い | SDXL, FLUX.1, HunyuanImage |
| VLM | 多い | Qwen-VL, DeepSeek-OCR |
| Embedding | 多い | BGE, E5, Qwen-Embedding |
| STT | 少ない (2件) | Whisper large-v3, turbo のみ |
| TTS | ほぼゼロ (1件) | Kokoro-82M のみ |
| 動画生成 | ゼロ | — |
| 物体検出 | ゼロ | — |

```python
from huggingface_hub import InferenceClient
client = InferenceClient(token=os.environ["HF_TOKEN"])
client.text_generation("Hello", model="MODEL_ID")
```

---

## 2. Inference Endpoints (専用デプロイ)

任意のHFモデルを専用GPUインスタンスにデプロイ。最も汎用的。

- **任意モデル**: ✅ HF上の全モデル + カスタム Docker イメージ
- **コスト**: 従量制 (分単位課金)、アイドル時は $0 (pause / scale-to-zero)
- **セットアップ**: `create_inference_endpoint()` 1行
- **GPU 選択肢**: T4, L4, A10G, L40S, A100, H200, H100

### GPU 料金 (AWS, 2026-02)

| インスタンス | VRAM | $/hr |
|-------------|------|------|
| nvidia-t4 | 14GB | $0.50 |
| nvidia-l4 | 24GB | $0.80 |
| nvidia-a10g | 24GB | $1.00 |
| nvidia-l40s | 48GB | $1.80 |
| nvidia-a100 | 80GB | $2.50 |
| nvidia-h200 | 141GB | $5.00 |

### Python API

```python
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    "my-endpoint",
    repository="nineninesix/kani-tts-2-en",
    task="text-to-speech",
    accelerator="gpu",
    vendor="aws",
    region="us-east-1",
    instance_type="nvidia-a10g",
)
endpoint.wait()
audio = endpoint.client.text_to_speech("Hello world")

# コスト管理
endpoint.pause()          # 停止 ($0)
endpoint.scale_to_zero()  # リクエスト時に自動起動
endpoint.delete()         # 完全削除
```

---

## 3. ZeroGPU Spaces (動的GPU割当)

Gradio Space に `@spaces.GPU` デコレータでGPUを動的割り当て。

- **任意モデル**: ✅ PyTorch ベースなら可
- **コスト**: 無料 (Pro $9/月が必要、日25分のGPUクォータ)
- **GPU**: H200 の半分 (70GB) or フル (141GB)
- **制約**: Gradio SDK のみ、`torch.compile` 非対応

### 日次クォータ

| アカウント | GPU 日次クォータ | 優先度 |
|-----------|---------------|--------|
| 無料 | 3.5 分 | 低 |
| Pro ($9/月) | 25 分 | 最高 |
| Enterprise | 45 分 | 最高 |

```python
import spaces
@spaces.GPU(duration=120)
def generate(prompt):
    return pipe(prompt).images
```

---

## 4. Inference Providers (サードパーティ統合)

HF のプロキシ経由でサードパーティ推論プロバイダにルーティング。

- **任意モデル**: ❌ プロバイダがホストするモデルのみ
- **コスト**: 従量制 (プロバイダの料金をパススルー、マークアップなし)
- **無料クレジット**: Free $0.10/月、Pro $2.00/月

### 統合プロバイダ (2026-02)

Cerebras, Cohere, Fal AI, Featherless AI, Fireworks AI, Groq,
HF Inference, Hyperbolic, Novita, Nscale, OVHcloud, Public AI,
Replicate, SambaNova, Scaleway, Together AI, WaveSpeed AI, Z.ai

```python
client = InferenceClient()
# 自動的に最速プロバイダを選択
client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1:fastest",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## agentic-bench での優先順位への示唆

| 優先度 | プロバイダ | 条件 | 理由 |
|--------|-----------|------|------|
| 1 | Inference API | 対応モデルのみ | 無料、HF_TOKEN のみ |
| 2 | Inference Endpoints | 任意モデル | HF_TOKEN のみで専用GPU、Modal より安い |
| 3 | Colab Pro | ~30B | 月額課金済み、Chrome MCP |
| 4 | Modal | 30B+ | $30/月無料枠 |
| 5 | beam.cloud | 代替 | 既存クレジット |

Inference Endpoints は HF_TOKEN だけで任意モデルをデプロイできるため、
Modal/beam のトークン登録が不要。TTS 等の Inference API 非対応モデルに最適。
