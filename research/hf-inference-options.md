# HuggingFace 推論オプション調査 (2026-02)

## 概要

HuggingFace は4つの推論実行手段を提供する。任意モデルのデプロイ可否と料金体系が異なる。

---

## 1. Serverless Inference API (HF Inference)

HF がホストする共有推論エンドポイント。対応モデルのみ。旧称「Inference API (serverless)」。2025年7月以降、CPU 推論が中心（Embedding, Text-ranking, Text-classification, BERT/GPT-2 等の小型 LLM）。

- **任意モデル**: 対応モデルのみ（HF が選定したカタログ）
- **コスト**: 無料クレジット内で利用可能（下記 Inference Providers の課金体系に統合）
- **セットアップ**: `InferenceClient` で provider に `hf-inference` を指定
- **GPU 推論**: 外部プロバイダ経由に移行済み（後述の Inference Providers を参照）

### 対応タスク

Chat Completion (LLM/VLM), Feature Extraction, Text to Image, ASR, Fill Mask, Image Classification, Image Segmentation, Object Detection, Question Answering, Summarization, Table QA, Text Classification, Text Generation, Token Classification, Translation, Zero-shot Classification

### Python API

```python
from huggingface_hub import InferenceClient

client = InferenceClient(provider="hf-inference")
client.text_generation("Hello", model="MODEL_ID")
```

### 公式ドキュメント

- HF Inference プロバイダ: https://huggingface.co/docs/inference-providers/en/providers/hf-inference
- Serverless Inference API: https://huggingface.co/docs/api-inference/en/index
- 対応モデル一覧: https://huggingface.co/models?inference_provider=hf-inference&sort=trending

---

## 2. Inference Endpoints (専用デプロイ)

任意の HF モデルを専用 GPU/CPU インスタンスにデプロイ。最も汎用的。

- **任意モデル**: HF 上の全モデル + カスタム Docker イメージ
- **コスト**: 従量制（分単位課金）、アイドル時は $0（pause / scale-to-zero）
- **セットアップ**: `create_inference_endpoint()` で作成
- **ベンダー**: AWS, GCP, Azure
- **推論エンジン**: vLLM, TGI, カスタムコンテナ対応

### GPU 料金 (AWS)

| インスタンス | サイズ | VRAM | $/hr |
|-------------|--------|------|------|
| nvidia-t4 | x1 | 14GB | $0.50 |
| nvidia-t4 | x4 | 56GB | $3.00 |
| nvidia-l4 | x1 | 24GB | $0.80 |
| nvidia-l4 | x4 | 96GB | $3.80 |
| nvidia-a10g | x1 | 24GB | $1.00 |
| nvidia-a10g | x4 | 96GB | $5.00 |
| nvidia-l40s | x1 | 48GB | $1.80 |
| nvidia-l40s | x4 | 192GB | $8.30 |
| nvidia-l40s | x8 | 384GB | $23.50 |
| nvidia-a100 | x1 | 80GB | $2.50 |
| nvidia-a100 | x2 | 160GB | $5.00 |
| nvidia-a100 | x4 | 320GB | $10.00 |
| nvidia-a100 | x8 | 640GB | $20.00 |
| nvidia-h200 | x1 | 141GB | $5.00 |
| nvidia-h200 | x2 | 282GB | $10.00 |
| nvidia-h200 | x4 | 564GB | $20.00 |
| nvidia-h200 | x8 | 1128GB | $40.00 |

### GPU 料金 (GCP)

| インスタンス | サイズ | VRAM | $/hr |
|-------------|--------|------|------|
| nvidia-t4 | x1 | 16GB | $0.50 |
| nvidia-l4 | x1 | 24GB | $0.70 |
| nvidia-l4 | x4 | 96GB | $3.80 |
| nvidia-a100 | x1 | 80GB | $3.60 |
| nvidia-a100 | x2 | 160GB | $7.20 |
| nvidia-a100 | x4 | 320GB | $14.40 |
| nvidia-a100 | x8 | 640GB | $28.80 |
| nvidia-h100 | x1 | 80GB | $10.00 |
| nvidia-h100 | x2 | 160GB | $20.00 |
| nvidia-h100 | x4 | 320GB | $40.00 |
| nvidia-h100 | x8 | 640GB | $80.00 |

### CPU 料金 (AWS)

| インスタンス | サイズ | vCPU / RAM | $/hr |
|-------------|--------|------------|------|
| intel-spr | x1 | 1 / 2GB | $0.033 |
| intel-spr | x2 | 2 / 4GB | $0.067 |
| intel-spr | x4 | 4 / 8GB | $0.134 |
| intel-spr | x8 | 8 / 16GB | $0.268 |
| intel-spr | x16 | 16 / 32GB | $0.536 |

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

### 公式ドキュメント

- Inference Endpoints トップ: https://huggingface.co/docs/inference-endpoints/en/index
- 料金表: https://huggingface.co/docs/inference-endpoints/en/pricing
- クイックスタート: https://huggingface.co/docs/inference-endpoints/en/quick_start
- 管理画面: https://endpoints.huggingface.co/

---

## 3. ZeroGPU Spaces (動的 GPU 割当)

Gradio Space に `@spaces.GPU` デコレータで NVIDIA H200 を動的割り当て。

- **任意モデル**: PyTorch ベースなら可
- **コスト**: 無料（PRO $9/月 で高優先度 + クォータ増）
- **GPU**: H200 MIG スライス（large: 70GB / xlarge: 141GB）
- **制約**: Gradio SDK のみ、`torch.compile` 非対応（ahead-of-time compilation は torch 2.8+ で対応）

### GPU サイズ

| サイズ | 実体 | VRAM | クォータ消費 |
|--------|------|------|------------|
| large (デフォルト) | Half NVIDIA H200 | 70GB | 1x |
| xlarge | Full NVIDIA H200 | 141GB | 2x |

### 日次クォータ

| アカウント | GPU 日次クォータ | 優先度 |
|-----------|---------------|--------|
| 未認証 | 2 分 | 低 |
| 無料 | 3.5 分 | 中 |
| PRO ($9/月) | 25 分 | 最高 |
| Team | 25 分 | 最高 |
| Enterprise | 45 分 | 最高 |

### ホスティング上限

- 個人 (PRO): 最大 10 ZeroGPU Spaces
- 組織 (Team/Enterprise): 最大 50 ZeroGPU Spaces

### Python API

```python
import spaces

@spaces.GPU(duration=120)          # デフォルトは60秒
def generate(prompt):
    return pipe(prompt).images

@spaces.GPU(size="xlarge")         # フル H200 (141GB)
def generate_large(prompt):
    return pipe(prompt).images
```

### 公式ドキュメント

- ZeroGPU Spaces: https://huggingface.co/docs/hub/spaces-zerogpu
- GPU Spaces 全般: https://huggingface.co/docs/hub/spaces-gpus
- ZeroGPU ahead-of-time compilation ガイド: https://huggingface.co/blog/zerogpu-aoti
- ZeroGPU Space 一覧: https://huggingface.co/spaces/enzostvs/zero-gpu-spaces

---

## 4. Inference Providers (サードパーティ統合)

HF のプロキシ経由でサードパーティ推論プロバイダにルーティング。OpenAI 互換 API。

- **任意モデル**: プロバイダがホストするモデルのみ
- **コスト**: 従量制（プロバイダの料金をパススルー、HF のマークアップなし）
- **ルーティング**: `router.huggingface.co/v1` 経由、OpenAI SDK でも利用可

### 無料クレジット

| アカウント | 月次クレジット | 超過時の従量課金 |
|-----------|-------------|--------------|
| Free | $0.10 | 不可 |
| PRO ($9/月) | $2.00 | 可 |
| Team/Enterprise | $2.00/席 | 可 |

### 統合プロバイダとタスク対応 (2026-02)

| プロバイダ | LLM | VLM | Embedding | Text-to-Image | Text-to-Video | STT |
|-----------|:---:|:---:|:---------:|:-------------:|:-------------:|:---:|
| Cerebras | o | | | | | |
| Cohere | o | o | | | | |
| Fal AI | | | | o | o | o |
| Featherless AI | o | o | | | | |
| Fireworks AI | o | o | | | | |
| Groq | o | o | | | | |
| HF Inference | o | o | o | o | | o |
| Hyperbolic | o | o | | | | |
| Novita | o | o | | | o | |
| Nscale | o | o | | o | | |
| OVHcloud | o | o | | | | |
| Public AI | o | | | | | |
| Replicate | | | | o | o | o |
| SambaNova | o | | o | | | |
| Scaleway | o | | o | | | |
| Together AI | o | o | | o | | |
| WaveSpeed AI | | | | o | o | |
| Z.ai | o | o | | | | |

### プロバイダ選択ポリシー

モデル ID にサフィックスを付与して選択方式を指定:

- `:fastest` (デフォルト) -- 最速プロバイダ（トークン/秒ベース）
- `:cheapest` -- 最安プロバイダ（出力トークン単価ベース）
- `:preferred` -- ユーザー設定の優先順に従う
- `:provider-name` -- 特定プロバイダを直接指定（例: `:sambanova`）

### Python API

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

# 自動的に最速プロバイダを選択
client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1:fastest",
    messages=[{"role": "user", "content": "Hello"}],
)

# OpenAI SDK 互換
from openai import OpenAI
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1:cheapest",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 公式ドキュメント

- Inference Providers トップ: https://huggingface.co/docs/inference-providers/en/index
- 料金・課金: https://huggingface.co/docs/inference-providers/pricing
- API リファレンス (タスク別): https://huggingface.co/docs/inference-providers/en/tasks/index
- 対応モデル検索: https://huggingface.co/inference/models
- プロバイダ設定: https://huggingface.co/settings/inference-providers
- HuggingFace 全体の料金ページ: https://huggingface.co/pricing
