# HF Inference Endpoints ガイド

HuggingFace Inference Endpoints: 任意の HF モデルを専用 GPU にデプロイ。
`HF_TOKEN` のみで利用可能。Modal/beam のトークン不要。

## GPU 料金 (AWS, 2026-02)

| インスタンス | VRAM | $/hr |
|-------------|------|------|
| nvidia-t4 | 14GB | $0.50 |
| nvidia-l4 | 24GB | $0.80 |
| nvidia-a10g | 24GB | $1.00 |
| nvidia-l40s | 48GB | $1.80 |
| nvidia-a100 | 80GB | $2.50 |

## 基本フロー

```python
import os
from huggingface_hub import create_inference_endpoint

# 1. エンドポイント作成
endpoint = create_inference_endpoint(
    "bench-run",
    repository="MODEL_ID",
    task="text-generation",  # or text-to-speech, text-to-image, etc.
    accelerator="gpu",
    vendor="aws",
    region="us-east-1",
    instance_type="nvidia-l4",  # GPU 選択
    token=os.environ["HF_TOKEN"],
)

# 2. 起動待ち
endpoint.wait()  # ~2-5 分

# 3. 推論実行
client = endpoint.client
response = client.text_generation("Hello", max_new_tokens=100)

# 4. 停止（課金停止）
endpoint.pause()   # 一時停止 ($0)
# endpoint.delete()  # 完全削除
```

## タスク別の推論

```python
# TTS
audio = endpoint.client.text_to_speech("Hello world")
with open("output.wav", "wb") as f:
    f.write(audio)

# 画像生成
image = endpoint.client.text_to_image("A cat on a cloud")
image.save("output.png")

# Embedding
vectors = endpoint.client.feature_extraction("Hello world")
```

## コスト管理

- `endpoint.pause()`: 停止、課金なし。再開は `endpoint.resume()`
- `endpoint.scale_to_zero()`: リクエスト時に自動起動（コールドスタート ~2min）
- `endpoint.delete()`: 完全削除

**ベンチマーク後は必ず `endpoint.pause()` または `endpoint.delete()` を呼ぶこと。**

## エラーハンドリング

| エラー | 対処 |
|-------|------|
| OOM | より大きい GPU (`nvidia-a100`) に変更 |
| Timeout (起動) | 大きいモデルは 5-10 分かかる。`endpoint.wait(timeout=600)` |
| Model not found | リポジトリ ID を確認。gated model は HF_TOKEN に accept が必要 |
| Task mismatch | `task` パラメータをモデルの pipeline_tag に合わせる |

## instance_type の選び方

gpu_estimator.py の推薦に従う。マッピング:

| gpu_estimator 出力 | instance_type |
|-------------------|---------------|
| T4 | nvidia-t4 |
| L4 | nvidia-l4 |
| A10G | nvidia-a10g |
| L40S | nvidia-l40s |
| A100-80GB | nvidia-a100 |
