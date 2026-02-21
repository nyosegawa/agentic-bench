# GPU クラウドプロバイダ調査 (2026-02)

## 目的

モデルのベンチマーク・推論テストを自動実行するための GPU クラウドを選定する。
重視するポイント:

- 無料枠の大きさ（継続的に無料で使えるか）
- API/SDK による完全自動化のしやすさ
- カスタムスクリプトの実行可否

---

## Tier 1: 無料枠が大きい（常時無料）

### Google Colab (Free)

- **公式サイト**: https://colab.research.google.com/
- **ドキュメント**: https://research.google.com/colaboratory/faq.html
- **価格ページ**: https://colab.research.google.com/signup
- **無料枠**: T4 GPU、セッション上限 ~12 時間、週あたりの利用量に制限あり（保証なし）
- **有料**: Pro $9.99/月、Pro+ $49.99/月、Pay As You Go
- **API**: 公開 REST API なし。Notebook ベースのみ
- **自動化**: ブラウザ操作（Chrome MCP 等）が必要
- **用途**: インタラクティブな実験、プロトタイピング

### Kaggle Notebooks

- **公式サイト**: https://www.kaggle.com/
- **ドキュメント**: https://www.kaggle.com/docs/notebooks
- **API ドキュメント**: https://www.kaggle.com/docs/api
- **GitHub (API)**: https://github.com/kaggle/kaggle-api
- **GitHub (CLI)**: https://github.com/Kaggle/kaggle-cli
- **無料枠**: P100 GPU (16GB) 30 時間/週、T4 GPU (16GB) 30 時間/週、20GB ストレージ、セッション最大 9 時間
- **API**: Kaggle API / CLI はデータセット・コンペ・Kernel 管理向け。任意 GPU 計算の直接起動は不可
- **自動化**: 限定的（Notebook ベース、Kernel Push で部分的に自動化可能）
- **用途**: データサイエンスコンペ、データセットを使った実験

### Hugging Face ZeroGPU

- **公式サイト**: https://huggingface.co/
- **ドキュメント**: https://huggingface.co/docs/hub/en/spaces-zerogpu
- **価格ページ**: https://huggingface.co/pricing
- **無料枠**: H200 スライス (~70GB VRAM) を動的共有。Free ユーザーは日次クォータあり
- **有料**: PRO ($9/月) で 8 倍のクォータ（H200 最大 25 分/日）、ZeroGPU Space ホスティング可能（最大 10 個）
- **有料 Spaces (専用 GPU)**: T4 $0.40/hr、A10G $1.00/hr、A100 $4.13/hr
- **API**: Inference API あり。Spaces で Gradio/FastAPI エンドポイント公開可能
- **自動化**: 良好（API 経由で推論可能）
- **用途**: モデルデモ、Gradio アプリ、軽量推論

### SambaNova Cloud

- **公式サイト**: https://sambanova.ai/
- **クラウドポータル**: https://cloud.sambanova.ai/
- **ドキュメント**: https://docs.sambanova.ai/
- **API リファレンス**: https://docs.sambanova.ai/cloud/api-reference/overview
- **価格ページ**: https://cloud.sambanova.ai/plans/pricing
- **GitHub (Python SDK)**: https://github.com/sambanova/sambanova-python
- **GitHub (Starter Kit)**: https://github.com/sambanova/ai-starter-kit
- **無料枠**: サインアップ時 $5 のクレジット付与（30 日有効、Llama 8B で約 3,000 万トークン相当）
- **価格（$/百万トークン）**:

  | モデル | 入力 | 出力 |
  |--------|------|------|
  | Llama 3.1 8B | $0.10 | $0.20 |
  | Llama 3.3 70B | $0.60 | $1.20 |
  | Qwen3-32B | $0.40 | $0.80 |
  | Qwen3-235B | $0.40 | $0.80 |
  | DeepSeek-V3.1 | $3.00 | $4.50 |
  | DeepSeek-R1-0528 | $5.00 | $7.00 |
  | Llama-4-Maverick-17B-128E | $0.63 | $1.80 |

- **API**: REST API、OpenAI 互換
- **制約**: API のみ（任意コード実行不可）。レート制限あり
- **用途**: LLM 推論 API

---

## Tier 2: 無料クレジットあり + API 自動化が優秀

### beam.cloud

- **公式サイト**: https://www.beam.cloud/
- **ドキュメント**: https://docs.beam.cloud/
- **価格ページ**: https://www.beam.cloud/pricing
- **GitHub**: https://github.com/beam-cloud/beta9
- **無料枠**: サインアップ時 15 時間分の GPU クレジット（ワンタイム）
- **価格（GPU 単体 + CPU/RAM 含む構成例）**:

  | GPU | GPU 単価/hr | 構成例/hr (4 vCPU, 8GB RAM) |
  |-----|------------|---------------------------|
  | A10G (24GB) | $1.05 | $1.97 |
  | RTX 4090 (24GB) | $0.69 | $1.61 |
  | H100 (80GB) | $3.50 | $4.42 |

  CPU: $0.190/core、RAM: $0.020/GB（GPU とは別途加算）

- **API**: Python SDK + REST API。Webhook、オートスケール、認証組み込み
- **課金**: ミリ秒単位。コールドスタートは課金対象外
- **自動化**: Python 関数をサーバーレスエンドポイントとしてデプロイ

### Modal

- **公式サイト**: https://modal.com/
- **ドキュメント**: https://modal.com/docs/guide
- **API リファレンス**: https://modal.com/docs/reference
- **価格ページ**: https://modal.com/pricing
- **GitHub (Python SDK)**: https://github.com/modal-labs/modal-client
- **GitHub (Examples)**: https://github.com/modal-labs/modal-examples
- **無料枠**: Starter プラン $30/月の計算クレジット（毎月リセット）
- **特別プログラム**: スタートアップ向け最大 $25,000、学術向け最大 $10,000
- **価格（秒単位課金）**:

  | GPU | $/sec | 換算 $/hr |
  |-----|-------|----------|
  | T4 (16GB) | $0.000164 | ~$0.59 |
  | L4 (24GB) | $0.000222 | ~$0.80 |
  | A10 (24GB) | $0.000306 | ~$1.10 |
  | L40S (48GB) | $0.000542 | ~$1.95 |
  | A100-40GB | $0.000583 | ~$2.10 |
  | A100-80GB | $0.000694 | ~$2.50 |
  | H100 (80GB) | $0.001097 | ~$3.95 |
  | H200 (141GB) | $0.001261 | ~$4.54 |
  | B200 (192GB) | $0.001736 | ~$6.25 |

  CPU: $0.0000131/core/sec、RAM: $0.00000222/GiB/sec（GPU とは別途加算）

- **API**: Python SDK、REST API。`@modal.function` デコレータで任意の Python をデプロイ
- **課金**: 秒単位。コンテナ、ボリューム、シークレット管理あり

### Lightning AI

- **公式サイト**: https://lightning.ai/
- **ドキュメント**: https://lightning.ai/docs/overview/home
- **価格ページ**: https://lightning.ai/pricing/
- **無料枠**: 月 15 Lightning クレジット（Free プラン）、T4 GPU 約 35 時間/月相当、100GB ストレージ
- **有料**: Pro $50/月（年払い）、Teams $140/ユーザー/月（年払い）
- **価格**:

  | GPU | $/hr |
  |-----|------|
  | T4 | $0.68 |
  | L4 | $0.70 |
  | A10G | $1.80 |

- **API**: Lightning AI Studios CLI/SDK
- **自動化**: 良好。ブラウザ内 VS Code 環境
- **用途**: フル開発環境、PyTorch Lightning ワークフロー

### RunPod

- **公式サイト**: https://www.runpod.io/
- **ドキュメント**: https://docs.runpod.io/
- **価格ページ**: https://www.runpod.io/pricing
- **GPU 価格一覧**: https://www.runpod.io/gpu-pricing
- **GitHub (Python SDK)**: https://github.com/runpod/runpod-python
- **無料枠**: 常時無料枠なし。サインアップ時 $5〜$500 のランダムボーナスクレジット
- **価格（Community Cloud、秒単位課金）**:

  | GPU | On-Demand $/hr |
  |-----|---------------|
  | RTX 4090 (24GB) | $0.20 |
  | RTX 6000 Ada (48GB) | $0.40 |
  | A40 (48GB) | $0.40 |
  | L40S (48GB) | $0.40 |
  | A100 PCIe (80GB) | $0.60 |
  | A100 SXM (80GB) | $0.79 |
  | H100 PCIe (80GB) | $1.35 |
  | H100 SXM (80GB) | $1.50 |
  | H200 (141GB) | $3.59 |

  Secure Cloud は Community Cloud より 10〜30% 割高。ストレージ: ~$0.07/GB/月

- **API**: REST API、Python SDK、サーバーレスエンドポイント
- **課金**: 秒単位。Docker コンテナ、サーバーレスワーカー対応

---

## Tier 3: 有料のみ

### Vast.ai

- **公式サイト**: https://vast.ai/
- **ドキュメント**: https://docs.vast.ai/
- **価格ページ**: https://vast.ai/pricing
- **GitHub (CLI)**: https://github.com/vast-ai/vast-cli
- **GitHub (Python SDK)**: https://github.com/vast-ai/vast-sdk
- **無料枠**: なし
- **価格（マーケットプレイス、中央値・変動あり）**:

  | GPU | $/hr (中央値) |
  |-----|--------------|
  | RTX 3090 (24GB) | $0.16 |
  | RTX 4090 (24GB) | $0.33 |
  | A40 (48GB) | $0.32 |
  | L40S (48GB) | $0.47 |
  | A100 PCIe (40GB) | $0.29 |
  | A100 SXM (80GB) | $0.67 |
  | H100 (80GB) | $1.55 |
  | H200 (141GB) | $1.97 |

  Interruptible インスタンスは On-Demand より 50%+ 安価。Reserved は最大 50% 割引

- **API**: REST API、CLI、Python SDK
- **自動化**: Docker ベース、SSH アクセス

### Lambda Labs

- **公式サイト**: https://lambda.ai/
- **ドキュメント**: https://docs.lambda.ai/
- **API ドキュメント**: https://cloud.lambdalabs.com/api/v1/docs
- **価格ページ**: https://lambda.ai/pricing
- **無料枠**: なし。学術向け 50% ディスカウントあり
- **価格（1x GPU インスタンス）**:

  | GPU | $/hr |
  |-----|------|
  | A10 (24GB) | $0.86 |
  | A100 SXM (40GB) | $1.48 |
  | GH200 (96GB) | $1.99 |
  | H100 PCIe (80GB) | $2.86 |
  | H100 SXM (80GB) | $3.78 |
  | B200 SXM (180GB) | $6.08 |

  1-Click Cluster: H100 $2.76/GPU-hr (On-Demand)、$2.63/GPU-hr (Reserved 1yr)。Egress 無料

- **API**: REST API（インスタンス管理）
- **自動化**: フル VM（SSH アクセス）
