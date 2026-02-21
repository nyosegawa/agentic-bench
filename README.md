# agentic-bench

Claude Code のスキルチェーンで任意の ML モデルを自律的に検証するフレームワーク。
モデル調査 → GPU 実行 → レポート生成を一気通貫で行う。

## Quick Start

### 1. セットアップ

```bash
# clone
ghq get git@github.com:nyosegawa/agentic-bench.git
cd $(ghq root)/github.com/nyosegawa/agentic-bench

# 依存インストール
pip install -e ".[dev]"

# .env 作成
cp .env.example .env
# .env を編集して API キーを設定（下記参照）
```

### 2. .env の設定

```bash
# [必須] HuggingFace — 最低限これだけあれば動く
HF_TOKEN=hf_xxxxxxxx          # https://huggingface.co/settings/tokens

# [任意] GPU プロバイダ（使うものだけ設定）
MODAL_TOKEN_ID=xxx             # Modal: modal token new で取得
MODAL_TOKEN_SECRET=xxx
BEAM_TOKEN=xxx                 # beam.cloud: beam config create で取得

# [任意] API モデル検証用（GPT-4o, Claude 等を検証する場合）
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=xxx
```

**最小構成**: `HF_TOKEN` だけで以下が利用可能:
- **HF Inference API** — 対応モデルの推論（無料）
- **HF Inference Endpoints** — 任意モデルを専用 GPU にデプロイ（従量制 $0.50–2.50/hr）

GPU 実行は Colab Pro（Chrome MCP 経由、追加トークン不要）も利用可能。
Modal/beam.cloud は各トークンの設定が必要。

### 3. 使い方

このリポジトリのディレクトリで Claude Code を起動し、自然言語で指示する:

```
claude

> Gemma 3 27B を検証して
> FLUX.1-dev をベンチマークして
> Whisper large-v3 を試して
> Qwen2.5-Coder-32B のコード生成能力を評価して
```

スキルが自動でトリガーされ、以下が実行される:

1. **Research** — モデル情報取得、VRAM 推定、コスト見積もり
2. **Cost Gate** — コスト推定を提示（$5 超なら確認を求める）
3. **Execute** — 推論コードを書いて GPU で実行
4. **Report** — HTML レポート + metrics.json を生成

結果は `results/YYYY-MM-DD_modelname/` に保存される。

## スクリプト単体利用

スキル内のスクリプトは単体でも使える:

```bash
# モデル情報の取得
python .claude/skills/model-researcher/scripts/hf_model_info.py google/gemma-3-27b-it --json

# Inference API の利用可否チェック
python .claude/skills/model-researcher/scripts/hf_inference_check.py google/gemma-3-27b-it --json

# モデル検索
python .claude/skills/model-researcher/scripts/hf_model_search.py --task llm --limit 10
python .claude/skills/model-researcher/scripts/hf_model_search.py --task vlm --inference-only

# VRAM + コスト推定
python .claude/skills/model-researcher/scripts/gpu_estimator.py --params 27B --model-type llm --json
python .claude/skills/model-researcher/scripts/gpu_estimator.py --params 7B --quant int4 --model-type llm
```

## アーキテクチャ

```
.claude/skills/
├── agentic-bench/          # オーケストレータ（全体フロー制御）
│   └── SKILL.md
├── model-researcher/       # Phase 1: 調査
│   ├── SKILL.md
│   ├── scripts/            # ヘルパースクリプト
│   │   ├── hf_model_info.py
│   │   ├── hf_inference_check.py
│   │   ├── hf_model_search.py
│   │   └── gpu_estimator.py
│   └── references/         # 評価ガイド（12 種別対応）
│       ├── eval-llm.md, eval-vlm.md, eval-code-gen.md
│       ├── eval-embedding.md, eval-image.md, eval-tts.md
│       ├── eval-stt.md, eval-audio.md, eval-video-gen.md
│       ├── eval-object-detection.md, eval-3d-gen.md
│       └── eval-timeseries.md
├── gpu-runner/             # Phase 2: GPU 実行
│   ├── SKILL.md
│   └── references/
│       ├── inference-patterns.md   # 種別ごとの推論コード例
│       ├── hf-endpoints.md         # HF Inference Endpoints
│       ├── colab-chrome-mcp.md
│       ├── modal.md
│       └── beam-cloud.md
└── eval-reporter/          # Phase 3: レポート生成
    ├── SKILL.md
    ├── scripts/
    │   ├── metrics_writer.py
    │   └── report_generator.py
    ├── references/
    │   └── report-format.md
    └── assets/
        └── report_template.html
```

### 設計思想

- **Agent as ML Engineer**: 固定パイプラインではなく、エージェントがモデルカードを読み、推論コードを書き、結果を判断する
- **scripts/ はオプショナルツール**: 失敗したら Web 検索やゼロからのコード記述にフォールバック
- **references/ は経験値**: エージェントへの知見提供であり、強制ルートではない
- **Progressive Disclosure**: SKILL.md → references/ → scripts/ の順に必要な情報だけ読み込む

## 対応モデル種別

| 種別 | 例 | 自動評価 |
|------|-----|---------|
| LLM | Llama 3, Gemma 3, Qwen | tokens/sec, 出力品質 |
| VLM | InternVL, Qwen-VL | tokens/sec, ハルシネーション |
| コード生成 | DeepSeek-Coder, StarCoder2 | pass@1, FIM |
| Embedding | BGE, GTE, E5 | embeddings/sec, 検索品質 |
| 画像生成 | Stable Diffusion, FLUX | sec/image, 視覚品質 |
| TTS | Bark, F5-TTS | RTF, 音声品質 |
| STT | Whisper, Canary | WER/CER, RTF |
| 音声/音楽 | MusicGen, AudioLDM2 | RTF, 聴覚品質 |
| 動画生成 | Wan2.1, CogVideoX | sec/frame, 時間的整合性 |
| 物体検出 | YOLOv11, SAM 2, DETR | mAP, FPS |
| 3D 生成 | TripoSR, Shap-E | 生成時間, メッシュ品質 |
| 時系列 | Chronos, TimesFM | MAE/RMSE |

## GPU プロバイダ（コスト順）

プロバイダは `.env` のトークン有無を確認し、コスト最安のものを自動選択。

| プロバイダ | 必要トークン | 料金 | 用途 |
|-----------|------------|------|------|
| HF Inference API | HF_TOKEN | 無料 (HF Pro $9/月) | API 対応モデル |
| HF Inference Endpoints | HF_TOKEN | $0.50–2.50/hr | 任意 HF モデル |
| Colab Pro | なし | $9.99/月 サブスク | ~30B、Chrome MCP |
| Modal | MODAL_TOKEN_ID/SECRET | $0.59–3.95/hr ($30/月無料) | サーバーレス |
| beam.cloud | BEAM_TOKEN | $0.54–3.50/hr | 専用エンドポイント |

> **Note:** プロバイダの無料枠・クレジット残量はランタイムで確認していない。HF Pro や Colab Pro のサブスク加入、Modal の無料クレジット残量はコード上の静的な仮定であり、実際の残量を API で問い合わせるわけではない。枠切れ時はエージェントが実行失敗を検知し、次に安いプロバイダへ自動フォールバックする。Colab は API ではなく Chrome MCP（ブラウザ自動操作）で実行するため、Chrome MCP 拡張が必要。

## 開発

```bash
# テスト
pytest

# リント
ruff check .
ruff format --check .
```
