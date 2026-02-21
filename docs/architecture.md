# アーキテクチャ設計 (2026-02)

## コンセプト

エージェント（Claude Code スキル）が自律的にモデルの検証を実行し、結果を HTML レポートとして GitHub に蓄積する。

## 対象

- **モデル種別**: LLM、VLM、コード生成、Embedding、画像生成、TTS、STT、音声/音楽、動画生成、物体検出、3D 生成、時系列（12 種別）
- **検証内容**: 動作確認 → 品質チェック → 速度/コスト測定（3 段階）
- **結果保存先**: GitHub リポジトリ + HTML レポート

---

## 検証フロー（3 段階）

すべてのモデルに共通の検証パイプライン。

### Stage 1: 動作確認 (Smoke Test)

- モデルがロードできるか
- 基本的な入力に対して出力が返るか
- エラーなく推論が完了するか
- **判定**: Pass / Fail

### Stage 2: 品質チェック (Quality)

- 実際の出力を人間が目で確認できる形で保存
- モダリティ別:
  | 出力タイプ | 保存形式 | HTML レポートでの表示 |
  |-----------|---------|-------------------|
  | テキスト | .txt / inline | そのまま表示 |
  | 画像 | .png / .jpg | `<img>` 埋め込み |
  | 音声 | .wav / .mp3 | `<audio>` プレイヤー |
  | 動画 | .mp4 | `<video>` プレイヤー |
  | 3D | .glb / .obj | viewer 埋め込み |
  | 数値/時系列 | .json | Chart.js 等でグラフ描画 |

### Stage 3: 速度/コスト測定 (Performance)

- 推論速度（tokens/sec、RTF、sec/image 等、種別依存）
- 実行時間
- GPU 使用量、推定コスト
- **複数回実行して中央値を取る**

---

## プロバイダ自動ルーティング

モデルサイズと利用可能なトークンからコスト最安のプロバイダを自動選択する。
**`.env` のトークン有無を確認し、利用可能なプロバイダの中で最安のものを選ぶ。**

```
モデル指定
  │
  ├─ API モデル（GPT-4o, Claude, Gemini 等）
  │   → 直接 API コール。GPU 不要。最安。
  │
  ├─ HF Inference API で利用可能？
  │   → YES → HF Inference API（HF Pro 枠で無料）
  │            HTTP リクエストだけ。最もシンプル。
  │
  ├─ gpu_estimator.py でコスト順推薦
  │   │ (VRAM 要件 + 各プロバイダの時間単価でソート)
  │   │
  │   ├─ HF Inference Endpoints ($0.50–2.50/hr)
  │   │   → HF_TOKEN のみで任意モデルをデプロイ可能
  │   │
  │   ├─ Colab Pro ($9.99/月サブスク)
  │   │   → Chrome MCP でノートブック操作
  │   │
  │   ├─ Modal ($0.59–3.95/hr)
  │   │   → MODAL_TOKEN_ID/SECRET 必要、$30/月無料枠
  │   │
  │   └─ beam.cloud ($0.54–3.50/hr)
  │       → BEAM_TOKEN 必要、既存クレジット
  │
  └─ 結果 → HTML レポート生成 → GitHub commit
```

### コスト見積もり (Cost Gate)

gpu_estimator.py が GPU 時間単価 × 推定実行時間でコストを見積もる。
Phase 2 (GPU 実行) の前に見積もりを提示し、$5 超の場合はユーザーに確認を求める。

### プロバイダ選択の制限事項

**無料枠・クレジット残量のランタイム確認は行わない。** `prepaid: True`（サブスク/無料クレジットあり）フラグは gpu_estimator.py 内の静的な定義であり、実際の残量を API で問い合わせているわけではない。

| プロバイダ | prepaid | 残量確認 | 備考 |
|-----------|---------|---------|------|
| HF Inference API | True | ❌ | HF Pro 加入を仮定。サブスク状態は未チェック |
| Colab Pro | True | ❌ | Colab Pro 加入を仮定。残コンピュートユニットの確認 API が存在しない |
| Modal | True | ❌ | $30/月の無料クレジットを仮定。CLI/API での確認は技術的に可能だが未実装 |
| HF Endpoints | False | — | 従量課金のみ |
| beam.cloud | False | — | 従量課金のみ |

**フォールバック方式で対処:** 枠切れやGPU不足で実行に失敗した場合、エージェントは同じプロバイダで2回リトライした後、次に安いプロバイダへ自動的に切り替える。実行失敗のコストは時間のみ（課金は成功した実行時間分のみ）であるため、事前の残量チェックより試行→フォールバックの方が実用的。

### プロバイダ別の実行方式

| プロバイダ | 方式 | 必要トークン | 料金 |
|-----------|------|------------|------|
| HF Inference API | REST API | HF_TOKEN | 無料 (Pro $9/月) |
| HF Inference Endpoints | Python SDK (`huggingface_hub`) | HF_TOKEN | 従量制 ($0.50–2.50/hr) |
| Colab Pro | Chrome MCP | なし | $9.99/月 サブスク |
| Modal | Python SDK | MODAL_TOKEN_ID/SECRET | 従量制 ($0.59–3.95/hr) |
| beam.cloud | Python SDK | BEAM_TOKEN | 従量制 ($0.54–3.50/hr) |

**Colab の実行方式について:** Colab は API ではなく Chrome MCP（ブラウザ自動操作）で実行する。エージェントが Chrome を操作して colab.research.google.com にアクセスし、ノートブックの作成・GPU ランタイム選択・セル実行・出力読み取りを行う。そのためトークンは不要だが、Chrome MCP 拡張がインストールされたブラウザが必要。

---

## リポジトリ構成

```
agentic-bench/
├── AGENTS.md / CLAUDE.md
├── .claude/
│   └── skills/
│       ├── agentic-bench/               # オーケストレーター
│       │   └── SKILL.md                 # 全体フロー: 調査→実行→レポート
│       │
│       ├── model-researcher/            # 調査フェーズ
│       │   ├── SKILL.md                 # モデル調査・要件整理の戦略
│       │   ├── scripts/
│       │   │   ├── hf_model_info.py     # HF model card 情報取得
│       │   │   ├── hf_inference_check.py # Inference API 利用可否チェック
│       │   │   ├── hf_model_search.py   # モデル検索
│       │   │   └── gpu_estimator.py     # VRAM 推定 + コスト見積もり
│       │   └── references/              # モデル種別ごとの検証知識 (12 種)
│       │       ├── eval-llm.md
│       │       ├── eval-vlm.md
│       │       ├── eval-code-gen.md
│       │       ├── eval-embedding.md
│       │       ├── eval-image.md
│       │       ├── eval-tts.md
│       │       ├── eval-stt.md
│       │       ├── eval-audio.md
│       │       ├── eval-video-gen.md
│       │       ├── eval-object-detection.md
│       │       ├── eval-3d-gen.md
│       │       └── eval-timeseries.md
│       │
│       ├── gpu-runner/                  # 実行フェーズ
│       │   ├── SKILL.md                 # GPU クラウドでの実行戦略
│       │   └── references/              # プロバイダ別ガイド
│       │       ├── hf-endpoints.md      # HF Inference Endpoints
│       │       ├── colab-chrome-mcp.md  # Colab + Chrome MCP
│       │       ├── modal.md             # Modal Python SDK
│       │       ├── beam-cloud.md        # beam.cloud SDK
│       │       └── inference-patterns.md # 種別ごとの推論コード例 (12 種)
│       │
│       └── eval-reporter/               # レポートフェーズ
│           ├── SKILL.md                 # レポート生成の戦略
│           ├── scripts/
│           │   └── metrics_writer.py    # metrics.json 書き出し
│           └── references/
│               └── report-format.md     # metrics.json スキーマ + デザインガイド
│
├── docs/                                # 設計ドキュメント
│   └── architecture.md
├── research/                            # 調査データ（純粋な調査情報のみ）
├── results/                             # 実行結果（自動 commit）
│   └── YYYY-MM-DD_model/
│       ├── report.html                  # 綺麗なレポート
│       ├── metrics.json                 # 構造化データ（比較用）
│       ├── artifacts/                   # 最終出力物（画像/音声等）
│       └── workspace/                   # 再現用（スクリプトのみ commit）
│           └── run.py                   # Agent が書いた実行コード
├── tests/                               # scripts/ のテスト
├── .env                                 # APIキー等（gitignore）
├── .env.example                         # 必要なキーの一覧（commit する）
└── .gitignore
```

**設計思想**:
- フローでスキルを分離: 調査 (model-researcher) → 実行 (gpu-runner) → レポート (eval-reporter)
- 各スキルが自分のドメインの references/ だけ持つ → コンテキスト効率が良い
- agentic-bench がオーケストレーターとして全体を制御
- モデル種別が増えたら model-researcher/references/ に足す
- プロバイダが増えたら gpu-runner/references/ に足す

---

## 環境変数 (.env)

```bash
# .env.example（リポジトリにcommit、値は空）
HF_TOKEN=             # HuggingFace トークン（必須: Inference API + Endpoints）
BEAM_TOKEN=           # beam.cloud API トークン（beam.cloud 使用時）
MODAL_TOKEN_ID=       # Modal トークン ID（Modal 使用時）
MODAL_TOKEN_SECRET=   # Modal トークンシークレット（Modal 使用時）
OPENAI_API_KEY=       # OpenAI API キー（API モデル検証時）
ANTHROPIC_API_KEY=    # Anthropic API キー（API モデル検証時）
GOOGLE_API_KEY=       # Google AI API キー（API モデル検証時）
```

scripts/ 内のスクリプトは `python-dotenv` でリポジトリ root の `.env` を読む:
```python
from dotenv import load_dotenv
load_dotenv()  # agentic-bench/.env
```

---

## ワークフロー

```
agentic-bench/ で Claude セッション開始
.env にAPIキー等が読み込まれた状態

ユーザー: "Gemma 3 27B を検証して"
  │
  ▼
[agentic-bench スキル発動] ← オーケストレーター
  │
  ├─ [model-researcher フェーズ]
  │   ├─ hf_model_info.py で HF model card を読む
  │   ├─ モデル種別を判定 → references/eval-*.md を参照
  │   ├─ gpu_estimator.py で VRAM 要件推定 + コスト見積もり
  │   ├─ hf_inference_check.py で Inference API 利用可否チェック
  │   └─ コスト最安のプロバイダを選択
  │
  ├─ [Cost Gate]
  │   └─ コスト推定を提示 → $5 超なら確認を求める
  │
  ├─ [gpu-runner フェーズ]
  │   ├─ references/inference-patterns.md + プロバイダ参照
  │   ├─ 推論スクリプトをその場で書く
  │   ├─ GPU クラウドで実行
  │   │   ├─ 成功 → 出力を取得
  │   │   └─ 失敗 → デバッグ → 修正 → 再実行 or 別プロバイダ
  │   └─ 性能測定: 速度・コスト計測
  │
  └─ [eval-reporter フェーズ]
      ├─ 出力を見て評価（テキスト読む / 画像見る）
      ├─ metrics.json 生成
      ├─ HTML レポート生成
      └─ results/ に保存 → git commit

results/2026-02-21_gemma3-27b/
├── report.html        ← 綺麗な最終レポート
├── metrics.json       ← 構造化データ（比較用）
├── artifacts/         ← 生成物（画像/音声等）
└── workspace/         ← Agent が書いたスクリプト（再現用）
    └── run.py
```

---

## 結果フォーマット (metrics.json)

```json
{
  "run_id": "2026-02-21T10:30:00Z",
  "model": "gemma-3-27b",
  "model_type": "llm",
  "provider": "hf_endpoints",
  "gpu": "A100-80GB",
  "stages": {
    "smoke": {
      "status": "pass",
      "load_time_seconds": 12.3
    },
    "quality": {
      "outputs": ["artifacts/output_001.txt"],
      "notes": null
    },
    "performance": {
      "tokens_per_second": 42.5,
      "latency_p50_ms": 23.4,
      "latency_p99_ms": 89.2,
      "num_runs": 5
    }
  },
  "cost_usd": 0.63,
  "duration_seconds": 180
}
```

---

## Claude Code スキル (.claude/skills/)

フロー × ドメイン知識でスキルを分離。各スキルが軽量で、自分の領域だけ知っている。

### agentic-bench (オーケストレーター)

- **トリガー**: "〇〇を検証して", "モデルをベンチマーク", "新しいモデルを試して"
- **動作**: 全体フローを制御。model-researcher → gpu-runner → eval-reporter の順に進める
- **自由度: 高い** — フロー自体を状況に応じて変える判断もする

### model-researcher (調査フェーズ)

- **役割**: モデルを調べて、何をどう検証すべきか整理する
- **scripts/**: hf_model_info.py, hf_inference_check.py, hf_model_search.py, gpu_estimator.py
- **references/**: モデル種別ごとの検証知識（12 種類の eval-*.md）
- **自由度: 高い** — 未知のモデル種別にも探索的に対応

### gpu-runner (実行フェーズ)

- **役割**: 適切な GPU クラウドでコードを実行する
- **references/**: プロバイダ別ガイド（hf-endpoints.md, colab-chrome-mcp.md, modal.md, beam-cloud.md, inference-patterns.md）
- **自由度: 中** — プロバイダの API は決まっているが、実行内容は可変

### eval-reporter (レポートフェーズ)

- **役割**: 結果から HTML レポートと metrics.json を生成する
- **scripts/**: metrics_writer.py
- **references/**: report-format.md（metrics.json スキーマ + デザインガイドライン）
- **自由度: 高** — HTML はエージェントが直接生成。テンプレートなし。デザインガイドラインのみ提供

### スキルチェーンのフロー

```
agentic-bench (トリガー・全体制御)
  → model-researcher (調査: 何のモデル？何を検証？どこで実行？)
  → gpu-runner (実行: コード書いて GPU クラウドで動かす)
  → eval-reporter (レポート: 結果を見て評価し、レポート生成)
```

SKILL.md 内で「次は gpu-runner/references/ を参照」等と指示することでチェーンを実現。
各スキルの references/ には自分のドメインの知識のみ → コンテキスト効率が良い。

---

## 決定事項

- [x] シークレット管理: `.env` + `.gitignore`、`.env.example` を commit
- [x] 中間成果物: スクリプトは workspace/ に commit、ログは gitignore
- [x] 結果の表示: HTML レポート（GitHub Pages で公開予定）
- [x] スキル構成: フローでチェーン（agentic-bench → model-researcher → gpu-runner → eval-reporter）
- [x] scripts/ は .env を python-dotenv で読む
- [x] SKILL.md 実装 — 全 4 スキル完了
- [x] 12 モデル種別対応 — 分類、評価ガイド、推論パターン
- [x] 5 プロバイダ対応 — HF Inference API, HF Endpoints, Colab, Modal, beam.cloud
- [x] コスト見積もり — gpu_estimator.py + Cost Gate
- [x] .gitignore ルール策定
