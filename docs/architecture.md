# アーキテクチャ設計 (2026-02)

## コンセプト

エージェント（Claude Code スキル）が自律的にモデルの検証を実行し、結果を HTML レポートとして GitHub に蓄積する。

## 対象

- **モデル種別**: LLM、画像生成、音声、時系列解析、ML 全般（サイズ・種類は可変）
- **検証内容**: 動作確認 → 品質チェック → 速度/コスト測定（3段階）
- **結果保存先**: GitHub リポジトリ + HTML レポート

---

## 検証フロー（3段階）

すべてのモデルに共通の検証パイプライン。同時実行も可。

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
  | 数値/時系列 | .json | Chart.js 等でグラフ描画 |

### Stage 3: 速度/コスト測定 (Performance)

- 推論速度（tokens/sec、レイテンシ p50/p99）
- 実行時間
- GPU 使用量、推定コスト
- **複数回実行して中央値を取る**

---

## プロバイダ自動ルーティング

モデルサイズと利用可能なサービスから最安のプロバイダを自動選択する。
**既に課金済みのサービスを最優先で使う。**

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
  ├─ モデルサイズ判定（VRAM 要件）
  │   │
  │   ├─ ~13B (≤16GB VRAM)
  │   │   → Colab Pro (T4/V100) ← 課金済み
  │   │     Chrome MCP でノートブック操作
  │   │
  │   ├─ 13-30B (≤40GB VRAM)
  │   │   → Colab Pro (A100) ← 課金済み
  │   │   → 取れなければ Modal ($30/月無料)
  │   │
  │   └─ 30B+ (>40GB VRAM)
  │       → Modal (A100-80GB) or beam.cloud
  │         量子化も検討
  │
  └─ 結果 → HTML レポート生成 → GitHub commit
```

### プロバイダ別の実行方式

| プロバイダ | 方式 | 課金状態 | 適用範囲 |
|-----------|------|---------|---------|
| HF Inference API | REST API | Pro 課金済 ($9/月) | HF 上のモデル |
| Colab Pro | Chrome MCP | 課金済 ($9.99/月) | ~30B、GPU が取れれば |
| Modal | Python SDK | 無料 ($30/月) | 30B+、Colab で取れない時 |
| beam.cloud | Python SDK | 既存クレジット | 30B+、専用スクリプト要 |

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
│       │   │   └── gpu_estimator.py     # VRAM 推定
│       │   └── references/              # モデル種別ごとの検証知識
│       │       ├── eval-llm.md
│       │       ├── eval-image.md
│       │       ├── eval-tts.md
│       │       └── eval-timeseries.md
│       │
│       ├── gpu-runner/                  # 実行フェーズ
│       │   ├── SKILL.md                 # GPU クラウドでの実行戦略
│       │   ├── scripts/                 # .env 読み込み等
│       │   └── references/              # プロバイダ別ガイド
│       │       ├── colab-chrome-mcp.md
│       │       ├── modal.md
│       │       └── beam-cloud.md
│       │
│       └── eval-reporter/               # レポートフェーズ
│           ├── SKILL.md                 # レポート生成の戦略
│           ├── scripts/
│           │   ├── report_generator.py  # HTML レポート生成
│           │   └── metrics_writer.py    # metrics.json 書き出し
│           ├── references/
│           │   └── report-format.md     # レポート仕様
│           └── assets/
│               └── report_template.html
│
├── docs/                                # 設計ドキュメント
│   └── architecture.md
├── research/                            # 調査メモ
├── results/                             # 実行結果（自動 commit）
│   └── YYYY-MM-DD_model/
│       ├── report.html                  # 綺麗なレポート（GitHub Pages 公開）
│       ├── metrics.json                 # 構造化データ（比較用）
│       ├── artifacts/                   # 最終出力物（画像/音声等）
│       └── workspace/                   # 再現用（スクリプトのみ commit）
│           └── run.py                   # Agent が書いた実行コード
├── .env                                 # APIキー等（gitignore）
├── .env.example                         # 必要なキーの一覧（commit する）
├── tests/                               # scripts/ のテスト
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
HF_TOKEN=             # HuggingFace Pro トークン（必須）
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
  │   ├─ HF model card を読む
  │   ├─ モデル種別を判定 → references/eval-llm.md を参照
  │   ├─ VRAM 要件を推定
  │   └─ 最安プロバイダを選択
  │
  ├─ [gpu-runner フェーズ]
  │   ├─ references/colab-chrome-mcp.md 等を参照
  │   ├─ 推論スクリプトをその場で書く
  │   ├─ GPU クラウドで実行
  │   │   ├─ 成功 → 出力を取得
  │   │   └─ 失敗 → デバッグ → 修正 → 再実行
  │   └─ 性能測定: 速度・コスト計測
  │
  └─ [eval-reporter フェーズ]
      ├─ 出力を見て評価（テキスト読む / 画像見る）
      ├─ HTML レポート生成
      └─ results/ に保存 → git commit

results/2026-02-21_gemma3-27b/
├── report.html        ← 綺麗な最終レポート（公開用）
├── metrics.json       ← 構造化データ（比較用）
├── artifacts/         ← 生成物（画像/音声等）
└── workspace/         ← Agent が書いたスクリプト（再現用、commit する）
    └── run.py            ログ等の一時ファイルは gitignore
```

---

## 結果フォーマット (metrics.json)

```json
{
  "run_id": "2026-02-21T10:30:00Z",
  "model": "gemma-3-27b",
  "model_type": "llm",
  "provider": "colab",
  "gpu": "A100-40GB",
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
  "cost_usd": 0.0,
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
- **references/**: モデル種別ごとの検証知識（eval-llm.md, eval-image.md, eval-tts.md...）
- **自由度: 高い** — 未知のモデル種別にも探索的に対応

### gpu-runner (実行フェーズ)

- **役割**: 適切な GPU クラウドでコードを実行する
- **references/**: プロバイダ別ガイド（colab-chrome-mcp.md, modal.md, beam-cloud.md）
- **scripts/**: .env 読み込み等のヘルパー
- **自由度: 中** — プロバイダの API は決まっているが、実行内容は可変

### eval-reporter (レポートフェーズ)

- **役割**: 結果から HTML レポートと metrics.json を生成する
- **scripts/**: report_generator.py, metrics_writer.py
- **assets/**: report_template.html
- **自由度: 中** — フォーマットは決まっているが、内容は Agent が書く

### スキルチェーンのフロー

```
agentic-bench (トリガー・全体制御)
  → model-researcher (調査: 何のモデル？何を検証？どこで実行？)
  → gpu-runner (実行: コード書いて GPU クラウドで動かす)
  → eval-reporter (レポート: 結果を見て評価し、レポート生成)
```

SKILL.md 内で「次は model-researcher/references/ を参照」等と指示することでチェーンを実現。
各スキルの references/ には自分のドメインの知識のみ → コンテキスト効率が良い。

---

## 決定事項

- [x] シークレット管理: `.env` + `.gitignore`、`.env.example` を commit
- [x] 中間成果物: スクリプトは workspace/ に commit、ログは gitignore
- [x] 結果の表示: HTML レポート（GitHub Pages で公開予定）
- [x] OSS 化: MVP 完成後に public 化
- [x] スキル構成: フローでチェーン（agentic-bench → model-researcher → gpu-runner → eval-reporter）
- [x] scripts/ は .env を python-dotenv で読む

## 未決定事項

- [ ] SKILL.md の実装（agentic-bench, model-researcher, gpu-runner, eval-reporter）
- [ ] HTML レポートのテンプレート設計
- [ ] GitHub Pages での結果一覧ダッシュボード
- [ ] .gitignore のルール策定
