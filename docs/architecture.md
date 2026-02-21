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
│   └── skills/                          # Agent Skills（本体）
│       ├── agentic-bench/               # メインスキル: 探索的モデル検証
│       │   ├── SKILL.md                 # 指示書
│       │   └── references/
│       │       ├── providers.md         # プロバイダ別ガイド
│       │       └── report-format.md     # レポート仕様
│       ├── beam-deploy/                 # beam.cloud 専用スキル
│       │   └── SKILL.md
│       └── colab-runner/                # Colab Chrome MCP 専用スキル
│           └── SKILL.md
├── docs/
│   └── architecture.md                  # この文書
├── research/                            # 調査・設計メモ
├── src/
│   ├── utils/                           # Agent が任意で使えるヘルパー
│   │   ├── hf_model_info.py             # HF model card 情報取得
│   │   ├── gpu_estimator.py             # VRAM 推定
│   │   ├── report_generator.py          # HTML レポートテンプレート
│   │   └── metrics_writer.py            # metrics.json 書き出し
│   └── providers/                       # プロバイダ定型処理
│       ├── colab_helpers.py
│       ├── modal_helpers.py
│       └── beam_helpers.py
├── results/                             # 実行結果（自動 commit）
│   └── YYYY-MM-DD_model/
│       ├── report.html                  # ブラウザで確認
│       ├── metrics.json                 # 構造化データ
│       └── artifacts/                   # 生成物（画像/音声等）
├── tests/                               # ユーティリティのテスト
└── .gitignore
```

**設計思想**: `.claude/skills/` の SKILL.md（指示書）が本体。
`src/` は Agent が使っても使わなくても良い補助ツール。
Agent は model card を読み、コードをその場で書き、出力を見て判断する。

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

Agent Skills は「コード」ではなく「指示書」。
Agent 自身が ML エンジニアとして探索的に検証する。

### agentic-bench (メインスキル)

- **トリガー**: "〇〇を検証して", "モデルをベンチマーク", "新しいモデルを試して"
- **動作**: model card を読む → 環境選択 → コードを書いて実行 → 出力を見て評価 → レポート
- **自由度: 高い** — Agent が状況に応じて判断。固定パイプラインではない

### beam-deploy (beam.cloud 専用)

- **トリガー**: beam.cloud でのデプロイが必要な時に agentic-bench から呼ばれる
- **動作**: beam.cloud SDK でサーバーレス関数を書いてデプロイ・実行
- **自由度: 中** — SDK の使い方は決まっているが、デプロイ内容は可変

### colab-runner (Colab 専用)

- **トリガー**: Colab での実行が必要な時に agentic-bench から呼ばれる
- **動作**: Chrome MCP で Colab を操作してモデルを実行
- **自由度: 中** — ブラウザ操作の手順はある程度決まっている

### Progressive Disclosure

```
SKILL.md (常時ロード)          → ゴール・戦略・判断基準
references/ (必要時ロード)     → プロバイダ詳細・フォーマット仕様
src/utils/ (任意で実行)        → ヘルパースクリプト
```

---

## 未決定事項

- [ ] シークレット管理方法（API キー等）: `.env` + gitignore? 1Password CLI?
- [ ] Colab Chrome MCP のワークフロー詳細（セル実行の待機、エラーハンドリング）
- [ ] SKILL.md の実装（agentic-bench, beam-deploy, colab-runner）
- [ ] HTML レポートのテンプレート設計
- [ ] GitHub Pages での結果一覧ダッシュボード
