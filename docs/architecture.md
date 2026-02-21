# アーキテクチャ設計メモ (2026-02)

## コンセプト

エージェント（Claude Code スキル）が自律的に GPU クラウド上でモデルのベンチマーク・推論テストを実行し、結果を GitHub リポジトリに蓄積する。

## 対象

- **モデル種別**: LLM、画像生成、ML 全般（複数種類）
- **検証内容**: ベンチマーク（精度・速度の定量評価）、推論テスト（特定入力に対する出力検証）
- **結果保存先**: GitHub リポジトリ (`nyosegawa/agentic-bench`)

---

## プロバイダ別の実行方式

### beam.cloud（Phase 1）

- **方式**: Python SDK でサーバーレス関数をデプロイ・実行
- **フロー**:
  1. スキルがベンチマークスクリプトを生成
  2. beam.cloud SDK でデプロイ (`beam deploy`)
  3. API 経由で実行をトリガー
  4. 結果を取得して GitHub に commit
- **必要なスキル**: beam.cloud 用の Agent Skill（スクリプト生成・デプロイ・実行）
- **備考**: 専用スクリプトが必要なため、Claude Code が実装するか他エージェントが対応

### Google Colab（Phase 1）

- **方式**: Chrome MCP でブラウザ操作
- **フロー**:
  1. スキルがノートブック（.ipynb）を生成
  2. Chrome MCP で Colab を開く
  3. ノートブックをアップロード・実行
  4. 結果セルの出力を読み取り
  5. 結果を GitHub に commit
- **制約**: ブラウザが必要。完全バックグラウンド実行は困難
- **利点**: Colab Pro の GPU を活用。既に課金済み

### Modal（Phase 2）

- **方式**: Python SDK (`@modal.function` デコレータ)
- **フロー**: beam.cloud とほぼ同じ。SDK の API が異なるのみ
- **利点**: $30/月の無料クレジットが毎月リセット

---

## リポジトリ構成

```
agentic-bench/
├── research/                    # 調査・設計メモ（このファイル等）
│   ├── gpu-cloud-providers.md
│   └── architecture.md
├── configs/
│   ├── models.yaml              # 検証対象モデル定義
│   ├── benchmarks.yaml          # ベンチマーク定義
│   └── providers.yaml           # プロバイダ接続設定（シークレットは除外）
├── scripts/
│   ├── run_benchmark.py         # プロバイダ共通エントリポイント
│   ├── providers/
│   │   ├── beam_adapter.py      # beam.cloud SDK 連携
│   │   ├── modal_adapter.py     # Modal SDK 連携
│   │   └── colab_adapter.py     # Colab ノートブック生成
│   └── tasks/
│       ├── inference_test.py    # 推論テスト（GPU 上で実行される）
│       └── benchmark.py         # ベンチマーク（GPU 上で実行される）
├── results/                     # 実行結果（自動 commit）
│   ├── YYYY-MM-DD_model_provider.json
│   └── summary.md               # 自動生成の比較サマリ
├── notebooks/                   # Colab 用生成ノートブック
└── .gitignore                   # シークレット・一時ファイル除外
```

---

## Claude Code スキル構成

### Option A: 単一スキル（model-verifier）

```
skill: model-verifier
├── ベンチマーク実行（プロバイダ選択 → 実行 → 結果収集）
├── 推論テスト（モデル指定 → テストケース実行 → 結果比較）
├── 結果確認（過去の結果を集計・比較・レポート）
└── 環境セットアップ（プロバイダの初期設定）
```

### Option B: プロバイダ別スキル + オーケストレーター

```
skill: agentic-bench        # オーケストレーター
skill: beam-deploy           # beam.cloud 専用
skill: colab-runner          # Colab Chrome MCP 専用
skill: modal-deploy          # Modal 専用
```

### 推奨: Option A から始めて、必要に応じて分離

最初は単一スキルで始め、プロバイダごとのロジックが複雑化したら分離する。

---

## 設定ファイルの例

### models.yaml

```yaml
models:
  - name: gemma-3-27b
    type: llm
    source: huggingface
    repo_id: google/gemma-3-27b
    requires_gpu: true
    min_vram_gb: 24

  - name: gpt-4o
    type: llm
    source: api
    provider: openai
    model_id: gpt-4o
    requires_gpu: false

  - name: flux-dev
    type: image_gen
    source: huggingface
    repo_id: black-forest-labs/FLUX.1-dev
    requires_gpu: true
    min_vram_gb: 24
```

### benchmarks.yaml

```yaml
benchmarks:
  - name: mmlu
    type: accuracy
    dataset: cais/mmlu
    metrics: [accuracy, f1]
    applicable_to: [llm]

  - name: inference_speed
    type: performance
    num_samples: 100
    metrics: [tokens_per_second, latency_p50, latency_p99]
    applicable_to: [llm]

  - name: image_quality
    type: quality
    num_samples: 50
    metrics: [fid, clip_score]
    applicable_to: [image_gen]
```

---

## 結果フォーマット (JSON)

```json
{
  "run_id": "2026-02-21T10:30:00Z",
  "model": "gemma-3-27b",
  "provider": "beam.cloud",
  "gpu": "A100-40GB",
  "benchmark": "inference_speed",
  "metrics": {
    "tokens_per_second": 42.5,
    "latency_p50_ms": 23.4,
    "latency_p99_ms": 89.2
  },
  "cost_usd": 0.12,
  "duration_seconds": 180
}
```

---

## 未決定事項

- [ ] beam.cloud のスクリプトテンプレート設計
- [ ] Colab Chrome MCP のワークフロー詳細（セル実行の待機、エラーハンドリング）
- [ ] シークレット管理方法（API キー等）: `.env` + gitignore? 1Password CLI?
- [ ] CI/CD 連携（GitHub Actions でトリガー等）の要否
- [ ] 結果の可視化方法（Markdown テーブル? GitHub Pages? Notion?）
