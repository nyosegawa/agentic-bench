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
├── docs/
│   └── architecture.md          # この文書
├── research/
│   └── gpu-cloud-providers.md   # プロバイダ調査
├── configs/
│   ├── models.yaml              # 検証対象モデル定義
│   └── providers.yaml           # プロバイダ接続設定（シークレット除外）
├── src/
│   ├── router.py                # プロバイダ自動ルーティング
│   ├── stages/
│   │   ├── smoke.py             # Stage 1: 動作確認
│   │   ├── quality.py           # Stage 2: 品質チェック
│   │   └── performance.py       # Stage 3: 速度/コスト測定
│   ├── providers/
│   │   ├── hf_api.py            # HF Inference API
│   │   ├── colab.py             # Colab (ノートブック生成)
│   │   ├── modal_adapter.py     # Modal SDK
│   │   └── beam_adapter.py      # beam.cloud SDK
│   └── report/
│       └── html_generator.py    # HTML レポート生成
├── results/                     # 実行結果（自動 commit）
│   └── YYYY-MM-DD_model/
│       ├── report.html          # ブラウザで開いて確認
│       ├── metrics.json         # 構造化データ（比較用）
│       └── artifacts/           # 生成物（画像/音声等）
├── notebooks/                   # Colab 用生成ノートブック
├── tests/                       # コアロジックのテスト
└── .gitignore
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

## Claude Code スキル

単一スキル `agentic-bench` から始め、必要に応じて分離する。

```
skill: agentic-bench
├── 検証実行: "Gemma 3 27B を検証して"
│   → ルーティング → 3段階検証 → HTML レポート → commit
├── 結果確認: "最新の結果を見せて"
│   → results/ から集計・比較
└── レポート比較: "Gemma 3 と Llama 3 を比較して"
    → 複数モデルの metrics.json を横断比較
```

beam.cloud 等の専用スクリプトが必要なプロバイダは Agent Skill として分離。

---

## 未決定事項

- [ ] シークレット管理方法（API キー等）: `.env` + gitignore? 1Password CLI?
- [ ] Colab Chrome MCP のワークフロー詳細（セル実行の待機、エラーハンドリング）
- [ ] beam.cloud 用 Agent Skill の設計
- [ ] HTML レポートのテンプレート設計
- [ ] GitHub Pages での結果一覧公開の要否
