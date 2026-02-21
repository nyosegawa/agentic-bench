# 類似プロジェクト調査 (2026-02)

## 調査の目的

agentic-bench の差別化ポイントを明確にするため、既存の類似プロジェクトを調査する。

---

## カテゴリ 1: モデルベンチマークフレームワーク

### EleutherAI/lm-evaluation-harness
- **GitHub**: https://github.com/EleutherAI/lm-evaluation-harness (~11,500 stars)
- **概要**: LLM の few-shot 評価の統一フレームワーク。60+ ベンチマーク。HF Open LLM Leaderboard のバックエンド
- **参考になる点**: YAML ベースのタスク設定、再現可能な評価
- **agentic-bench との差**: GPU クラウド連携なし、エージェント自動化なし、HTML レポートなし、コスト追跡なし

### Stanford CRFM HELM
- **GitHub**: https://github.com/stanford-crfm/helm (~2,700 stars)
- **概要**: 7 指標（精度、公平性、毒性等）× 42 シナリオの包括的評価。マルチモーダル対応
- **参考になる点**: 多次元評価、公式 Web リーダーボード
- **agentic-bench との差**: GPU クラウド連携なし、重量級で複雑、コスト比較なし

### SemiAnalysis InferenceX
- **GitHub**: https://github.com/SemiAnalysisAI/InferenceX (~516 stars)
- **概要**: 毎晩自動で推論ベンチマークを実行。GB200/MI355X/B200/H100 等のハードウェア比較。ライブダッシュボード
- **参考になる点**: **夜間自動ベンチマーク**、パレートフロンティア手法、ライブダッシュボード
- **agentic-bench との差**: 物理ハードウェア前提（クラウド抽象化なし）、モデル品質評価なし

### HuggingFace Optimum-Benchmark
- **GitHub**: https://github.com/huggingface/optimum-benchmark (~329 stars)
- **概要**: LLM のレイテンシ・スループット・メモリ・消費電力を測定。LLM-Perf Leaderboard
- **参考になる点**: マルチバックエンド対応、エネルギー測定
- **agentic-bench との差**: HF エコシステム限定、クラウド連携なし

### OpenAI Evals
- **GitHub**: https://github.com/openai/evals (~17,700 stars)
- **概要**: LLM 評価フレームワーク。YAML/JSON でローコード定義。LLM-as-judge 対応
- **参考になる点**: ローコード eval 作成、モデルによる自動採点
- **agentic-bench との差**: GPU 推論なし（API ベース）、スケジューリングなし

### MLCommons MLPerf Inference
- **GitHub**: https://github.com/mlcommons/inference (~1,500 stars)
- **概要**: 業界標準の推論ベンチマーク。NVIDIA/AMD 等が正式提出
- **参考になる点**: ベンダー中立、多様なモデルカバレッジ
- **agentic-bench との差**: 重量級プロセス、リアルタイムダッシュボードなし、四半期ごとの発表

---

## カテゴリ 2: LLM リーダーボード

### HuggingFace Open LLM Leaderboard
- **URL**: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **概要**: 最も参照されるオープン LLM リーダーボード。lm-evaluation-harness がバックエンド
- **参考になる点**: コミュニティ駆動、標準化されたスコア
- **agentic-bench との差**: HF インフラ集中、コスト/性能分析なし、ハードウェア性能指標なし

### LMSYS Chatbot Arena / FastChat
- **GitHub**: https://github.com/lm-sys/FastChat (~39,400 stars)
- **概要**: ユーザーが 2 つの匿名 LLM を比較投票。Elo ベースランキング。600 万票超
- **参考になる点**: 人間の好みベース評価（自動ベンチマークと相補的）
- **agentic-bench との差**: 手動（投票ベース）、自動化不可

### DeepEval
- **GitHub**: https://github.com/confident-ai/deepeval (~13,700 stars)
- **概要**: Pytest ライクな LLM 評価フレームワーク。50+ メトリクス。CI/CD 統合
- **参考になる点**: Pytest スタイルの API、回帰テスト
- **agentic-bench との差**: アプリケーション評価向け（RAG、チャットボット）、GPU 性能計測なし

---

## カテゴリ 3: GPU クラウドオーケストレーション

### SkyPilot
- **GitHub**: https://github.com/skypilot-org/skypilot (~9,500 stars)
- **概要**: UC Berkeley 発。20+ クラウドを抽象化し、自動コスト最適化で ML ワークロードを実行
- **参考になる点**: **マルチクラウド GPU オーケストレーション**、YAML ジョブ定義、スポットインスタンス管理
- **agentic-bench との差**: ベンチマーク機能なし、エージェント自動化なし、レポート生成なし

### LiteLLM
- **GitHub**: https://github.com/BerriAI/litellm (~36,500 stars)
- **概要**: 100+ LLM プロバイダの OpenAI 互換統一 API。コスト追跡、ロードバランシング
- **参考になる点**: **統一 API**、コスト追跡、ルーティング
- **agentic-bench との差**: API レベルの抽象化のみ（GPU プロビジョニングなし）、ベンチマークなし

### vLLM
- **GitHub**: https://github.com/vllm-project/vllm (~70,800 stars)
- **概要**: LLM 推論/サービングエンジン。ベンチマーク CLI 内蔵
- **参考になる点**: `benchmark_serving.py` / `benchmark_throughput.py`、GPU モニタリング
- **agentic-bench との差**: 単一エンジン、クラウド抽象化なし、ダッシュボードなし

---

## カテゴリ 4: エージェント駆動の ML テスト

### AgentBench (THUDM)
- **GitHub**: https://github.com/THUDM/AgentBench (~3,200 stars)
- **概要**: LLM-as-agent を 8 環境（OS、DB、Web 等）で評価。Docker コンテナ化。ICLR 2024
- **注意**: エージェントの**能力**を評価するもの。モデルの**性能**を計測するものではない

### lmms-eval
- **GitHub**: https://github.com/EvolvingLMMs-Lab/lmms-eval (~3,700 stars)
- **概要**: マルチモーダル（テキスト/画像/動画/音声）評価ツールキット
- **参考になる点**: lm-evaluation-harness のマルチモーダル拡張

---

## agentic-bench が埋めるギャップ

| 機能 | 既存プロジェクト | 状況 |
|------|----------------|------|
| モデル品質評価 | lm-eval-harness, HELM, OpenAI Evals | 充実 |
| 推論性能ベンチマーク | InferenceX, vLLM, Optimum-Benchmark | あるが HW 固定 |
| Web UI / リーダーボード | Open LLM Leaderboard, Chatbot Arena | 充実 |
| マルチクラウド GPU 連携 | SkyPilot | あるがベンチマーク非対応 |
| マルチ LLM プロバイダ抽象化 | LiteLLM | API のみ |
| **エージェント自律ベンチマーク** | **なし** | **大きなギャップ** |
| **Colab/Modal/RunPod/Beam 自動起動** | **なし** | **大きなギャップ** |
| **クロスプロバイダ コスト/性能比較** | **なし (SkyPilot 部分的)** | **大きなギャップ** |
| **自動 HTML レポート公開** | **なし (InferenceX 部分的)** | **ギャップ** |

### agentic-bench の独自ポジション

**既存のどのプロジェクトも、以下を組み合わせたものは存在しない:**

1. **エージェント**がモデルとプロバイダを自律的に選択
2. **複数 GPU クラウド**を抽象化（Colab, Modal, beam.cloud, RunPod）
3. **エンドツーエンド自動化**: 計算リソース確保 → ベンチマーク実行 → 結果収集 → 公開
4. **クロスプロバイダのコスト/性能比較** + HTML レポート自動生成

最も近いのは **InferenceX**（夜間自動+ダッシュボード）だが、固定ハードウェア前提。
**SkyPilot** はマルチクラウド対応だが、ベンチマーク機能がない。

### 活用すべき既存ツール

agentic-bench はゼロから作るのではなく、以下を内部で活用できる:

- **lm-evaluation-harness**: LLM ベンチマークの実行エンジンとして
- **LiteLLM**: API モデルの統一アクセス層として
- **vLLM の benchmark スクリプト**: 推論性能測定として
