# 類似プロジェクト調査 (2026-02)

## 調査の目的

LLM ベンチマーク・評価・GPU クラウドオーケストレーション領域の既存プロジェクトを調査し、各プロジェクトの機能・特徴・制約・最新状況を記録する。

---

## カテゴリ 1: モデルベンチマークフレームワーク

### EleutherAI/lm-evaluation-harness
- **GitHub**: https://github.com/EleutherAI/lm-evaluation-harness (~11,500 stars)
- **公式ドキュメント**: https://www.eleuther.ai/projects/large-language-model-evaluation
- **最新リリース**: v0.4.11 (2026-02-13)
- **概要**: LLM の few-shot 評価の統一フレームワーク。60+ ベンチマーク。HF Open LLM Leaderboard のバックエンドとして使用されていた
- **特徴**: YAML ベースのタスク設定、再現可能な評価、NVIDIA/Cohere/BigScience 等が内部利用
- **制約**: GPU クラウド連携なし、スケジューリングなし、HTML レポートなし
- **最終更新**: 2026-02-15（活発に開発中）

### Stanford CRFM HELM
- **GitHub**: https://github.com/stanford-crfm/helm (~2,700 stars)
- **公式サイト**: https://crfm.stanford.edu/helm/
- **最新リリース**: v0.5.12 (2026-01-28)
- **概要**: 7 指標（精度、公平性、毒性等）× 多数シナリオの包括的評価フレームワーク。マルチモーダル対応
- **特徴**: 多次元評価、公式 Web リーダーボード、再現可能性重視
- **制約**: 重量級で複雑、GPU クラウド連携なし、コスト比較機能なし
- **最終更新**: 2026-02-19（活発に開発中）

### SemiAnalysis InferenceX
- **GitHub**: https://github.com/SemiAnalysisAI/InferenceX (~520 stars)
- **ライブダッシュボード**: https://inferencex.semianalysis.com/
- **旧名**: InferenceMAX (https://github.com/InferenceMAX/InferenceMAX からリネーム)
- **概要**: 毎晩自動で推論ベンチマークを実行。GB200 NVL72 / MI355X / B200 / GB300 NVL72 / H100 等のハードウェア比較。ライブダッシュボード
- **特徴**: 夜間自動ベンチマーク、パレートフロンティア手法、vLLM/TensorRT-LLM/SGLang 統合、ライブダッシュボード
- **制約**: 物理ハードウェア前提（クラウド抽象化なし）、モデル品質評価なし
- **最終更新**: 2026-02-21（活発に開発中）

### HuggingFace Optimum-Benchmark
- **GitHub**: https://github.com/huggingface/optimum-benchmark (~330 stars)
- **関連リーダーボード**: HF LLM-Perf Leaderboard
- **概要**: LLM のレイテンシ・スループット・メモリ・消費電力を測定。マルチバックエンド対応
- **特徴**: マルチバックエンド対応、エネルギー測定、Docker サポート
- **制約**: HF エコシステム限定、クラウド連携なし
- **最終更新**: 2025-09-25（開発停滞気味）

### HuggingFace LightEval
- **GitHub**: https://github.com/huggingface/lighteval (~2,300 stars)
- **公式ドキュメント**: https://github.com/huggingface/lighteval/blob/main/README.md
- **最新リリース**: v0.13.0 (2025-11-24)
- **概要**: HuggingFace が開発するオールインワン LLM 評価ツールキット。複数バックエンド対応
- **特徴**: 軽量設計、vLLM/TGI/Nanotron 等複数バックエンド対応、Open LLM Leaderboard の後継評価基盤
- **制約**: HF エコシステム中心、GPU プロビジョニングなし
- **最終更新**: 2026-02-20（活発に開発中）
- **備考**: Optimum-Benchmark の後継的位置づけ。2025 年以降 HF の主力評価ツール

### OpenAI Evals
- **GitHub**: https://github.com/openai/evals (~17,700 stars)
- **Web プラットフォーム**: https://evals.openai.com/
- **API ドキュメント**: https://platform.openai.com/docs/guides/evals
- **概要**: LLM 評価フレームワーク。YAML/JSON でローコード定義。LLM-as-judge 対応
- **特徴**: ローコード eval 作成、モデルによる自動採点、Web ダッシュボード (evals.openai.com)、ツール使用評価対応
- **制約**: OpenAI API ベース（GPU 推論なし）、スケジューリングなし
- **最終更新**: 2025-11-03（OSS リポジトリは更新頻度低下、Web プラットフォームに軸足移行）

### OpenAI simple-evals
- **GitHub**: https://github.com/openai/simple-evals (~4,400 stars)
- **概要**: MMLU、MATH、GPQA、DROP、HumanEval、SimpleQA、BrowseComp、HealthBench 等のリファレンス実装
- **特徴**: シンプルな実装、新モデルのベンチマーク結果を含む
- **制約**: 2025-07 以降新モデル・ベンチマーク結果の更新停止。リファレンス実装のみ維持
- **最終更新**: 2025-07-31

### MLCommons MLPerf Inference
- **GitHub**: https://github.com/mlcommons/inference (~1,500 stars)
- **公式サイト**: https://mlcommons.org/working-groups/benchmarks/inference/
- **最新リリース**: v5.1.1 (2025-10-28)
- **概要**: 業界標準の推論ベンチマーク。NVIDIA/AMD 等が正式提出
- **特徴**: ベンダー中立、多様なモデルカバレッジ、業界認知度が高い
- **制約**: 重量級プロセス、リアルタイムダッシュボードなし、四半期ごとの発表サイクル
- **最終更新**: 2026-02-20（活発に開発中）

---

## カテゴリ 2: LLM リーダーボード

### HuggingFace Open LLM Leaderboard
- **URL**: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **概要**: かつて最も参照されたオープン LLM リーダーボード。lm-evaluation-harness がバックエンド
- **特徴**: コミュニティ駆動、標準化されたスコア、13,000+ モデルを評価
- **現状**: 2025 年に正式に引退（retired）。モデル能力の進化により既存ベンチマークが陳腐化したため。後継は lighteval ベースのコミュニティリーダーボードに分散
- **制約**: コスト/性能分析なし、ハードウェア性能指標なし

### LM Arena (旧 LMSYS Chatbot Arena)
- **GitHub**: https://github.com/lm-sys/FastChat (~39,400 stars)
- **Web プラットフォーム**: https://lmarena.ai/
- **概要**: ユーザーが 2 つの匿名 LLM を比較投票。Elo ベースランキング。400+ モデル、累計 350 万票超
- **特徴**: 人間の好みベース評価、業界標準の LLM ランキング
- **運営変遷**: 2025 年に LMArena として独立法人化。2025-05 に $100M シード（評価額 $600M）、2026-01 に $150M Series A（評価額 $1.7B）を調達
- **制約**: 手動（投票ベース）、自動化不可、FastChat リポジトリ自体は 2025-06 以降更新停滞
- **最終更新 (FastChat)**: 2025-06-02

### DeepEval
- **GitHub**: https://github.com/confident-ai/deepeval (~13,700 stars)
- **公式サイト**: https://www.confident-ai.com/
- **公式ドキュメント**: https://deepeval.com/docs/getting-started
- **最新リリース**: v3.8.5 (2025-12-01)
- **概要**: Pytest ライクな LLM 評価フレームワーク。50+ メトリクス。CI/CD 統合
- **特徴**: Pytest スタイルの API、回帰テスト、RAG 評価、月間 300 万ダウンロード
- **制約**: アプリケーション評価向け（RAG、チャットボット）、GPU 性能計測なし
- **最終更新**: 2026-02-19（活発に開発中）

---

## カテゴリ 3: GPU クラウドオーケストレーション

### SkyPilot
- **GitHub**: https://github.com/skypilot-org/skypilot (~9,500 stars)
- **公式ドキュメント**: https://skypilot.readthedocs.io/
- **最新リリース**: v0.11.1 (2025-12-15)
- **概要**: UC Berkeley 発。Kubernetes + 20+ クラウドを抽象化し、自動コスト最適化で ML ワークロードを実行
- **特徴**: マルチクラウド GPU オーケストレーション、YAML ジョブ定義、スポットインスタンス管理、Kubernetes 統合
- **制約**: ベンチマーク機能なし、レポート生成なし
- **最終更新**: 2026-02-21（活発に開発中）

### LiteLLM
- **GitHub**: https://github.com/BerriAI/litellm (~36,500 stars)
- **公式ドキュメント**: https://docs.litellm.ai/
- **最新リリース**: v1.81.13 (2026-02-21、頻繁にリリース)
- **概要**: 100+ LLM プロバイダの OpenAI 互換統一 API。Python SDK + Proxy Server (AI Gateway)
- **特徴**: 統一 API、コスト追跡、ロードバランシング、ガードレール、ロギング
- **制約**: API レベルの抽象化のみ（GPU プロビジョニングなし）、ベンチマークなし
- **最終更新**: 2026-02-21（非常に活発に開発中）

---

## カテゴリ 4: 推論エンジン（ベンチマーク機能内蔵）

### vLLM
- **GitHub**: https://github.com/vllm-project/vllm (~70,800 stars)
- **公式ドキュメント**: https://docs.vllm.ai/
- **最新リリース**: v0.15.1 (2026-02-04)
- **概要**: 高スループット・メモリ効率の LLM 推論/サービングエンジン。ベンチマーク CLI 内蔵
- **特徴**: `benchmark_serving.py` / `benchmark_throughput.py`、GPU モニタリング、PagedAttention
- **制約**: 単一エンジン、クラウド抽象化なし、ダッシュボードなし
- **最終更新**: 2026-02-21（非常に活発に開発中）
- **備考**: 2025 年に PyTorch エコシステムに正式参加

### SGLang
- **GitHub**: https://github.com/sgl-project/sglang (~23,600 stars)
- **公式ドキュメント**: https://sgl-project.github.io/
- **最新リリース**: v0.5.8 (2026-01-23)
- **概要**: LMSYS 発の高性能 LLM/マルチモーダルサービングフレームワーク。ベンチマークユーティリティ内蔵
- **特徴**: RadixAttention、高速スケジューリング、TPU 対応 (SGLang-Jax)、Diffusion モデル対応
- **制約**: サービングエンジンであり、評価フレームワークではない
- **最終更新**: 2026-02-21（非常に活発に開発中）
- **備考**: 2025-03 に PyTorch エコシステムに統合。a16z オープンソース AI グラント受領。本番環境で 40 万+ GPU 上で稼働

### GuideLLM (vLLM Project)
- **GitHub**: https://github.com/vllm-project/guidellm (~860 stars)
- **最新リリース**: v0.5.3 (2026-01-23)
- **概要**: LLM デプロイメントの評価・最適化ツール。TTFT、ITL、スループット等を測定
- **特徴**: マルチモーダルベンチマーク対応（ビジョン・オーディオ）、HTML レポート出力、HuggingFace データセット統合
- **制約**: 単一エンドポイントの評価のみ、クラウドオーケストレーションなし
- **最終更新**: 2026-02-21（活発に開発中）
- **備考**: 2025 年にフルリファクタリング実施。vLLM プロジェクト公式のベンチマークツール

### NVIDIA AIPerf (旧 GenAI-Perf)
- **GitHub**: https://github.com/ai-dynamo/aiperf (~140 stars)
- **公式ドキュメント**: https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html
- **最新リリース**: v0.5.0 (2026-02-12)
- **概要**: NVIDIA 公式の生成 AI ベンチマークツール。GenAI-Perf の後継
- **特徴**: TTFT、ITL、TPS、RPS 等の主要指標測定。NIM/Triton/TensorRT-LLM 対応
- **制約**: NVIDIA エコシステム中心
- **最終更新**: 2026-02-21（活発に開発中）

---

## カテゴリ 5: エージェント駆動の ML テスト

### AgentBench (THUDM)
- **GitHub**: https://github.com/THUDM/AgentBench (~3,200 stars)
- **論文**: ICLR 2024
- **概要**: LLM-as-agent を 8 環境（OS、DB、Web 等）で評価。Docker コンテナ化
- **特徴**: エージェントの能力を多環境で評価、再現可能
- **制約**: エージェントの「能力」を評価するもの。モデルの「推論性能」を計測するものではない
- **最終更新**: 2026-02-08

### lmms-eval
- **GitHub**: https://github.com/EvolvingLMMs-Lab/lmms-eval (~3,700 stars)
- **最新リリース**: v0.6.1 (2026-02-19)
- **概要**: マルチモーダル（テキスト/画像/動画/音声）評価ツールキット
- **特徴**: lm-evaluation-harness のマルチモーダル拡張、One-for-All 設計
- **制約**: 評価のみ、推論性能計測なし
- **最終更新**: 2026-02-21（活発に開発中）
- **備考**: v0.6 で大規模リファクタリング実施（アーキテクチャ刷新、統計分析強化）

---

## カテゴリ 6: 推論最適化ベンチマーク（2025-2026 新規）

### BentoML llm-optimizer
- **GitHub**: https://github.com/bentoml/llm-optimizer (~170 stars)
- **公式ブログ**: https://www.bentoml.com/blog/announcing-llm-optimizer
- **概要**: LLM 推論のベンチマーク・最適化ツール。SGLang/vLLM 対応
- **特徴**: 制約ベースチューニング（「TTFT < 200ms」等）、パラメータスイープ自動化、ダッシュボード可視化、性能推定機能
- **制約**: SGLang/vLLM のみ対応、クラウドオーケストレーションなし
- **最終更新**: 2025-09-12（開発停滞気味）
- **備考**: 2025-09 にリリース。LLM Performance Explorer という Web 版も提供

---

## スター数サマリ (2026-02-21 時点、GitHub API で取得)

| プロジェクト | Stars | 最新リリース | 最終 Push |
|-------------|------:|-------------|-----------|
| vLLM | 70,829 | v0.15.1 | 2026-02-21 |
| FastChat (LM Arena) | 39,411 | - | 2025-06-02 |
| LiteLLM | 36,462 | v1.81.13 | 2026-02-21 |
| SGLang | 23,618 | v0.5.8 | 2026-02-21 |
| OpenAI Evals | 17,716 | - | 2025-11-03 |
| DeepEval | 13,740 | v3.8.5 | 2026-02-19 |
| lm-evaluation-harness | 11,461 | v0.4.11 | 2026-02-15 |
| SkyPilot | 9,470 | v0.11.1 | 2026-02-21 |
| OpenAI simple-evals | 4,354 | - | 2025-07-31 |
| lmms-eval | 3,702 | v0.6.1 | 2026-02-21 |
| AgentBench | 3,166 | - | 2026-02-08 |
| HELM | 2,682 | v0.5.12 | 2026-02-19 |
| LightEval | 2,308 | v0.13.0 | 2026-02-20 |
| MLPerf Inference | 1,532 | v5.1.1 | 2026-02-20 |
| GuideLLM | 860 | v0.5.3 | 2026-02-21 |
| InferenceX | 518 | - | 2026-02-21 |
| Optimum-Benchmark | 329 | - | 2025-09-25 |
| BentoML llm-optimizer | 168 | - | 2025-09-12 |
| NVIDIA AIPerf | 140 | v0.5.0 | 2026-02-21 |
