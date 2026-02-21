# 自動モデル評価ライブラリ調査 (2026-02)

## 1. LLM 評価フレームワーク

### 一覧表

| ライブラリ | GitHub URL | Stars | 最新バージョン | 主要機能 | Python API | エージェント統合適性 |
|---|---|---|---|---|---|---|
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | ~11.3k | v0.4.10 | 60+の標準ベンチマーク、HF/vLLM/API対応、チャットテンプレート、few-shot評価 | あり (`lm_eval` パッケージ) | 高: CLI/Python API両方、YAML設定でタスク定義、GPU上でバッチ実行可能 |
| HELM | https://github.com/stanford-crfm/helm | ~2.7k | 継続的リリース | ホリスティック評価(精度/効率/バイアス/毒性)、VHELM/HEIM/MedHELM拡張 | あり (`crfm-helm` パッケージ) | 中: 設定が複雑だがPython API有、適応的サンプリング対応 |
| DeepEval | https://github.com/confident-ai/deepeval | ~12.8k | v3.x系 | G-Eval、エージェントトレース評価、マルチターン自動検出、14+ メトリクス | あり (`deepeval` パッケージ) | 高: pytest統合、シンプルなAPI、エージェントトレース対応、カスタムメトリクス容易 |
| OpenAI Evals | https://github.com/openai/evals | ~17.6k | 継続的更新 | OpenAIモデル向けeval registry、グレーダー、HealthBench | あり (Python SDK) | 中: OpenAIモデル中心、カスタムeval登録可能だがエコシステムに依存 |
| Ragas | https://github.com/explodinggradients/ragas | ~12.4k | v0.4.3 | RAG評価特化(faithfulness/context precision/recall/answer relevancy) | あり (`ragas` パッケージ) | 高: リファレンスフリー評価、シンプルAPI、エージェントワークフロー対応 |
| TruLens | https://github.com/truera/trulens | ~2.8k | v2.7.0 | OpenTelemetryトレーシング、RAG三角形(groundedness/relevance/context)、LLM-as-Judge | あり (`trulens` パッケージ) | 高: OTel統合、Snowflake支援、エージェントトレース対応 |
| Promptfoo | https://github.com/promptfoo/promptfoo | ~10.5k | v0.120.x | プロンプトテスト、レッドチーミング、CI/CD統合、マルチプロバイダ比較 | 限定的 (Node.js中心、YAML設定) | 中: CI/CD向き、YAML宣言的設定、Python APIは間接的 |
| LangSmith Evaluation | https://docs.langchain.com/langsmith/evaluation | N/A (SaaS) | 継続的更新 | マルチターン評価、Insights Agent、pytest/Vitest統合、ペアワイズ比較 | あり (langsmith SDK) | 高: LangChain/LangGraphとの深い統合、エージェント評価に最適化 |
| Braintrust | https://github.com/braintrustdata/autoevals | N/A (SaaS) | 継続的更新 | AutoEvals、実験追跡、GitHub Actions統合、コスト分析 | あり (braintrust SDK) | 高: CI/CD統合、自動評価スコアリング、エージェントトレース対応 |
| OpenEvals | https://github.com/langchain-ai/openevals | 新規 | v0.0.17 | LLM-as-Judge、JSON完全一致、抽出評価、RAG評価 | あり (`openevals` パッケージ) | 高: 軽量、LangChain非依存でも使用可能、プログラマティック |
| AgentEvals | https://github.com/langchain-ai/agentevals | 新規 | 開発中 | エージェント軌跡評価、参照軌跡マッチング、非同期対応 | あり (`agentevals` パッケージ) | 高: エージェント軌跡専用、asyncio対応 |
| Inspect AI | https://github.com/UKGovernmentBEIS/inspect_ai | ~2k+ | 継続的更新 | 100+のプリビルト評価、エージェント評価、サンドボックス実行、マルチエージェント | あり (`inspect-ai` パッケージ) | 非常に高: エージェント評価専用設計、サンドボックス、UK AISI公式 |
| Unitxt | https://github.com/IBM/unitxt | ~1.5k | 継続的更新 | カスタマイズ可能な評価パイプライン、HF/lm-eval統合、LLM-as-Judge | あり (`unitxt` パッケージ) | 高: モジュラー設計、カタログベース、エンタープライズ向け |

### 各ライブラリ詳細

#### lm-evaluation-harness (EleutherAI)

- **概要**: LLMの少数ショット評価フレームワーク。HuggingFace Open LLM Leaderboardのバックエンド
- **対応ベンチマーク**: MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval 他60+
- **モデル対応**: HuggingFace transformers, vLLM, OpenAI API, GGUF, Megatron-DeepSpeed
- **Python API**: `lm_eval.simple_evaluate()` で1行実行可能
- **GPU要件**: モデルサイズに依存。vLLM使用時は高速推論可能
- **カスタム評価**: YAMLベースのタスク定義で追加容易
- **メンテナンス**: 非常にアクティブ。NVIDIA, Cohere, BigScience等が利用

#### HELM (Stanford CRFM)

- **概要**: ホリスティックな基盤モデル評価。精度以外にも効率・バイアス・毒性を測定
- **拡張版**: VHELM (VLM)、HEIM (画像生成)、MedHELM (医療)、HELM Capabilities
- **Python API**: `helm-run` コマンドおよび Python スクリプトからの呼び出し
- **GPU要件**: 評価対象モデルに依存
- **特記**: 適応的サンプリング (RAME) 対応。Python 3.10+ 必須

#### DeepEval (Confident AI)

- **概要**: pytest統合のLLM評価フレームワーク。v3.0でエージェント評価を大幅強化
- **メトリクス**: G-Eval, Faithfulness, Answer Relevancy, Contextual Precision/Recall, Hallucination, Toxicity, Bias等
- **v3.0新機能**: エージェントトレース評価 (タスク完了度を軌跡から評価)、マルチターン自動検出
- **Python API**: `deepeval.evaluate()`, pytest連携 `assert_test()`
- **GPU要件**: 評価にLLM-as-Judgeを使用するため、外部API (OpenAI等) またはローカルGPU
- **カスタム評価**: `BaseMetric` 継承で容易に追加

#### Ragas

- **概要**: RAGパイプライン特化の評価フレームワーク。リファレンスフリー評価が特徴
- **メトリクス**: Faithfulness, Context Precision, Context Recall, Answer Relevancy, Answer Correctness
- **v0.4系**: エージェントワークフロー評価にも対応
- **Python API**: `ragas.evaluate()` でDatasetから一括評価
- **GPU要件**: LLM-as-Judgeで外部APIまたはローカルLLMを使用

#### TruLens

- **概要**: OpenTelemetryベースのLLMアプリ評価・トレーシングツール
- **Snowflake支援**: Snowflake Inc.がメンテナンス
- **メトリクス**: Groundedness, Relevance, Context Relevance (RAG三角形)
- **Python API**: デコレータベースのトレーシング + フィードバック関数
- **特記**: MCP (Model Context Protocol) ベースのツールコール評価に対応

---

## 2. マルチモーダル評価

### 一覧表

| ライブラリ | GitHub URL | Stars | 最新バージョン | 対応モダリティ | Python API | エージェント統合適性 |
|---|---|---|---|---|---|---|
| lmms-eval | https://github.com/EvolvingLMMs-Lab/lmms-eval | ~3.4k | v0.6 | テキスト, 画像, 動画, 音声 | あり | 高: lm-eval-harness互換インターフェース、CLIとPython API |
| MTEB | https://github.com/embeddings-benchmark/mteb | ~3.1k | v2.0.0 | テキスト埋め込み (1000+言語) | あり (`mteb` パッケージ) | 高: HuggingFace統合、シンプルなPython API |
| bigcode-evaluation-harness | https://github.com/bigcode-project/bigcode-evaluation-harness | ~254 | 継続開発 | コード生成 (Python, Java等) | あり | 中: accelerateによるマルチGPU対応、実行ベース評価 |

### 各ライブラリ詳細

#### lmms-eval (EvolvingLMMs-Lab)

- **概要**: マルチモーダルモデル (LMM) のオールインワン評価ツールキット
- **対応モデル**: Qwen2.5-VL, Qwen3-VL, LLaVA, OpenAI API互換モデル
- **タスク**: VQA, 画像キャプション, 動画理解, 音声理解など多数
- **lm-eval-harness互換**: EleutherAIのlm-evaluation-harnessの設計を踏襲
- **GPU要件**: モデルサイズに依存。マルチGPU推論対応
- **最新機能**: ツールコール評価、MCP Server対応

#### MTEB (Massive Text Embedding Benchmark)

- **概要**: テキスト埋め込みモデルの包括的ベンチマーク。500+タスク、250+言語
- **タスク種別**: 分類, クラスタリング, 検索, リランキング, STS, 要約
- **MMTEB**: 多言語拡張版。命令追従、長文書検索、コード検索タスクを含む
- **Python API**: `mteb.run()` でモデル評価を実行
- **GPU要件**: 埋め込みモデルの推論に使用

#### bigcode-evaluation-harness

- **概要**: コード生成LLMの評価フレームワーク
- **ベンチマーク**: HumanEval, MBPP, MultiPL-E, DS-1000等
- **評価方法**: ユニットテスト実行による pass@k 測定
- **GPU要件**: accelerateによるデータ並列評価。モデルは1 GPU想定
- **メンテナンス**: BigCode Projectが管理。最終更新2025年10月

---

## 3. 画像生成モデルの自動評価

### 一覧表

| ライブラリ / メトリクス | GitHub URL | Stars | 最新バージョン | 主要メトリクス | Python API | エージェント統合適性 |
|---|---|---|---|---|---|---|
| torch-fidelity | https://github.com/toshas/torch-fidelity | ~600+ | v0.3.0 | FID, IS, KID | あり | 高: Python API完備、キャッシュ機構、学習ループ統合可能 |
| pytorch-fid | https://github.com/mseitzer/pytorch-fid | ~3.6k | v0.3.0 | FID | あり (CLI中心) | 中: CLIベース、メンテナンス非アクティブだが広く利用 |
| clean-fid | https://github.com/GaParmar/clean-fid | ~1.1k | v0.1.35 | FID, KID, CLIP-FID | あり | 高: 正確なFID計算、CLIP特徴量対応、Python API |
| ImageReward | https://github.com/THUDM/ImageReward | ~1.6k | v1.5 | ImageReward (人間嗜好スコア) | あり (`image-reward` パッケージ) | 高: pip install + 3行で推論、GPU必要だが軽量 |
| torchmetrics (image) | https://github.com/Lightning-AI/torchmetrics | ~2.2k | v1.8.2 | FID, IS, KID, LPIPS, SSIM | あり | 高: PyTorch Lightning統合、統一API |
| T2I-Eval | https://github.com/maziao/T2I-Eval | 新規 | ACL 2025 | テキスト-画像整合性 (蒸留MLLM) | あり | 中: 蒸留モデルによる高速評価、MiniCPM-V-2_6ベース |

### メトリクス解説

#### FID (Frechet Inception Distance)
- 2つの画像データセット間の分布距離を測定
- InceptionV3の中間特徴量を使用
- 低いほど良い。生成画像品質の業界標準メトリクス
- **推奨ライブラリ**: clean-fid (正確性重視) または torch-fidelity (速度重視)

#### CLIP Score
- テキストと画像の整合性を CLIP モデルで測定
- リファレンスフリー: 生成画像とプロンプトのみで計算可能
- torchmetrics の `CLIPScore` クラスで利用可能
- **GPU要件**: CLIPモデルの推論にGPU推奨

#### IS (Inception Score)
- 生成画像の品質と多様性を測定
- 高いほど良い。条件付きクラス分布の鮮明さを評価
- torch-fidelity, torchmetrics で利用可能

#### KID (Kernel Inception Distance)
- FIDの代替。少数サンプルでもバイアスが少ない
- torch-fidelity, clean-fid で利用可能

#### ImageReward
- テキスト-画像生成の人間嗜好報酬モデル
- 137kの専門家比較ペアで学習
- CLIP Score比38.6%、Aesthetic比39.6%の改善
- ReFL (Reward Feedback Learning) でモデルの直接最適化にも使用可能

---

## 4. 音声モデルの自動評価

### 一覧表

| ライブラリ / メトリクス | GitHub URL | Stars | カテゴリ | 主要メトリクス | Python API | エージェント統合適性 |
|---|---|---|---|---|---|---|
| UTMOSv2 | https://github.com/sarulab-speech/UTMOSv2 | ~100+ | MOS予測 | 自動MOS (Mean Opinion Score) | あり | 高: `model.predict()` で直接予測 |
| SpeechMOS | https://github.com/tarepan/SpeechMOS | ~200+ | MOS予測 | UTMOS22 Strong/Weak | あり (torch.hub) | 高: torch.hub.load で1行ロード |
| pesq | https://github.com/ludlows/PESQ | ~400+ | 音声品質 | PESQ (WB/NB) | あり (`pesq` パッケージ) | 高: pip install pesq、関数1つで計算 |
| pystoi | PyPI | ~200+ | 音声明瞭度 | STOI, ESTOI | あり (`pystoi` パッケージ) | 高: シンプルな関数API |
| jiwer | https://github.com/jitsi/jiwer | ~800+ | ASR精度 | WER, CER, MER, WIL | あり (`jiwer` パッケージ) | 非常に高: 軽量、GPU不要、RapidFuzz使用で高速 |
| whisper-normalizer | https://github.com/kurianbenoy/whisper_normalizer | ~200+ | テキスト正規化 | ASR出力の正規化 | あり | 高: jiwerと組み合わせてWER計算の前処理 |
| speechmetrics | https://github.com/aliutkus/speechmetrics | ~300+ | 統合ラッパー | MOSNet, PESQ, STOI, SRMR, SISDR | あり | 中: numpy互換性問題あり、メンテナンス限定的 |
| DiscreteSpeechMetrics | https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics | ~100+ | 参照あり | SpeechBERTScore, SpeechBLEU | あり | 中: 研究段階だが革新的アプローチ |
| torchmetrics (audio) | https://github.com/Lightning-AI/torchmetrics | ~2.2k | 統合 | PESQ, STOI, SI-SDR, NISQA | あり | 高: PyTorch統合、統一API |

### メトリクス解説

#### MOS予測 (UTMOS/SpeechMOS)
- **UTMOS**: SSLモデル (wav2vec2.0等) を用いた自動MOS予測
- **UTMOSv2**: VoiceMOS Challenge 2024向け改良版。16kHz入力
- **用途**: TTS (Text-to-Speech) モデルの品質評価に最適
- **GPU要件**: 推論にGPU推奨だが、CPUでも動作可能

#### PESQ (Perceptual Evaluation of Speech Quality)
- ITU-T P.862標準に基づく音声品質メトリクス
- スコア範囲: -0.5 ~ 4.5 (高いほど良い)
- **参照音声が必要** (intrusive metric)
- 16kHz (WB) または 8kHz (NB) のサンプリングレート対応

#### STOI (Short-Time Objective Intelligibility)
- 音声の明瞭度を測定
- スコア範囲: 0 ~ 1 (高いほど良い)
- **参照音声が必要**
- 音声強調や音声分離の評価に有用

#### WER (Word Error Rate) / jiwer
- ASR出力の精度測定の標準メトリクス
- WER = (S + D + I) / N (置換 + 削除 + 挿入 / 参照単語数)
- v4.0で空参照対応、RapidFuzz (C++) による高速計算
- **GPU不要**: CPU上で高速実行

#### whisper-normalizer
- ASR評価前のテキスト正規化
- 句読点、数字の読み、略語等の標準化
- 正規化なしのWERは不公平な比較になるため必須
- 英語 + 多言語 (Indic言語含む) 対応

---

## 5. LLM-as-Judge パターン

### 一覧表

| ライブラリ / 手法 | GitHub URL | Stars | タイプ | 主要特徴 | Python API | エージェント統合適性 |
|---|---|---|---|---|---|---|
| AlpacaEval | https://github.com/tatsu-lab/alpaca_eval | ~1.9k | ベンチマーク | GPT-4ベース自動評価、Chatbot Arenaと相関0.98 | あり | 高: $10以下で実行、3分以内で完了 |
| MT-Bench | https://github.com/lm-sys/FastChat (内包) | ~38k (FastChat全体) | ベンチマーク | マルチターン80問、LLM判定、人間と80%+一致 | あり (FastChat経由) | 中: FastChatの一部として提供 |
| G-Eval | https://github.com/nlpyang/geval | ~200+ | 手法 | CoTベースのLLM判定、トークン重み付きスコアリング | 限定的 (DeepEval経由推奨) | 高: DeepEvalに統合済み、5行で実装可能 |
| Prometheus | https://github.com/prometheus-eval/prometheus-eval | ~500+ | モデル | オープンソースLLM Judge、カスタムルーブリック対応 | あり (`prometheus-eval` パッケージ) | 高: 7Bモデルは16GB VRAMで動作、API不要 |
| JudgeLM | https://github.com/baaivision/JudgeLM | ~400+ | モデル | 7B/13B/33Bスケーラブル、GPT-4判定と90%+一致 | あり | 中: ICLR 2025 Spotlight、8xA100で5K件3分 |
| judges (Quotient AI) | https://github.com/quotient-ai/judges | ~300+ | ライブラリ | 研究ベースの既製LLMジャッジ、Classifier/Grader/Jury | あり (`judges` パッケージ) | 高: pip install、軽量、カスタムジャッジ追加容易 |
| OpenEvals LLM-as-Judge | https://github.com/langchain-ai/openevals | 新規 | ライブラリ | `create_llm_as_judge()`、プロンプトベース | あり | 高: LangChain非依存でも利用可能 |

### パターン詳細

#### AlpacaEval 2.0
- **手法**: 強力なLLM (GPT-4等) が対象モデルの出力をリファレンス出力と比較
- **Length-Controlled Win Rate**: 出力長バイアスを制御した改良版
- **コスト**: OpenAI APIで$10以下、3分以内
- **相関**: Chatbot Arena (人間評価) とSpearman相関0.98

#### MT-Bench / LLM Judge
- **手法**: 80のマルチターン質問セット、GPT-4がスコア1-10で評価
- **カテゴリ**: 文章力, ロールプレイ, 推論, 数学, コーディング, 抽出, STEM, 人文科学
- **人間との一致率**: 80%以上
- **3K専門家投票 + 30K会話** が公開データセットとして利用可能

#### G-Eval
- **手法**: (1) LLMがCoTで評価ステップ生成 → (2) 生成ステップで出力評価 → (3) ログ確率で重み付きスコア
- **利点**: 自然言語で評価基準を定義可能
- **DeepEval統合**: `GEval` メトリクスとして5行で利用可能

#### Prometheus / Prometheus 2
- **手法**: カスタムスコアルーブリック対応のオープンソース評価LLM
- **Prometheus 2**: 直接評価 + ペアワイズランキングの両方に対応
- **7B版**: 16GB VRAMで動作 (消費者GPUで実行可能)
- **8x7B MoE版**: より高精度だがリソース要件が高い
- **Prometheus-Vision**: VLM向けの視覚評価モデル

#### JudgeLM
- **手法**: GPT-4判定データでファインチューンしたスケーラブルジャッジ
- **スケール**: 7B, 13B, 33Bパラメータ
- **速度**: JudgeLM-7Bで5Kサンプルを8xA100で3分
- **一致率**: 教師モデルとの一致率90%超 (人間間一致率を上回る)

---

## 6. エージェント統合適性分析

### エージェント自律実行に必要な要件

| 要件 | 説明 |
|---|---|
| プログラマティックAPI | CLIだけでなく、Pythonコードから呼び出し可能 |
| 設定のコード化 | YAML/JSON/Pythonで評価設定を定義可能 |
| 結果の構造化出力 | JSON/dictで結果を取得し、後続処理可能 |
| 非対話的実行 | 人間の入力なしで最初から最後まで実行可能 |
| エラーハンドリング | 例外処理が適切で、エージェントが失敗を検知可能 |
| GPU自動検出 | 利用可能なGPUを自動で使用 |
| 依存関係の軽さ | pip install で完結、特殊なシステム依存がない |

### Tier 1: エージェント統合に最適

| ライブラリ | 理由 |
|---|---|
| **lm-evaluation-harness** | `simple_evaluate()` で1行実行、dict形式の結果、60+ベンチマーク、vLLM対応 |
| **DeepEval** | pytest統合、エージェントトレース評価、シンプルAPI、カスタムメトリクス容易 |
| **jiwer** | GPU不要、`process_words()` で即計算、ASR評価の必須ツール |
| **Inspect AI** | エージェント評価専用設計、100+プリビルト、サンドボックス実行 |
| **torchmetrics** | 統一API、画像/音声/テキスト全対応、PyTorch統合 |

### Tier 2: エージェント統合に適している

| ライブラリ | 理由 |
|---|---|
| **lmms-eval** | lm-eval-harness互換だが設定がやや複雑、マルチモーダル唯一の包括的ツール |
| **Ragas** | RAG評価に特化、API依存あるが自動実行可能 |
| **clean-fid** | FID計算の正確性最高、Python APIシンプル |
| **ImageReward** | pip install + 3行、人間嗜好スコアを自動計算 |
| **Prometheus** | オープンソースLLM Judge、API不要だがGPU要件あり |
| **judges (Quotient AI)** | 軽量、既製ジャッジ、pip installで即利用 |
| **MTEB** | 埋め込みモデル評価に特化、HuggingFace統合 |
| **AlpacaEval** | 安価で高速、人間評価との高相関 |
| **TruLens** | OTel統合、エージェント対応だが設定がやや複雑 |

### Tier 3: エージェント統合に工夫が必要

| ライブラリ | 理由 |
|---|---|
| **HELM** | 包括的だが設定が複雑、実行時間が長い |
| **OpenAI Evals** | OpenAIモデル中心、エコシステム依存 |
| **Promptfoo** | Node.js中心、Python APIは間接的 |
| **bigcode-evaluation-harness** | コード実行のサンドボックスが必要、セキュリティ考慮 |
| **MT-Bench** | FastChat依存、単体ライブラリではない |
| **speechmetrics** | numpy互換性問題、メンテナンス限定的 |

---

## 7. モダリティ別推奨構成

### LLMテキスト評価
```
lm-evaluation-harness (ベンチマーク) + DeepEval (カスタム評価) + AlpacaEval (比較評価)
```

### マルチモーダル (VLM) 評価
```
lmms-eval (包括的ベンチマーク) + torchmetrics (個別メトリクス)
```

### 画像生成評価
```
clean-fid (FID/KID) + torchmetrics (CLIP Score/IS) + ImageReward (人間嗜好)
```

### 音声/TTS評価
```
UTMOSv2 (MOS予測) + pesq (PESQ) + pystoi (STOI) + jiwer + whisper-normalizer (WER)
```

### コード生成評価
```
bigcode-evaluation-harness (HumanEval/MBPP) + lm-evaluation-harness (その他)
```

### RAG評価
```
Ragas (RAG特化) + DeepEval (補完的メトリクス)
```

### エージェント軌跡評価
```
Inspect AI (エージェント評価) + AgentEvals (軌跡マッチング) + DeepEval v3 (トレース評価)
```

---

## 8. 参考リンク

- [LLM Evaluation Landscape 2026](https://research.aimultiple.com/llm-eval-tools/) - 主要フレームワーク比較
- [Awesome-LLMs-as-Judges](https://github.com/CSHaitao/Awesome-LLMs-as-Judges) - LLM-as-Judge論文の包括的リスト
- [Awesome-Evaluation-of-Visual-Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation) - 画像/動画生成評価のリスト
- [SHEET](https://github.com/unilight/sheet) - Speech Human Evaluation Estimation Toolkit
- [OpenAI Simple Evals](https://github.com/openai/simple-evals) - 軽量ベンチマーク参照実装
