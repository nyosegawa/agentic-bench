# モデル種別ごとの検証戦略調査 (2026-02)

## なぜ種別ごとの設計が必要か

モデル種別によって以下がすべて異なる:
- **入力データ**: テキスト、画像、音声、時系列データ、コード...
- **出力形式**: テキスト、画像、音声、動画、ベクトル、数値...
- **品質の測り方**: 自動指標が使えるもの、人間の目/耳が必要なもの
- **GPU 要件**: CPU で十分なものから 80GB VRAM が必要なものまで
- **コスト感**: 1 推論 0.001 秒のものから数分かかるものまで

---

## 種別一覧と検証設計

### 1. LLM (テキスト生成)

**例**: [Llama 3](https://huggingface.co/meta-llama), [Gemma 3](https://huggingface.co/google/gemma-3-27b-it), [Mistral](https://huggingface.co/mistralai), [Qwen](https://huggingface.co/Qwen), [DeepSeek](https://huggingface.co/deepseek-ai)

| 項目 | 内容 |
|------|------|
| 入力 | テキスト (プロンプト) |
| 出力 | テキスト |
| Smoke | プロンプトを投げて応答が返るか |
| Quality | 出力テキストを表示。自動評価は [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (MMLU, ARC 等) |
| Perf | tokens/sec, latency p50/p99 |
| GPU | 1B: CPU可, 7B: T4, 13B: V100/A10G, 30B+: A100, 70B+: A100-80GB or 量子化 |
| ツール | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [vLLM](https://github.com/vllm-project/vllm) benchmark, [LiteLLM](https://github.com/BerriAI/litellm) (API系) |
| 特記 | サイズ幅が極端に大きい（1B〜405B）。量子化(GGUF/GPTQ/AWQ)で要件が大きく変わる |

### 2. VLM (Vision-Language Model)

**例**: GPT-4o, Gemini Pro Vision, [LLaVA](https://huggingface.co/llava-hf), [InternVL](https://huggingface.co/OpenGVLab), [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

| 項目 | 内容 |
|------|------|
| 入力 | 画像 + テキスト |
| 出力 | テキスト |
| Smoke | サンプル画像を渡して説明文が返るか |
| Quality | 画像の説明精度を人間が確認。自動は VQA ベンチマーク |
| Perf | tokens/sec (画像入力時), latency |
| GPU | LLM とほぼ同じだが、画像エンコーダ分の VRAM が追加で必要 |
| ツール | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) |
| 特記 | サンプル画像のセットが必要。テスト画像の著作権に注意 |

### 3. 画像生成

**例**: [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large), [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev), DALL-E 3, [PixArt](https://huggingface.co/PixArt-alpha)

| 項目 | 内容 |
|------|------|
| 入力 | テキスト (プロンプト)。一部は画像 (img2img) |
| 出力 | 画像 |
| Smoke | プロンプトから画像が生成されるか |
| Quality | **生成画像を目で見る**のが最重要。自動は FID ([pytorch-fid](https://github.com/mseitzer/pytorch-fid)), CLIP Score ([torchmetrics](https://github.com/Lightning-AI/torchmetrics)) |
| Perf | 秒/枚, ステップ数, 解像度 |
| GPU | 8-24GB 典型。SDXL: 8GB, Flux: 12-24GB |
| ツール | [diffusers](https://github.com/huggingface/diffusers), [ComfyUI](https://github.com/comfyanonymous/ComfyUI) |
| 特記 | 同じプロンプトでも seed で結果が変わる。固定 seed でのテストと複数生成の両方が必要 |

### 4. 音声合成 (TTS)

**例**: [Bark](https://huggingface.co/suno/bark), [XTTS v2](https://huggingface.co/coqui/XTTS-v2), [Parler-TTS](https://huggingface.co/parler-tts), [F5-TTS](https://huggingface.co/SWivid/F5-TTS), StyleTTS2, Fish Speech

| 項目 | 内容 |
|------|------|
| 入力 | テキスト (+ 話者音声サンプルの場合も) |
| 出力 | 音声ファイル (.wav/.mp3) |
| Smoke | テキストから音声ファイルが生成されるか |
| Quality | **聴いて確認**が最重要。自動は MOS 予測 ([UTMOS](https://github.com/sarulab-speech/UTMOS22)), 自然さ |
| Perf | RTF (Real-Time Factor), レイテンシ |
| GPU | 小さいもの多い。CPU 可のものも。GPU なら T4 で十分なことが多い |
| ツール | 各モデル固有のスクリプト |
| 特記 | 多言語対応の確認（日本語テスト必須）。ストリーミング対応の有無 |

### 5. 音声認識 (STT)

**例**: [Whisper](https://huggingface.co/openai/whisper-large-v3), [Canary](https://huggingface.co/nvidia/canary-1b), [wav2vec2](https://huggingface.co/facebook/wav2vec2-large-960h), Conformer

| 項目 | 内容 |
|------|------|
| 入力 | 音声ファイル |
| 出力 | テキスト (書き起こし) |
| Smoke | サンプル音声を書き起こせるか |
| Quality | WER (Word Error Rate), CER。正解テキストとの比較。評価には [jiwer](https://github.com/jitsi/jiwer) 等を利用 |
| Perf | RTF, レイテンシ, バッチスループット |
| GPU | Whisper large: GPU 推奨, small/medium: CPU 可 |
| ツール | [whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [NeMo](https://github.com/NVIDIA-NeMo/NeMo) |
| 特記 | テスト用音声データが必要。言語ごとの WER 比較 |

### 6. 音声/音楽生成

**例**: [MusicGen](https://huggingface.co/facebook/musicgen-large), [AudioCraft](https://github.com/facebookresearch/audiocraft), [Stable Audio](https://huggingface.co/stabilityai/stable-audio-open-1.0), Riffusion

| 項目 | 内容 |
|------|------|
| 入力 | テキスト (+ 参照音声の場合も) |
| 出力 | 音声ファイル |
| Smoke | プロンプトから音声が生成されるか |
| Quality | **聴いて確認**。自動は FAD (Frechet Audio Distance) ([frechet-audio-distance](https://github.com/gudgud96/frechet-audio-distance), [Microsoft FADTK](https://github.com/microsoft/fadtk)) |
| Perf | 秒/生成秒数, RTF |
| GPU | 8-16GB 典型 |
| 特記 | 生成が遅い（数分かかることも）。コスト注意 |

### 7. 動画生成

**例**: [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B), [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo), [CogVideo](https://huggingface.co/THUDM/CogVideoX-5b), [AnimateDiff](https://huggingface.co/guoyww/animatediff), Sora

| 項目 | 内容 |
|------|------|
| 入力 | テキスト (+ 画像の場合も) |
| 出力 | 動画ファイル (.mp4) |
| Smoke | プロンプトから動画が生成されるか |
| Quality | **見て確認**。自動は FVD (Frechet Video Distance) |
| Perf | 秒/フレーム, 総生成時間 |
| GPU | **非常に高い。24-80GB+**。最もコストがかかる種別 |
| 特記 | 1回の生成に数分〜数十分。コスト管理が最重要。無料枠では厳しい場合が多い |

### 8. Embedding モデル

**例**: text-embedding-3, [BGE](https://huggingface.co/BAAI/bge-m3), [GTE](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5), [E5](https://huggingface.co/intfloat/e5-large-v2), Cohere Embed

| 項目 | 内容 |
|------|------|
| 入力 | テキスト (or 画像) |
| 出力 | ベクトル (数値配列) |
| Smoke | テキストを埋め込んでベクトルが返るか、次元数が正しいか |
| Quality | [MTEB](https://github.com/embeddings-benchmark/mteb) ベンチマーク (検索精度、分類精度等) |
| Perf | embeddings/sec, バッチスループット |
| GPU | 小さい。CPU でも十分なことが多い |
| ツール | [MTEB](https://github.com/embeddings-benchmark/mteb), [sentence-transformers](https://github.com/huggingface/sentence-transformers) |
| 特記 | **目で見る出力がない**。数値の類似度比較やランキングで評価。コスト最小の種別 |

### 9. 時系列予測

**例**: [TimesFM](https://huggingface.co/google/timesfm-2.0-500m-pytorch), [Chronos](https://huggingface.co/amazon/chronos-t5-large), [Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama), [PatchTST](https://huggingface.co/ibm/patchtst-etth1-forecast), [Moirai](https://huggingface.co/Salesforce/moirai-1.1-R-large)

| 項目 | 内容 |
|------|------|
| 入力 | 時系列データ (数値配列) |
| 出力 | 予測値 (数値配列) |
| Smoke | サンプルデータで予測が返るか |
| Quality | MAE, RMSE, MASE。**予測 vs 実績をグラフで可視化** |
| Perf | 推論時間, バッチスループット |
| GPU | 小さめ。CPU 可のものも多い |
| ツール | [GluonTS](https://github.com/awslabs/gluonts), 各モデル固有 |
| 特記 | データセット依存が大きい。ドメイン（株価、気象、需要等）で性能が全く異なる |

### 10. 物体検出 / セグメンテーション

**例**: [YOLOv11](https://github.com/ultralytics/ultralytics), [SAM 2](https://huggingface.co/facebook/sam2-hiera-large), [DETR](https://huggingface.co/facebook/detr-resnet-50), [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)

| 項目 | 内容 |
|------|------|
| 入力 | 画像 (or 動画) |
| 出力 | バウンディングボックス / マスク |
| Smoke | サンプル画像で検出結果が返るか |
| Quality | mAP, IoU。**検出結果を画像上にオーバーレイして可視化** |
| Perf | FPS, レイテンシ |
| GPU | 小〜中。YOLO は T4 で十分、SAM は A10G 推奨 |
| ツール | [ultralytics](https://github.com/ultralytics/ultralytics), [MMDetection](https://github.com/open-mmlab/mmdetection) |
| 特記 | リアルタイム性が求められる場合がある。動画処理の FPS が重要 |

### 11. コード生成

**例**: [DeepSeek Coder](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct), [CodeLlama](https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf), [StarCoder2](https://huggingface.co/bigcode/starcoder2-15b), [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)

| 項目 | 内容 |
|------|------|
| 入力 | テキスト (コードプロンプト) |
| 出力 | テキスト (コード) |
| Smoke | コード補完が返るか |
| Quality | HumanEval pass@k, MBPP。**生成コードを実際に実行して検証** |
| Perf | tokens/sec (LLM と同じ) |
| GPU | LLM と同じ |
| ツール | [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) |
| 特記 | 生成コードの実行にはサンドボックスが必要（セキュリティ） |

### 12. 3D 生成

**例**: [TripoSR](https://huggingface.co/stabilityai/TripoSR), [Point-E](https://huggingface.co/openai/point-e-large), [Shap-E](https://huggingface.co/openai/shap-e), [InstantMesh](https://huggingface.co/TencentARC/InstantMesh)

| 項目 | 内容 |
|------|------|
| 入力 | 画像 or テキスト |
| 出力 | 3D メッシュ (.obj/.glb) |
| Smoke | 入力から 3D モデルが生成されるか |
| Quality | **3D ビューアで確認**。自動は Chamfer Distance 等 |
| Perf | 生成時間 |
| GPU | 8-24GB |
| 特記 | HTML レポートに three.js で 3D ビューア埋め込みが理想 |

---

## 種別横断の共通パターン

### GPU 要件マッピング

| 種別 | 典型 VRAM | 最安プロバイダ候補 |
|------|----------|-----------------|
| LLM (≤7B) | ≤8GB | Colab Free / HF ZeroGPU |
| LLM (7-13B) | 8-16GB | Colab Pro (T4/V100) |
| LLM (13-30B) | 16-40GB | Colab Pro (A100) / Modal |
| LLM (30B+) | 40-80GB+ | Modal / beam.cloud |
| 画像生成 | 8-24GB | Colab Pro (T4-A100) |
| TTS / STT | ≤8GB | Colab Free / CPU / HF API |
| 動画生成 | 24-80GB+ | Modal / beam.cloud (高コスト注意) |
| Embedding | ≤4GB | CPU / HF API |
| 時系列 | ≤8GB | CPU / Colab Free |
| 物体検出 | 4-16GB | Colab Pro (T4) |
| コード生成 | LLM と同じ | LLM と同じ |
| 3D 生成 | 8-24GB | Colab Pro |

### 品質評価の自動化レベル

| レベル | 種別 | 方法 |
|--------|------|------|
| 完全自動 | Embedding ([MTEB](https://github.com/embeddings-benchmark/mteb)), STT (WER), コード (pass@k), 時系列 (MAE) | 正解データとの数値比較 |
| 半自動 | LLM ([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), 物体検出 (mAP) | 標準ベンチマーク |
| 人間判定必須 | 画像生成, TTS, 音楽生成, 動画生成, 3D | 目/耳で確認が基本 |

→ **人間判定必須の種別こそ、HTML レポートの価値が高い**

### HTML レポートでの表示方法

| 出力タイプ | HTML 要素 | 備考 |
|-----------|----------|------|
| テキスト | `<pre>` / Markdown | LLM, コード生成 |
| 画像 | `<img>` | 画像生成, 物体検出 (オーバーレイ) |
| 音声 | `<audio controls>` | TTS, STT (元音声), 音楽生成 |
| 動画 | `<video controls>` | 動画生成 |
| 3D | three.js viewer | 3D 生成 |
| グラフ | Chart.js / Plotly | 時系列, Embedding (t-SNE), 性能比較 |
| 数値テーブル | `<table>` | 全種別の metrics |
