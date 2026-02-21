# Vast.ai & RunPod 調査 (2026-02)

## Vast.ai

- **モデル**: 分散マーケットプレイス（個人・データセンターがGPUを出品、価格変動あり）
- **課金**: 秒単位。On-demand / Reserved / Interruptible（ビッド方式、最安）
- **代表的価格**: T4 $0.05-0.15/hr, A100-40GB $0.52-1.00/hr, H100 $1.49-2.50/hr
- **認証**: `VAST_API_KEY` 環境変数。CLI/SDK両方対応
- **SDK**: `pip install vastai-sdk` (vastai_sdk.VastAI)。検索→インスタンス作成→SSH/SCP→破棄
- **環境**: Docker コンテナベース。SSH/Jupyterアクセス
- **無料枠**: なし（スタートアップ向け$2,500クレジットプログラムあり）
- **特徴**: 最安値GPU（マーケットプレイス競争）、コンシューマGPU(3090,4090)も利用可能
- **欠点**: コールドスタートが分単位、ホスト品質にばらつき、DXはModal/beamより低い

## RunPod

- **モデル**: マネージドクラウド（Pods = 永続VM、Serverless = オートスケール）
- **課金**: 秒単位。Community Cloud（安い）vs Secure Cloud（高信頼）
- **代表的価格**: RTX4090 $0.34/hr, A100-80GB $1.19-1.39/hr, H100 $2.69/hr
- **認証**: `RUNPOD_API_KEY` 環境変数。`rpa_` プレフィックス
- **SDK**: `pip install runpod`。Pod作成 or Serverlessエンドポイント
- **環境**: Docker + テンプレート (runpod/pytorch等)。SSH/Jupyterアクセス
- **無料枠**: なし（$10使うと$5-500ランダムボーナス。スタートアッププログラムあり）
- **特徴**: Pods+Serverless両対応、FlashBoot(1-2秒コールドスタート)、20+GPU種類
- **欠点**: T4はPodで非対応（A4000以上）、Serverlessは要Dockerfile事前ビルド

## agentic-bench での活用方針

- 両者とも **Pod（インスタンス）方式** が agentic-bench に適合
  - Modal/beam のようなデコレータベースではなく、SSH + スクリプト実行
  - Docker イメージ + onstart スクリプトで環境構築→推論→結果取得→破棄
- Vast.ai は **最安プロバイダ** として位置付け（マーケットプレイス価格）
- RunPod は **バランス型** として位置付け（安定性 + 価格 + Serverless対応）
