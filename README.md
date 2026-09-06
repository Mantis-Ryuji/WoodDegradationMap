<h1 align="center">WoodDegradationMap</h1>

[![Python](https://img.shields.io/badge/python-3.11.15-blue)](./.python-version)
[![chemomae](https://img.shields.io/badge/chemomae-0.2.2-orange)](https://github.com/Mantis-Ryuji/ChemoMAE)
[![CI](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 古材の近赤外ハイパースペクトル画像を教師なしで領域分割し、表現空間上のクラスタ品質、クラスタの空間的一貫性、ノイズ摂動に対するクラスタラベルの安定性を比較する研究リポジトリです。

## 現在の段階

本番入力は `data/processed/production_v1/`、本番実験rootは
`outputs/experiments/production_v1/` です。前処理、実験pipeline、評価・OOF集計の実装と
preflightは完了し、production CVを実行中です。runごとの完了状況と次の作業は
[ToDo.md](ToDo.md)、実行方法は[実験runbook](docs/experiment_runbook.md)を参照してください。

vMFクラスタリングの補助実験は主7条件・5-fold・3反復・共通7Kの735 fitsを計画しています。
追加ニューラル学習はありません。v0.2.2の数値検証・設定確定と補助実験pipelineの実装が残っています。
範囲と数値仕様は[実験プロトコル第5.2節](docs/design/experiment_protocol.md#vmf-supplementary)、
評価方法は[評価指標第8.4節](docs/design/evaluation_metrics.md#vmf-evaluation)で管理します。

## 文書

| 文書 | 役割 |
| --- | --- |
| [研究設計](docs/design/README.md) | 固定した研究条件と各設計文書への入口 |
| [前処理仕様](docs/design/preprocessing.md) | 200 Hz入力、mask、反射率、SNV、保存schema |
| [実験プロトコル](docs/design/experiment_protocol.md) | split、学習、主実験・mask率・vMF補助実験、実行順序 |
| [評価指標](docs/design/evaluation_metrics.md) | LLA、LFR、silhouette、ARI、集約と比較 |
| [可視化と解釈](docs/design/visualization_and_interpretation.md) | 全体学習、ラベル整列、本文代表例、解釈上の制約 |
| [ChemoMAEの位置づけ](docs/chemomae_positioning.md) | PCAとの関係、構成の特徴と限界、関連研究と出典 |
| [実験runbook](docs/experiment_runbook.md) | 本番CLI、再開、完了判定、OOF作成 |
| [検証履歴](docs/verification_history.md) | テスト、preflight、本番runと環境更新の確認結果 |
| [ToDo](ToDo.md) | 現在の実行状態と残作業 |

設計文書を研究条件の正とし、runbookには運用手順、検証履歴には実行済みの工学的確認だけを置きます。

## 主な配置

```text
data/raw/                          raw原本
data/processed/production_v1/     本番前処理済みデータ
outputs/preprocessing/production_v1/
                                   前処理の確認図
outputs/experiments/preflight_v1/ 動作確認の成果物
outputs/experiments/production_v1/
                                   本番manifest・数値結果・図・checkpoint
src/wood_degradation_map/         実装
scripts/                          CLI
tests/                            テスト
```

Python環境とコマンド実行には `uv` を使用します。

```powershell
uv sync
uv run pytest
```

`outputs/` 内のconfig、manifest、数値結果、図はGit管理対象です。モデル重み、checkpoint、
optimizer stateは容量が大きいためGit管理対象外です。
