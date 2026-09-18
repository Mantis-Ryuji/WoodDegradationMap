<h1 align="center">WoodDegradationMap</h1>

[![Python](https://img.shields.io/badge/python-3.11.15-blue)](./.python-version)
[![chemomae](https://img.shields.io/badge/chemomae-0.2.2-orange)](https://github.com/Mantis-Ryuji/ChemoMAE)
[![CI](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 古材の近赤外ハイパースペクトル画像からスペクトル表現を学び、教師なしで試料表面の領域をマッピングする研究リポジトリです。

中心となる問いは、化学状態の正解ラベルを与えずに帯域間の関係を学ぶことで、
状態の違いを探索・解釈できる表現を得られるか、です。マスク再構成にSNV制約を保つdenoisingを組み合わせ、
得られたマップの空間的一貫性・指定摂動への安定性・反復間再現性を試料単位CVで比較します。
化学的な意味はNIRスペクトルと位置対応FT-IRから検討する計画です。
FT-IRの詳細設計・結果と、他材料への有効性は未確認です。

## 読み始める場所

| 目的 | 文書 |
| --- | --- |
| 研究の問いと全体像 | [研究の目的と説明文](docs/research_overview.md) |
| 固定条件・評価・可視化の設計 | [研究設計](docs/design/README.md) |
| 現在の進捗と残作業 | [ToDo](ToDo.md) |
| 実行・再開・完了確認 | [実験runbook](docs/experiment_runbook.md) |
| 採用理由・関連研究・解釈の全体案内 | [ドキュメント案内](docs/README.md) |

## 現在の段階

主7条件の5-fold・3反復の学習・クラスタリング・評価は完了し、次はOOF集計です。
OOF集計後は主条件の図表生成と全体fit・可視化・解釈を先に進め、実装と出力確認を固めてから
mask率・vMFのCV補助実験へ移ります。全体fit用vMFの数値仕様・共通処理の検証は先行します。
完了範囲と作業順、図表生成・全体fit・補助実験の残作業は[ToDo](ToDo.md)で管理します。

## 主な配置

```text
data/raw/                              raw原本
data/processed/production_v1/           前処理済み入力
outputs/preprocessing/production_v1/    前処理確認図
outputs/experiments/preflight_v1/       動作確認の成果物
outputs/experiments/production_v1/      本番manifest・結果・checkpoint
outputs/sanity_checks/                 探索的な確認図・数値
docs/                                  設計・手順・研究説明
src/wood_degradation_map/               実装
scripts/                               CLI
tests/                                 テスト
```

## 環境と検証

Pythonの指定は[.python-version](.python-version)、依存関係は[pyproject.toml](pyproject.toml)と`uv.lock`にあります。
環境構築には`uv`を使用します。

```powershell
uv sync
uv run --no-sync pytest
```

本番の実行手順は[runbook](docs/experiment_runbook.md)を参照してください。
`data/`は再配布せず、`outputs/`内のconfig・manifest・数値結果・図はGit管理対象です。
重み・checkpointはGit管理対象外とし、[保存規約](docs/experiment_runbook.md#artifact-records)に従います。
