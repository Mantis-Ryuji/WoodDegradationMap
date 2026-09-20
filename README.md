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

主7条件の5-fold・3反復の学習・クラスタリング・評価、`main_oof_v1`のOOF集計、
主条件の図表生成・出力確認まで完了しました。図は2行3列のPNG 3枚、表はCSV 11個で、
再生成方法と保存先は[主条件OOF図表](docs/experiment_runbook.md#oof-reporting)を参照してください。
代表指標の優先順はLLA（補正後）、LFR(TGN+FS)、ARI、Cosine-Silhouette、Cluster Occupancyです。

全体fitはB0/PCAの準備・保存復元確認、3条件のGPU smoke、A0・M00・M11の各800 epochまで完了しました。
2026-09-20に保存記録を確認し、`training-check`の完了はユーザー報告に基づいて記録しました。
後続の表現抽出・Cosine-KMeans・全画素予測は`global_cluster.py`、PNG・CSV生成は`visualize_global.py`へ分けて実装しました。
合成データのCPU検証済みで、本番実行はこれからです。[全体fit後の手順](docs/experiment_runbook.md#global-post-fit)に実行コマンドと保存先を記載しています。
5条件のクラスタマップ・代表二次微分スペクトルを確認してから、帯域とUMAP設定を決め、潜在空間・空間mapとの対応と化学的解釈を進めます。
2026-09-20の指定により、mask率sweepとvMFは低優先度で計画に残し、この解析と図表・解釈の整理を一通り終えた後に回します。
vMFの数値仕様・共通処理の検証と全体fit用5 fitsも後回しとし、先行する可視化の前提にはしません。
完了範囲と作業順、図表生成・全体fit・補助実験の残作業は[ToDo](ToDo.md)で管理します。

## 主な配置

```text
data/raw/                              raw原本
data/processed/production_v1/           前処理済み入力
outputs/preprocessing/production_v1/    前処理確認図
outputs/experiments/preflight_v1/       動作確認の成果物
outputs/experiments/production_v1/      本番manifest・結果・checkpoint
outputs/experiments/global_v1/          全体fit専用manifest・結果・checkpoint
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
