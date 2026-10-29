<h1 align="center">WoodDegradationMap</h1>

[![Python](https://img.shields.io/badge/python-3.11.15-blue)](./.python-version)
[![chemomae](https://img.shields.io/badge/chemomae-0.2.2-orange)](https://github.com/Mantis-Ryuji/ChemoMAE)
[![CI](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

古材の近赤外ハイパースペクトル画像（NIR-HSI）から、化学状態の違いを探索するための研究コードです。
自己教師あり表現学習手法ChemoMAEで画素ごとのスペクトルを圧縮し、クラスタリングによって試料表面の領域マップを作成します。

対応論文：*Self-Supervised Spectral Learning Reveals Chemical Heterogeneity in Historic Wood through Near-Infrared Hyperspectral Imaging*

## 研究の概要

ChemoMAEは、SNV変換後の平均・ノルムを保つTangent Gaussian Noise（TGN）とFractional Shift（FS）を入力に加え、
可視帯域からマスク帯域の摂動前の値を復元します。学習した16次元表現をCosine-KMeansで分割し、
クラスタ割当を画像上へ戻します。座標や近傍画素は学習・クラスタリングの入力に用いません。

古材49試料を対象に、次の解析を行います。

- **未知試料での評価**：試料単位の5-fold交差検証・3反復で、入力の直接利用、PCA、AE、denoising AE、ChemoMAEを比較します。局所ラベル一致を偶然一致で補正したLLAと、指定摂動による割当変化を測るLFRを中心に評価します。
- **共通の基準によるマッピング**：全49試料で表現とクラスタ中心を求め、主要6条件・K=8の分布を比較します。M11では学習・クラスタリングの反復による領域と境界の再現性も調べます。
- **NIRスペクトルによる解釈**：クラスタの空間分布と観測スペクトルを、文献帰属・材組織・環境履歴・試料情報に対応付けます。劣化過程のモデルは観測に基づく仮説として扱います。
- **補助実験**：mask率の感度解析と、Raman Bacteria-IDにおけるクラスタリング・教師あり分類を扱います。

評価指標はマップの性質を測るもので、画素単位の劣化検出精度や化学成分量を示しません。
研究の根拠と解釈の範囲は[研究概要](docs/research_overview.md)にまとめています。

## ドキュメント

| 目的 | 参照先 |
| --- | --- |
| 研究の目的・方法・結果の範囲を知る | [研究概要](docs/research_overview.md) |
| 前処理・条件・指標の定義を確認する | [研究設計](docs/design/README.md) |
| 実験と図表を再現する | [実験手順](docs/experiment_runbook.md) |
| 解析ごとの成果物を探す | [成果物案内](docs/results.md) |
| 数理・実装・関連研究を読む | [ドキュメント一覧](docs/README.md) |

## 環境

Python 3.11.15と`uv`を使用します。依存関係は[pyproject.toml](pyproject.toml)と[uv.lock](uv.lock)で固定し、
PyTorchはCUDA 13.0用の配布元を指定しています。

```powershell
uv sync --locked --dev --group bacteria-hpo
uv run --no-sync pytest
```

コマンドはリポジトリのルートで実行します。学習にはGPUを用います。
入力データの配置、前処理、学習、評価、再開条件は[実験手順](docs/experiment_runbook.md)を参照してください。
Raman補助実験は[クラスタリング](docs/bacteria_id_runbook.md)と[分類・HPO](docs/bacteria_id_hpo_runbook.md)に分けて説明しています。

## ディレクトリとデータ

```text
src/wood_degradation_map/        前処理・学習・クラスタリング・評価
scripts/                        実験・図表生成のCLI
tests/                          テスト
docs/                           研究仕様・再現手順・技術的補足
data/                           原本・前処理済みデータ（Git管理対象外）
outputs/                        実験記録・数値結果・図表（Git管理対象外）
```

本リポジトリにはコード・設定・文書を収録します。古材データ、実験出力、学習済み重みは同梱しません。
古材解析の再実行には、[前処理仕様](docs/design/preprocessing.md)の入力を別途用意する必要があります。
Raman Bacteria-IDの出典と入力構成は[補助実験仕様](docs/design/bacteria_id_experiment_plan.md)を参照してください。
文書中の`data/`・`outputs/`以下のパスは、ローカルで配置・生成するファイルを指します。

## ライセンス

本リポジトリのコードは[MIT License](LICENSE)です。
基盤ライブラリ[ChemoMAE](https://github.com/Mantis-Ryuji/ChemoMAE)と第三者データには、それぞれのライセンス・利用条件が適用されます。
