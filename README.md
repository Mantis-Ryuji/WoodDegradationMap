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
| Thesisで執筆を始める | [執筆への引き継ぎ](docs/manuscript_handoff.md) |
| 実行・再開・完了確認 | [実験runbook](docs/experiment_runbook.md) |
| 採用理由・関連研究・解釈の全体案内 | [ドキュメント案内](docs/README.md) |

## 現在の段階

主8条件のCV・OOF図表、全体fitの6条件・K8マップとスペクトル・PCA座標、
Raman Bacteria-IDの[Table 4](docs/bacteria_id_runbook.md)と
[Table 5](docs/bacteria_id_hpo_runbook.md)は完了しました。
mask率25%・50%・75%のCV・OOF比較と、全体fitのM11(-50)・M11-75の付録用比較も生成済みです。
残る実験は位置対応FT-IRです。進捗と成果物は[ToDo](ToDo.md)、
実行・再開方法は[runbook](docs/experiment_runbook.md)にまとめています。
mask率比較の引き渡しファイルは[Thesis向け一覧](docs/manuscript_handoff.md#mask-rate-handoff)を参照してください。

## FT-IRによる化学的解釈の暫定方針

2026-10-07時点の案です。古材49試料のうち、事前知識に基づいて劣化に関する比較の問いを立てられる試料を6試料程度選定します。
標準構成M11のクラスタの空間分布を、同一試料内のNIRスペクトル差と局所のFT-IRの所見に結び付け、
化学状態の空間分布とその構造を探索・解釈する実例を示します。
FT-IRの対象は、条件間のマップ比較用に既選定の7試料とは別の目的で選び、以下を候補選定の視点とします。

| 選定の視点 | 試料内で比較する問い |
| --- | --- |
| 地面からの影響が疑われる材（ピンク色が見られる候補材を含む） | 地面との位置関係に関する事前知識を踏まえ、影響が想定される領域と周囲のスペクトル差を調べる。 |
| 虫食いが激しく、虫孔周辺に異なる劣化の関与が疑われる材 | 虫孔周辺と離れた部位の化学的特徴を比較し、その差の空間的な広がりを調べる。 |
| 部材表面の気象劣化と内部の経年変化を比較できそうな材 | 外部露出面からの位置に応じた化学的特徴とクラスタの空間分布の違いを調べる。内部を未劣化の基準とは仮定しない。 |

3つの視点に各2試料をそろえることは必須とせず、試料内の比較部位を確保できることを重視します。
選定理由と比較部位はFT-IR測定前に記録し、測定部位をNIR-HSI画像上の位置と対応付けます。
想定する劣化の関与は仮説として扱い、期待した差が得られなかった結果やNIR・FT-IRの所見の不一致も含めて報告します。

掲載は本文で2〜3試料を詳述し、残りを付録に簡潔に示す案とします。
全49試料の網羅的な化学分析と劣化機構の体系的な解釈は、別論文としての展開を検討します。
具体的な試料ID、測定部位、最終的な分析・掲載試料数は未確定です。

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
