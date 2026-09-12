# 修士論文ドラフト

更新日: 2026-09-12

Markdownで下書きし、最終稿はLaTeXへ移す。現在は第3章、第4章のLLA・補正LLA、付録A〜Cの定義・実装に基づく草稿を収録する。原稿の存在は、実験や診断の完了を意味しない。

| 目的 | 参照先 |
| --- | --- |
| 現在の本文を通して読む | 下の第3章一覧を3.1から順に読む |
| 論文全体の論旨・各章の問いを確認する | [構成案](outline.md) |
| 次に書く範囲・資料・図表を確認する | [執筆計画](writing_plan.md) |
| 本文と付録の記号を照合する | [共通記号表](notation.md) |
| 研究条件・採用理由・実施記録を確認する | [研究文書の案内](../docs/README.md) |

## 第3章 解析手法

| 節 | 原稿 | 内容 |
| --- | --- | --- |
| 3.1 | [解析の全体像と入出力](chapters/3_analysis_methods/overview.md) | 解析の目的、学習と利用の違い、全体フローの参考図 |
| 3.2 | [共通前処理とSNVスペクトル](chapters/3_analysis_methods/preprocessing.md) | マスク、反射率、波長範囲、補間、有効画素、SNV |
| 3.3 | [SNV制約を保つスペクトル摂動](chapters/3_analysis_methods/spectral_augmentation.md) | TGN・shiftの定義、強度・確率、仮定 |
| 3.4 | [Masked denoisingによる表現学習](chapters/3_analysis_methods/representation_learning.md) | Patch、encoder、単位潜在、decoder、loss、全可視抽出 |
| 3.5 | [Cosineクラスタリングと空間マッピング](chapters/3_analysis_methods/clustering_mapping.md) | B0・PCA・NN表現、中心推定、固定中心への割当、座標への配置 |

## 第4章 比較実験・評価および解釈の設計

| 節 | 原稿 | 現在書いた範囲 |
| --- | --- | --- |
| 4.4 | [主評価と補助診断](chapters/evaluation_protocol/metrics_aggregation.md) | LLAの畳み込み表現、有効近傍対による正規化、占有率による補正、未定義条件と解釈範囲 |

比較条件、他の指標、集計、解釈設計の本文は未執筆。

## 付録と本文の対応

| 付録 | 原稿 | 現在書いた範囲 |
| --- | --- | --- |
| A | [前処理の定義と確認資料](appendices/preprocessing_diagnostics.md) | マスク、SNR proxy、補間、除外規約、既存のcutoff図 |
| B | [数理的補足](appendices/mathematical_details.md) | SNV、TGN、shift、masked loss、入力差と潜在差、クラスタの分離・広がり、摂動応答と損失の関係 |
| C | [モデル・学習・数値設定](appendices/implementation_details.md) | モデル表、条件表、学習率、抽出・クラスタリングの数値設定、再現記録 |

付録B.5は、MAE群での変動の強調・抑制を調べるための数理を扱う。[補助診断の実施計画](../docs/design/representation_geometry_diagnostics.md)の詳細条件はOpenであり、診断は未実施である。付録D（評価・集計）とE（補足結果）も未執筆。

## 図と再作図用の記録

- [解析フロー参考図](figures/analysis_workflow_reference.png)：画像生成による一枚図。実測データや結果を表すものではない。
- [作図記録](figures/analysis_workflow_notes.md)：生成promptと最終作図時の修正事項。図3.1のcaption案は第3.1節に置く。

## 原稿の読み方と未整備事項

本文末の「執筆メモ」に参照資料・照合先・未整備事項を置く。数式はKaTeX互換の記法と仮の式番号を用い、LaTeX移行時に章・節・式・図への参照をlabel/refへ置き換える。

Materials・Measurements、標準手法の引用・書誌、最終実行環境の追記が必要である。執筆上の残作業は[執筆計画](writing_plan.md)、研究の進捗と確認結果は[ToDo](../ToDo.md)・[検証履歴](../docs/verification_history.md)を参照する。
