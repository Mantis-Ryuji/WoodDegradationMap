# 修士論文ドラフト

更新日: 2026-09-12

Markdownで本文と数式を下書きし、最終稿はLaTeXへ移す。現在は第3章「解析手法」の第1稿と、付録A〜Cのうち現行の定義・実装から書ける部分を収録している。章構成は[構成案](outline.md)、今後の順序と資料対応は[執筆計画](writing_plan.md)を参照する。

## 第3章 解析手法

| 節 | 原稿 | 内容 |
| --- | --- | --- |
| 3.1 | [解析の全体像と入出力](chapters/3_analysis_methods/overview.md) | 解析の目的、学習と利用の違い、全体フローの参考図 |
| 3.2 | [共通前処理とSNVスペクトル](chapters/3_analysis_methods/preprocessing.md) | マスク、反射率、波長範囲、補間、有効画素、SNV |
| 3.3 | [SNV制約を保つスペクトル摂動](chapters/3_analysis_methods/spectral_augmentation.md) | TGN・shiftの定義、強度・確率、仮定 |
| 3.4 | [Masked denoisingによる表現学習](chapters/3_analysis_methods/representation_learning.md) | Patch、encoder、単位潜在、decoder、loss、全可視抽出 |
| 3.5 | [Cosineクラスタリングと空間マッピング](chapters/3_analysis_methods/clustering_mapping.md) | B0・PCA・NN表現、中心推定、固定中心への割当、座標への配置 |

節番号順に読むと一つの章になる。第4章へ渡す比較条件や評価規約は必要な範囲を要約し、第4章そのものは今回の執筆対象に含めていない。

## Appendix

| 付録 | 原稿 | 現在書いた範囲 |
| --- | --- | --- |
| A | [前処理の定義と確認資料](appendices/preprocessing_diagnostics.md) | マスク、SNR proxy、補間、除外規約、既存のcutoff図 |
| B | [数理的補足](appendices/mathematical_details.md) | SNV、TGN、shift、masked lossの期待値、潜在とPCA、decoderの距離・平均・normの成立条件 |
| C | [モデル・学習・数値設定](appendices/implementation_details.md) | モデル表、条件表、学習率、抽出・クラスタリングの数値設定、再現記録 |

付録B.5には、[SNV方向情報とPCA score](appendices/mathematical_details.md#snv-coordinate-learning)、[decoderが定める距離](appendices/mathematical_details.md#decoder-distance)、[出力平均](appendices/mathematical_details.md#decoder-mean)・[一定normの成立条件](appendices/mathematical_details.md#decoder-norm)を補足した。議論の出典と、会話から修正した仮定は付録末尾の執筆メモに記録している。

付録Dの評価・集計と、付録Eの補足結果は未執筆。vMF数値仕様、全体fitの実施結果、FT-IR詳細を未確認のまま補完していない。

## 図と再作図用の記録

- [解析フロー参考図](figures/analysis_workflow_reference.png)：画像生成による一枚図。実測データや結果を表すものではない。
- [生成prompt・確認事項・最終作図時の整理](figures/analysis_workflow_notes.md)：生成と修正のpromptを保存。
- 図3.1のcaption案は第3.1節に含めた。英文ラベルは構成検討用であり、最終作図では本文の言語・用語とそろえる。

## この第1稿の状態

本文は論文調の文章として記述し、各節末の「執筆メモ」を原稿本文と分けた。学習・評価の完了や性能改善を先取りせず、定義・設計・既存の前処理記録を区別している。

数式記号の対応と初出箇所は[共通記号表](notation.md)にまとめた。同じ量は本文と付録で同じ記号を用い、各節でも初出時に意味を説明する。図A.1には各パネルの集計順・分母・対象画素と読み取り方を追記した。

数式はKaTeX互換の記法と仮の式番号を用いる。LaTeX移行時には章・節・式・図への参照をlabel/refへ置き換える。Markdown段階では節へのリンクと式番号で参照する。

標準手法の原典の引用、書誌情報、Materials・Measurements、最終の実行環境の記載は追記が必要である。参照したローカル文書・コードは各原稿の末尾に残した。今回、文献の新規調査、学習・評価・前処理やその検証コードの実行は行っていない。
