# 設計上の決定記録

本書は現行仕様に至る主要な判断と決定時点をまとめる。固定値・数式はリンク先の設計文書を正とする。
実行済みの工学的確認は[検証履歴](../verification_history.md)、未検証の原因候補は[解釈メモ](../interpretation_notes.md)へ分ける。

## 主要な決定

| 決定日 | 決定内容 | 決定時点・適用範囲 | 現行仕様 |
| --- | --- | --- | --- |
| 2026-09-05 | 採用49試料をKYOw単位で分割。異なるKYOw間の同一原材関係は不明のまま扱う | CV用split作成前のユーザー確認。由来的な独立性を確認したという意味ではない | [試料・CV](experiment_protocol.md#cv-design) |
| 2026-09-05 | CV用seed計画の基準値を20260905へ固定 | 実装時・CV開始前。split・抽出・反復・摂動の用途を分離 | [seed規約](experiment_protocol.md#seed-plan) |
| 2026-09-06 | 補間後・SNV前に負の反射率を含む画素を背景化 | train・test共通で適用し、production_v1を再生成 | [前処理](preprocessing.md) |
| 2026-09-06 | 各樹種で保存有効画素数が最大の7試料を本文表示例に固定 | 結果を見る前の表示用選択。評価対象49試料は変更しない | [代表試料](visualization_and_interpretation.md#representative-samples) |
| 2026-09-07 | 主7条件のvMF補助実験735 fits、v0.2.2採用、退化成分の扱いを確定 | A0の一部run完了後。vMF結果を見る前に残る数値仕様を確定する | [vMF補助実験](experiment_protocol.md#vmf-supplementary) |
| 2026-09-08 | 全体解釈にもvMFを追加 | CV開始後。全体学習の表現を再利用し、追加の表現学習なし | [全体可視化](visualization_and_interpretation.md) |
| 2026-09-09 | 全体学習・可視化へA0を追加 | CV開始後・全体学習前。5条件×2手法、全体のNN学習は3回。CVの105学習は変更しない | [全体fit条件](visualization_and_interpretation.md#global-fit) |
| 2026-09-10 | 位置対応FT-IRによる解釈の計画を文書化 | CV開始後。測定・解析の詳細はOpen、結果は未確認 | [FT-IR計画](visualization_and_interpretation.md#ftir-interpretation) |
| 2026-09-10 | 全体fit後のHungarian matching基準をB0からA0のCosine-KMeansへ変更 | B0・B1 OOF可視化後・全体fit前。mask・Aug前の再構成学習を比較の起点とする | [matching基準](visualization_and_interpretation.md#matching-reference) |
| 2026-09-10 | B0・B1 OOF sanityの保存をlabel・silhouetteのPNGと数値CSVへ整理 | 探索的な表示仕様。fold内B0基準のmatchingを使用 | [OOF sanity](oof_sanity_visualization.md) |

全体fitのA0基準は表示番号の整列に用いる。A0の劣化検出性能や解釈可能性が確認されたことを意味しない。
また、CV開始後の追加判断を、CV開始前から固定した条件として遡及記載しない。
