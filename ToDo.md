# 研究・実装 ToDo

現状と残作業を管理する。研究条件は[研究設計](docs/design/README.md)、
操作と成果物の確認方法は[runbook](docs/experiment_runbook.md)を参照する。
状態は2026-09-19までに確認・共有された実行記録に基づく。

## 現在の状態

| 項目 | 状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root・環境 | `outputs/experiments/production_v1/`、ChemoMAE v0.2.2 |
| 入力・manifest | 前処理・入力照合済み。split・共通train座標・augmentation contractは確定済み |
| 主ニューラルCV | A0・M00・M10・M01・M11の5 folds×3反復、計75学習が完了 |
| baseline | B1 PCAは5 foldsでfit済み。B0はfit不要 |
| 主7条件のclustering・評価 | 全5 folds×3反復、計105組合せのclustering・評価・checkが完了 |
| OOF集計 | pipelineの実装・preflightは完了。主条件snapshotは未作成 |
| OOF sanity | B0・B1・A0・M00のPNG 5枚・CSV 3つを`outputs/sanity_checks/a0_m00_oof_visualization/`に保存済み |
| 未実装 | vMF補助実験、最終報告用図表、全体fitのpipeline |

**次の作業は主7条件の`main_oof_v1`の作成・check。**
[OOF集計手順](docs/experiment_runbook.md#oof-aggregation)を使用する。

2026-09-19のユーザー指定により、作業順は**主条件のOOF集計 → 主条件の図表生成 → 全体fit・可視化・解釈 → 補助実験**とする。
図表・可視化の実装と出力確認を先に固める。補助実験の完了を主条件図表・全体fitの着手条件にはしない。
研究条件・比較範囲は維持し、mask率・vMFのCV補助実験とその図表は第4〜5節で追加する。

## 1. 主条件のOOF集計

- [ ] 主7条件の5 folds×3反復の完全性を確認し、`main_oof_v1`を作成・checkする。
- [ ] 反復間ARI、欠損・失敗・中断、未定義指標と理由が集計に保持されていることを確認する。

## 2. 主条件CV完了後の図表生成

この節と第3節の実装はAstraで進める（ユーザー指定）。作業時のモデル選択であり、研究条件には含めない。
図表は[評価指標の報告規約](docs/design/evaluation_metrics.md#reporting)に従う。
主7条件の`main_oof_v1`から図表生成・可視化の実装と出力確認を先に固める。
mask率・vMFの補助実験に依存する図表は、それぞれのOOF集計完了後に追加する。

- [ ] OOF snapshotから、代表$K_0=8$の主表と診断表を生成するpipelineを実装する。
- [ ] 全7Kの指標・計画contrast・反復別曲線、paired差、2×2交互作用の図表を実装する。
- [ ] 未定義理由・共通対象数、試料間SD・反復間SD、ARI・occupancy等の必須診断を照合する。
- [ ] 元snapshotと図表のsource hash、条件・K・反復・集計対象、captionに必要な定義を保存する。
- [ ] `main_oof_v1`から主条件の必須図表を生成し、数値・表示・原稿の掲載候補を照合する。

## 3. 全体fitと解釈

[全体可視化設計](docs/design/visualization_and_interpretation.md)に従い、B0・B1・A0・M00・M11の5条件×2手法を比較する。
M00＋Cosine-KMeans基準のmatching、代表スペクトルの平均集計、SG二次微分はFixedであり、以下は実装・実施の残作業である。
CV用CLIをそのまま全体fitへ使わない。
全体fit用のvMF 5 fitsに必要な数値仕様・共通処理の検証はこの段階で先行し、CV補助実験の735 fitsは第5節で行う。

- [ ] 全49試料の共通画素抽出・manifest・seed適用・実行記録・保存先・CLIを実装する。
- [ ] PCAの全体fitと、A0・M00・M11の各1回（計3回）の全体学習を実装・実施する。
- [ ] 同じ表現・抽出座標・$K_0=8$でCosine-KMeansを5 fits行う。
- [ ] v0.2.2の数値関数・公開helper・初期化・最終尤度・保存復元・退化成分を検証する。16次元・256次元の参照値比較、CPU小規模、chunk、GPU最小確認を含む。
- [ ] [実験プロトコル第5.2.3節](docs/design/experiment_protocol.md#vmf-supplementary)に従い、数値精度・EM停止条件・集中度設定を検証・ユーザー確認のうえ固定する。全体fitと後続のCV補助実験で共用する。
- [ ] 確定・検証した数値仕様で全体fit用のvMF処理を実装し、同じ表現で5 fitsを行う。CV補助実験の735 fitsとは分ける。
- [ ] 全49試料のhard label map、SNV類似度行列・matching対応表、確認用contingency・overlap、occupancy・使用クラスタ数、潜在空間図を保存する。
- [ ] [代表スペクトルの仕様](docs/design/visualization_and_interpretation.md#representative-spectra)に従い、反射率・SNV・疑似吸光度の平均集計、SG二次微分、四分位範囲、差スペクトルを実装する。寄与試料・画素数と追加除外数も保存する。
- [ ] 固定7代表試料について、行をCosine-KMeans・vMF、列を5条件とする比較図を作り、試料IDと共通描画規約を確認する。
- [ ] マップと観測スペクトルから領域差を探索的に解釈し、CV指標の改善と化学的対応を区別する。
- [ ] 第2節の主条件図表と本節の全体fit・可視化・探索的解釈を確認し、実装上の残件と解釈の限界を記録してから補助実験へ移る。第6節のFT-IR・正式な目視評価は未確定事項として別途扱う。

## 4. Mask率補助実験

第2〜3節の図表生成・全体fit・解釈を終えてから着手する。
[1 runの手順](docs/experiment_runbook.md#neural-run)に従い、各runを800 epochで学習し、clustering・評価・checkまで完了する。

- [ ] M11-25の5 folds×3反復を完了する（15 runs）。
- [ ] M11-75の5 folds×3反復を完了する（15 runs）。
- [ ] 50%は主実験M11を再利用し、3条件の`mask_rate_oof_v1`を作成・checkする。
- [ ] 第2節の図表生成pipelineへmask率依存性の図表を追加し、`mask_rate_oof_v1`から生成・照合する。

主条件とmask率のOOFは別snapshotとする。

## 5. vMF補助実験

第2〜3節の図表生成・全体fit・解釈を終えてから着手する。
主7条件×5 folds×3反復×7Kの735 fits。範囲・利用版・退化成分の扱いはFixed、数値仕様と専用pipelineは未完了。
[実験プロトコル](docs/design/experiment_protocol.md#vmf-supplementary)と[評価規約](docs/design/evaluation_metrics.md#vmf-evaluation)に従う。
第3節で確定・検証する数値仕様と共通処理を再利用する。

- [ ] 第3節の数値仕様・検証記録を確認し、vMFのtest結果を見る前に固定した設定をCV補助実験でも使用する。
- [ ] CV専用のfit・評価・check・OOFと独立した出力先を設計・実装する。
- [ ] 既存の主7条件の重み・PCAと共通train画素で735 fitsを実施する。NN学習・PCA fitは追加しない。
- [ ] 同じtest全画素・共通摂動で評価し、完了・失敗・未定義値を保持してOOF集計する。
- [ ] 第2節の図表生成pipelineへvMF比較を追加し、計画contrast・2×2交互作用・K依存性をCosine-KMeansと併記する。

## 6. 未確定事項

[研究設計のOpen事項](docs/design/README.md#open-items)を確認する。
位置対応FT-IRと正式な目視評価は詳細設計が必要。任意の形状診断・vMF責務マップは、採用する場合だけ定義を固定する。

## 7. 外部執筆への資料提供

論文の章立て・本文・図の体裁・引用・執筆進捗は `C:\Users\PC_User\Python\Thesis` で管理する。
実験・図表生成は本書、資料の対応と引き渡し時の確認事項は[執筆への引き継ぎ](docs/manuscript_handoff.md)を参照する。
