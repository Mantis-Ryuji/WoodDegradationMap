# 研究・実装 ToDo

現状と残作業を管理する。研究条件は[研究設計](docs/design/README.md)、
操作と成果物の確認方法は[runbook](docs/experiment_runbook.md)を参照する。
状態は2026-09-19までに確認・共有された実行記録に基づく。
2026-09-20に全体fit後の可視化案を追記し、mask率sweep・vMFの優先順位を下げた。実装・実行状況の更新ではない。

## 現在の状態

| 項目 | 状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root・環境 | `outputs/experiments/production_v1/`、ChemoMAE v0.2.2 |
| 入力・manifest | 前処理・入力照合済み。split・共通train座標・augmentation contractは確定済み |
| 主ニューラルCV | A0・M00・M10・M01・M11の5 folds×3反復、計75学習が完了 |
| baseline | B1 PCAは5 foldsでfit済み。B0はfit不要 |
| 主7条件のclustering・評価 | 全5 folds×3反復、計105組合せのclustering・評価・checkが完了 |
| OOF集計 | `main_oof_v1`の作成・check完了（checkはユーザー完了報告）。完了記録は49試料・105 source runs・72,030 score records |
| 主条件の図表 | `results/figures/main_oof_v1/`にPNG 3枚・CSV 11個と出典・完了記録を生成済み。主指標・分布・pairedの全図が2行3列 |
| OOF sanity | B0・B1・A0・M00のPNG 5枚・CSV 3つを`outputs/sanity_checks/a0_m00_oof_visualization/`に保存済み |
| 全体fit準備 | 共通manifestの本番作成・check完了。PCA保存復元時の配列配置変更による不一致を修正し、PCA再fit待ち。合成データCPU検証済み。GPU smoke・学習は未実行 |
| 未実装 | 全体クラスタリング・マップ・スペクトル集計、vMF・mask率補助実験の図表生成対応、vMF補助実験pipeline |

**次の作業はPCAの再fit・check、GPU smoke確認、A0・M00・M11の一括学習。**
[全体fit手順](docs/experiment_runbook.md#global-fit-pipeline)の専用CLIを使用する。保存先は`outputs/experiments/global_v1/`。
主条件図表は[再生成手順](docs/experiment_runbook.md#oof-reporting)から更新できる。

2026-09-20のユーザー指定により、作業順は**主条件のOOF図表 → 全体fit → Cosine-KMeansの5条件による潜在空間・空間map・観測スペクトルの解析 → 図表・化学的解釈の整理 → 低優先度の補助実験**とする。
mask率sweepとvMFは計画に残し、先行する解析・図表・解釈を一通り終えるまで着手しない。
vMFの数値検証・共通処理実装・全体fit用5 fitsも第5節へ移し、先行する可視化・解釈の前提にしない。
学習条件・比較範囲は維持する。第4〜5節へ着手する時点で必要性と工数を再確認する。
主条件の代表指標・表示構成については第2節に今回のユーザー指定を反映する。

## 1. 主条件のOOF集計

- [x] 主7条件の5 folds×3反復の完全性を確認し、`main_oof_v1`を作成・checkする。
- [x] 反復間ARI・未定義指標と理由を保持し、欠損・失敗・中断した入力を黙って除外しない処理を確認する。

## 2. 主条件CV完了後の図表生成

この節と第3節の実装はAstraで進める（ユーザー指定）。作業時のモデル選択であり、研究条件には含めない。
図表は[評価指標の報告規約](docs/design/evaluation_metrics.md#reporting)に従う。
主7条件の`main_oof_v1`から図表生成・可視化の実装と出力確認を先に固める。
mask率・vMFの補助実験に依存する図表は、それぞれのOOF集計完了後に追加する。

- [x] OOF snapshotから、代表$K_0=8$と全7Kの指標・paired比較・交互作用のCSV表を生成するpipelineを実装する。
- [x] 主指標サマリ・$K_0=8$の試料別分布・主要3比較のpaired K依存図を、各2行3列のPNGとして生成する。
- [x] 未定義理由・共通対象数、試料間SD・反復間SD、ARIの退化情報、clean testのoccupancyを保存し、集計契約を合成データで検証する。
- [x] 元snapshotと図表のsource hash、条件・K・反復・集計対象、captionに必要な定義を保存する。
- [x] `main_oof_v1`から本番のPNG 3枚・CSV 11個を生成し、図の表示を確認する。

2026-09-19のユーザー指定により、表示はLLA（補正後）、LFR(TGN+FS)、ARI、Cosine-Silhouette、
Cluster Occupancyの優先順とする。LLAの窓3・5・9は個別に残す。Occupancyと交互作用はCSVのみ。
LFRは図ではTGN+FSのみ、CSVではLFR(TGN+FS)・LFR(TGN)・LFR(FS)の3種類を保存する。
PNGは`01_main_metrics_k_sweep.png`、`02_k8_distributions.png`、`03_paired_k_sweep.png`の連番とし、旧PNGは削除済み。
補正後LLAの交互作用と試料単位のpaired ARIは報告pipelineで算出し、元OOF snapshotを保持する。
原稿への採用・captionの最終調整は第7節の執筆側で管理する。

## 3. 全体fitと解釈

[全体可視化設計](docs/design/visualization_and_interpretation.md)に従い、B0・B1・A0・M00・M11の5条件を、まずCosine-KMeansで比較する。
M00＋Cosine-KMeans基準のmatching、代表スペクトルの平均集計、SG二次微分はFixedであり、以下は実装・実施の残作業である。
CV用CLIをそのまま全体fitへ使わない。
本節では潜在空間・空間map・観測スペクトルの対応と化学的解釈を優先する。vMFとの手法間比較は第5節で後から追加する。

- [x] ROOT_SEED=20260905・SHA-256方式でfoldを`global`へ置換し、反復ID 1を使用するseed対応と、独立root `global_v1`をユーザー確認する（2026-09-19）。
- [x] 全49試料×8,192画素（401,408画素）の共通manifest・実行記録・保存先・CLIを実装する。
- [x] B0/PCAの準備・保存復元と、A0・M00・M11の各1回を順次実行する学習・明示的再開・完了checkを実装し、合成データのCPU検証を行う。
- [x] 本番manifestを作成・checkする（49試料・401,408画素、ユーザー実行ログ確認）。
- [ ] B0/PCAの準備・全体fit・checkを完了する。初回の復元不一致は修正済み、途中出力を退避して再fit待ち。
- [ ] 実寸model・batch size 1024のGPU smokeで、全3条件の保存復元・全可視抽出・epoch境界再開を確認する。
- [ ] A0・M00・M11を各800 epochで1回、計3回学習し、`training-check`で確認する。
- [ ] 同じ表現・抽出座標・$K_0=8$でCosine-KMeansを5 fits行う。
- [ ] 全49試料のhard label map、SNV類似度行列・matching対応表、確認用contingency・overlap、occupancy・使用クラスタ数、潜在空間図を保存する。
- [ ] [代表スペクトルの仕様](docs/design/visualization_and_interpretation.md#representative-spectra)に従い、反射率・SNV・疑似吸光度の平均集計、SG二次微分、四分位範囲、差スペクトルを実装する。寄与試料・画素数と追加除外数も保存する。
- [x] [UMAP・連続スペクトル指標mapの案](docs/design/visualization_and_interpretation.md#latent-spectral-maps)を文書化する（2026-09-20）。PNG・CSV、cosine UMAP、クラスタ所属を使わない空間平滑化の方針を記録する。実装は全体fit待ち。
- [ ] 全体fit後のクラスタ平均二次微分曲線・試料間変動を確認し、候補帯域・選択理由を記録する。平滑化方式・数値設定、積分端点・符号・計算法、共通color scaleを確定する。
- [ ] UMAPの共通表示画素・数値設定・seedを確定し、5条件でクラスタ・metadata・同一帯域指標を色分けしたPNGと元数値CSVを生成する。
- [ ] クラスタに依存しない連続スペクトル指標mapと、UMAP・空間位置・観測スペクトルの対応を示す詳細PNGを生成し、ChemoMAEとbaselineで探索できる領域差を比較する。
- [ ] 固定7代表試料について、Cosine-KMeansの5条件の比較図を作り、試料IDと共通描画規約を確認する。
- [ ] マップと観測スペクトルから領域差を探索的に解釈し、CV指標の改善と化学的対応を区別する。
- [ ] 第2節の主条件図表と本節の全体fit・可視化・探索的解釈を一通り完了し、図表・知見・解釈の限界を整理する。低優先度の補助実験はその後に扱う。第6節のFT-IR・正式な目視評価は未確定事項として別途扱う。

## 4. Mask率補助実験

**低優先度。計画は維持し、第2〜3節の解析・図表・解釈の整理を一通り終えた後に回す。**
[1 runの手順](docs/experiment_runbook.md#neural-run)に従い、各runを800 epochで学習し、clustering・評価・checkまで完了する。

- [ ] M11-25の5 folds×3反復を完了する（15 runs）。
- [ ] M11-75の5 folds×3反復を完了する（15 runs）。
- [ ] 50%は主実験M11を再利用し、3条件の`mask_rate_oof_v1`を作成・checkする。
- [ ] 第2節の図表生成pipelineへmask率依存性の図表を追加し、`mask_rate_oof_v1`から生成・照合する。

主条件とmask率のOOFは別snapshotとする。

## 5. vMF補助実験

**低優先度。数値検証・実装・全体fit用5 fitsを含め、第2〜3節の解析・図表・解釈の整理を一通り終えた後に回す。**
全体fit用5 fitsと、主7条件×5 folds×3反復×7KのCV用735 fitsを計画に残す。範囲・利用版・退化成分の扱いはFixed、数値仕様と専用pipelineは未完了。
[実験プロトコル](docs/design/experiment_protocol.md#vmf-supplementary)と[評価規約](docs/design/evaluation_metrics.md#vmf-evaluation)に従う。
数値仕様と共通処理を本節で確定・検証し、全体fitとCV補助実験で共用する。

- [ ] v0.2.2の数値関数・公開helper・初期化・最終尤度・保存復元・退化成分を検証する。16次元・256次元の参照値比較、CPU小規模、chunk、GPU最小確認を含む。
- [ ] [実験プロトコル第5.2.3節](docs/design/experiment_protocol.md#vmf-supplementary)に従い、数値精度・EM停止条件・集中度設定を検証・ユーザー確認のうえ固定する。vMFのtest結果を見る前に固定し、全体fitとCV補助実験で共用する。
- [ ] 確定・検証した数値仕様で全体fit用のvMF処理を実装し、同じ表現で5 fitsを行う。CV補助実験の735 fitsとは分ける。
- [ ] 全49試料のvMF map・matching・スペクトル要約を保存し、固定7代表試料の2手法×5条件の比較図を追加する。先行するUMAP座標と帯域指標を共用する。
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
