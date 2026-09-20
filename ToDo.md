# 研究・実装 ToDo

現状と残作業を管理する。研究条件は[研究設計](docs/design/README.md)、
操作と成果物の確認方法は[runbook](docs/experiment_runbook.md)を参照する。
状態は2026-09-21時点の保存記録とユーザーの完了報告に基づく。
今回の整理では完了JSONと実装を読み合わせた。学習・描画・テスト・成果物の全hash検証は再実行していない。

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
| 全体fit準備 | 共通manifest・B0/PCAの準備・check完了。PCAは49試料・401,408画素でfitし、保存復元probeの最大絶対誤差0。3条件のGPU smoke・再開probeはすべて合格 |
| 全体NN学習 | A0・M00・M11の各800 epochが完了。各313,600 attempted updates。`completion.json`・attempt記録を確認し、`training-check`完了はユーザー報告による |
| 全体クラスタリング・代表図表 | 5条件・K8・全49試料。M00基準のSNV cosine＋Hungarianで表示番号を整列。完了記録はPNG 56枚・CSV 37個、`checks_passed=true` |
| PCA | 共通401,408画素の入力・5条件の座標・2行5列PNG 1枚とCSV 5個の完了記録を確認。B1は既存PC1/PC2を直接使用。保存先は`global_k8_v1/pca-latent-2d/` |
| 現在の作業 | Thesisで既存結果を書き始め、主張と根拠を対応づけて必要な追加可視化を絞る。観察粒度はK8 |
| mask率補助実験 | 実施する。25%・75%を各15 runs追加し、50%は主条件M11を再利用。学習・評価・OOF集計は既存CLIを使用し、mask率図表は追加実装が必要 |
| 保留・未実装 | 試料ごとの詳細図、PCAの追加色分け、連続指標map。採否・具体的な構成は執筆後に検討 |

**次は[執筆への引き継ぎ](docs/manuscript_handoff.md)を入口に、Thesisで本文・図表の対応を整理する。**
追加解析を先に増やさず、CVで比較できる性質と、M11・K8の試料内で解釈したい領域差を分けて書く。
既存の試料別スペクトルCSVは利用可能。詳細PNGの対象試料・構成・帯域は未確定である。
再生成が必要な場合だけ、[OOF図表](docs/experiment_runbook.md#oof-reporting)、
[K8マップ・スペクトル](docs/experiment_runbook.md#global-post-fit)、[PCA](docs/experiment_runbook.md#global-pca)の各手順を使う。

mask率sweepは第4節の固定条件で実施する。追加可視化や物理化学的解釈の完了を着手条件にしない。
主条件の学習・評価条件と比較範囲は維持する。

## 1. 主条件のOOF集計

- [x] 主7条件の5 folds×3反復の完全性を確認し、`main_oof_v1`を作成・checkする。
- [x] 反復間ARI・未定義指標と理由を保持し、欠損・失敗・中断した入力を黙って除外しない処理を確認する。

## 2. 主条件CV完了後の図表生成

この節と第3節の実装はAstraで進める（ユーザー指定）。作業時のモデル選択であり、研究条件には含めない。
図表は[評価指標の報告規約](docs/design/evaluation_metrics.md#reporting)に従う。
主7条件の`main_oof_v1`から図表生成・可視化の実装と出力確認を先に固める。
mask率の補助図表は、専用OOF集計の完了後に追加する。

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
原稿への採用・captionの最終調整は第6節の執筆側で管理する。

## 3. 全体fitと解釈

[全体可視化設計](docs/design/visualization_and_interpretation.md)に従う。
比較の全体像はB0・B1・A0・M00・M11の5条件で保持し、詳細な観察はM11・K8を中心に検討する。
M11の選択は既存結果を見た後の探索方針であり、全指標での優位性や化学的妥当性の確定とは区別する。

### 完了した成果物

- [x] ROOT_SEED=20260905、fold位置を`global`、反復ID 1とする共通manifest・seed・保存規約を実装し、本番manifestを作成・checkする。49試料×8,192＝401,408画素、保存rootは`global_v1`。
- [x] B0/PCA準備・保存復元確認と3条件のGPU smokeを完了し、A0・M00・M11を各800 epoch学習する。学習完了は保存記録、`training-check`完了はユーザー報告による。
- [x] `global_cluster.py`で5条件のCosine-KMeansをK8でfitし、全49試料・3,902,250有効画素のラベルと中心・出典・完了記録を保存する。
- [x] `visualize_global.py`でM00基準のSNV cosine＋Hungarian、条件別マップ、SNV類似度・対応表、occupancy、代表・試料別・差スペクトルを出力する。元のクラスタ所属は変えない。
- [x] 各条件の`labels/`に00〜06の1×7、07の固定代表7試料、08の全49試料7×7を保存する。rootの代表7試料×5条件図を含め、完了記録はPNG 56枚・CSV 37個。
- [x] 共通401,408画素のPCA入力・5条件の座標・PNG一枚・CSVを保存する。B1は既存PCを利用し、他4条件は中心化したPC1/PC2を使う。`pca-latent-2d/`の完了記録を確認した。

集計・matching・表示の定義は設計書、テスト履歴・再生成・checkコマンドはrunbookに集約する。
保存記録の確認と、現在のコードでテスト・全成果物checkを再実行したことは区別する。

### 執筆と追加解析の判断

- [ ] 既存のCV図表を本文に配置し、LLA・LFRの傾向とARI・silhouette・occupancyのtrade-offを整理する。
- [ ] M11・K8で注目する試料と領域を絞り、既存`sample_spectra.csv`・寄与画素数を用いて、試料内クラスタの観測スペクトルを確認する。全試料macro平均だけで個別領域を解釈しない。
- [ ] 本文の根拠に不足する図を選び、対象試料・panel・比較対象を決めてから詳細PNGを追加する。現在のPCA一枚の採否もここで判断する。
- [ ] PCAのmetadata・帯域指標による色分け、連続スペクトル指標mapは、必要性を確認した場合だけ具体化する。採用する場合は[帯域・平滑化・積分のOpen事項](docs/design/visualization_and_interpretation.md#spectral-band-selection)を確定する。
- [ ] 空間的一貫性と物理化学的解釈を分け、外観・スペクトルの対応、例外、未確認の帰属を記述する。第5節のFT-IR・正式な目視評価は別の未確定事項として扱う。

## 4. Mask率補助実験

**実施する補助実験。M11のmask率25%・50%・75%の感度解析とする。**
[1 runの手順](docs/experiment_runbook.md#neural-run)に従い、各runを800 epochで学習し、clustering・評価・checkまで完了する。

- [ ] M11-25の5 folds×3反復を完了する（15 runs）。
- [ ] M11-75の5 folds×3反復を完了する（15 runs）。
- [ ] 50%は主実験M11を再利用し、3条件の`mask_rate_oof_v1`を作成・checkする。
- [ ] 第2節の図表生成pipelineへmask率依存性の図表を追加し、`mask_rate_oof_v1`から生成・照合する。

主条件とmask率のOOFは別snapshotとする。800 epoch、split・seed・共通画素・augmentation強度・全7Kと評価指標は主実験と同一。
最良mask率の選択や主条件M11の置換は行わない。[実施手順](docs/experiment_runbook.md#mask-rate-sweep)を参照する。

## 5. 未確定事項

[研究設計のOpen事項](docs/design/README.md#open-items)を確認する。
位置対応FT-IRと正式な目視評価は詳細設計が必要。任意の形状診断は、採用する場合だけ定義を固定する。

## 6. 外部執筆への資料提供

論文の章立て・本文・図の体裁・引用・執筆進捗は `C:\Users\PC_User\Python\Thesis` で管理する。
実験・図表生成は本書、資料の対応と引き渡し時の確認事項は[執筆への引き継ぎ](docs/manuscript_handoff.md)を参照する。
