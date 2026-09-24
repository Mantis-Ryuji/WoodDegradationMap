# 研究・実装 ToDo

現状と残作業を管理する。研究条件は[研究設計](docs/design/README.md)、
操作と成果物の確認方法は[runbook](docs/experiment_runbook.md)を参照する。
状態は2026-09-25のユーザー完了報告に基づく。A1追加、主CV・OOF再集計・図表、
全体fit・K8マップ・PCAの再生成まで完了した。今回の文書更新では実行・成果物の再検証は行っていない。

## 残るToDo

- [ ] **mask ratio sweep**：M11-25・M11-75の各15 runs、M11を含むOOF集計・図表（第4節）。
- [ ] **FT-IR**：位置対応測定の詳細設計・実施・化学的対応の検討（第5節）。

残作業はこの2件のみ。追加可視化・正式な目視評価・任意の形状診断は現在の実施対象に含めない。
設計書の候補・Open事項を完了扱いにしたものではない。論文執筆はThesisで管理する。

## 現在の状態

| 項目 | 状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root・環境 | `outputs/experiments/production_v1/`、ChemoMAE v0.2.2 |
| 入力・manifest | 前処理・入力照合済み。split・共通train座標・augmentation contractは確定済み |
| 主ニューラルCV | A0・A1・M00・M10・M01・M11の5 folds×3反復、計90学習が完了 |
| baseline | B1 PCAは5 foldsでfit済み。B0はfit不要 |
| 主8条件のclustering・評価 | B0・B1・A0・A1・M00・M10・M01・M11の全5 folds×3反復、計120組合せが完了 |
| OOF集計 | A1を含む`main_oof_v1`の再集計・check完了。対象は49試料・120 source runs・82,320 score records |
| 主条件の図表 | 同じ`results/figures/main_oof_v1/`へPNG 3枚・CSV 11個を再生成済み。主指標図は2×2、分布・paired図は2×3。A1−A0・M11−A1を追加 |
| OOF sanity | B0・B1・A0・M00のPNG 5枚・CSV 3つを`outputs/sanity_checks/a0_m00_oof_visualization/`に保存済み |
| 全体fit準備 | 共通manifest・B0/PCAの準備・check完了。PCAは49試料・401,408画素でfitし、保存復元probeの最大絶対誤差0。既存A0・M00・M11のGPU smoke・再開probeは過去の記録で合格、A1追加工程は今回のユーザー報告で完了 |
| 全体NN学習 | A0・A1・M00・M11の各800 epochが完了 |
| 全体クラスタリング・代表図表 | B0・B1・A0・A1・M00・M11の6条件・K8・全49試料。M00基準のSNV cosine＋Hungarianで表示番号を整列。同じ`global_k8_v1`へPNG 67枚・CSV 44個、代表図6×7を再生成済み |
| PCA | 共通401,408画素の入力・6条件の座標・2×6 PNG 1枚とCSV 5個を再生成済み。B1は既存PC1/PC2を直接使用。保存先は`global_k8_v1/pca-latent-2d/` |
| mask率補助実験 | 実施する。25%・75%を各15 runs追加し、50%は主条件M11を再利用。学習・評価・OOF集計は既存CLIを使用し、mask率図表は追加実装が必要 |
| FT-IR | 位置対応測定の詳細設計・実施・解釈が残る |

[執筆への引き継ぎ](docs/manuscript_handoff.md)から既存成果物を参照できる。
A1追加の実行記録・手順は[追補runbook](docs/a1_extension_runbook.md)を参照する。
再生成が必要な場合だけ、[OOF図表](docs/experiment_runbook.md#oof-reporting)、
[K8マップ・スペクトル](docs/experiment_runbook.md#global-post-fit)、[PCA](docs/experiment_runbook.md#global-pca)の各手順を使う。

mask率sweepは第4節の固定条件で実施する。追加可視化や物理化学的解釈の完了を着手条件にしない。
主条件の学習・評価条件と比較範囲は維持する。

## 1. 主条件のOOF集計

- [x] A1を含む主8条件の5 folds×3反復を揃え、`main_oof_v1`を上書き再集計・checkする。
- [x] 反復間ARI・未定義指標と理由を保持し、欠損・失敗・中断した入力を黙って除外しない処理を確認する。

## 2. 主条件CV完了後の図表生成

この節と第3節の実装はAstraで進める（ユーザー指定）。作業時のモデル選択であり、研究条件には含めない。
図表は[評価指標の報告規約](docs/design/evaluation_metrics.md#reporting)に従う。
主8条件の`main_oof_v1`から図表生成・可視化まで完了した。
mask率の補助図表は、専用OOF集計の完了後に追加する。

- [x] OOF snapshotから、代表$K_0=8$と全7Kの指標・paired比較・交互作用のCSV表を生成するpipelineを実装する。
- [x] 主指標サマリを2×2、$K_0=8$の試料別分布・paired K依存図を2×3で生成する。paired図は主要3比較にA1−A0・M11−A1を追加する。
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
比較の全体像はB0・B1・A0・A1・M00・M11の6条件で保持し、詳細な観察はM11・K8を中心に検討する。
M11の選択は既存結果を見た後の探索方針であり、全指標での優位性や化学的妥当性の確定とは区別する。

### 完了した成果物

- [x] ROOT_SEED=20260905、fold位置を`global`、反復ID 1とする共通manifest・seed・保存規約を実装し、本番manifestを作成・checkする。49試料×8,192＝401,408画素、保存rootは`global_v1`。
- [x] B0/PCA準備・保存復元確認とGPU smokeを経て、A0・A1・M00・M11を各800 epoch学習する。A1追加分の完了は2026-09-25のユーザー報告による。
- [x] `global_cluster.py`で6条件のCosine-KMeansをK8でfitし、全49試料・3,902,250有効画素のラベルと中心・出典・完了記録を保存する。
- [x] `visualize_global.py`でM00基準のSNV cosine＋Hungarian、条件別マップ、SNV類似度・対応表、occupancy、代表・試料別・差スペクトルを出力する。元のクラスタ所属は変えない。
- [x] 各条件の`labels/`に00〜06の1×7、07の固定代表7試料、08の全49試料7×7を保存する。rootの`01_representative_samples_6x7.png`を含めPNG 67枚・CSV 44個を出力する。
- [x] 共通401,408画素のPCA入力・6条件の座標・2×6 PNG一枚・CSVを保存する。B1は既存PCを利用し、他5条件は中心化したPC1/PC2を使う。

集計・matching・表示の定義は設計書、テスト履歴・再生成・checkコマンドはrunbookに集約する。
保存記録の確認と、現在のコードでテスト・全成果物checkを再実行したことは区別する。

## 4. Mask率補助実験

**実施する補助実験。M11のmask率25%・50%・75%の感度解析とする。**
[1 runの手順](docs/experiment_runbook.md#neural-run)に従い、各runを800 epochで学習し、clustering・評価・checkまで完了する。

- M11-25の5 folds×3反復を完了する（15 runs）。
- M11-75の5 folds×3反復を完了する（15 runs）。
- 50%は主実験M11を再利用し、3条件の`mask_rate_oof_v1`を作成・checkする。
- 第2節の図表生成pipelineへmask率依存性の図表を追加し、`mask_rate_oof_v1`から生成・照合する。

主条件とmask率のOOFは別snapshotとする。800 epoch、split・seed・共通画素・augmentation強度・全7Kと評価指標は主実験と同一。
最良mask率の選択や主条件M11の置換は行わない。[実施手順](docs/experiment_runbook.md#mask-rate-sweep)を参照する。

## 5. FT-IR

[位置対応FT-IR計画](docs/design/visualization_and_interpretation.md#ftir-interpretation)に従い、
対象試料・領域、NIRとの位置対応、測定・反復条件、前処理・比較指標を確定して測定する。
クラスタ間の化学的対応を検討し、測定範囲と解釈の限界を記録する。詳細設計・測定・解釈は未完了。

## 6. 外部執筆への資料提供

論文の章立て・本文・図の体裁・引用・執筆進捗は `C:\Users\PC_User\Python\Thesis` で管理する。
実験・図表生成は本書、資料の対応と引き渡し時の確認事項は[執筆への引き継ぎ](docs/manuscript_handoff.md)を参照する。
