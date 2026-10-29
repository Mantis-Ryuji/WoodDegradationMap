# 成果物案内

解析ごとの出力先と、その結果が表す範囲を示す。パスはリポジトリルートからの相対パスである。
`outputs/`はGit管理対象外であり、以下のファイルはローカルで生成する成果物である。
実行方法は[実験手順](experiment_runbook.md)、指標の定義は[評価仕様](design/evaluation_metrics.md)を参照する。

## 前処理と入力の確認

| 出力先 | 内容 |
| --- | --- |
| `outputs/preprocessing/production_v1/` | 波長cutoff、反射率・SNVの帯域分布、品質確認図 |
| `outputs/experiments/production_v1/manifests/` | CVの試料分割、共通抽出画素、条件とseed |
| `outputs/sanity_checks/augmentation_strengths_train_fold1/` | train画素によるTGN・FSの変形例と強度の要約 |

前処理の候補画素図は品質確認用であり、劣化・異常の正解ラベルではない。
摂動の例図は学習用画素の観測に基づき、CV指標を最適化した結果とは区別する。

## 主実験のOOF評価

集計先は`outputs/experiments/production_v1/results/oof/main_oof_v1/`、
図表の出力先は`outputs/experiments/production_v1/results/figures/main_oof_v1/`である。

| ファイル・内容 | 読み方 |
| --- | --- |
| `01_main_metrics_k_sweep.png` | 主8条件のLLA-3・LLA-5・LLA-9・LFR(TGN+FS)とK依存性 |
| `02_k8_distributions.png` | K=8における試料ごとの分布とmacro平均 |
| `03_paired_k_sweep.png` | 同じ試料・K・反復における条件間の対応差 |
| CSV表 | 条件別指標、対応差、摂動の交互作用、occupancy、対象数と未定義情報 |
| `report.json` | 指標・表示の定義と集計元の出典 |

対象は49試料、5-fold、3学習反復、K=2/4/6/8/10/12/14である。
LLAは偶然一致を補正した値を指し、保存キーは`adjusted_lla`、補正前の値は`lla`である。
未定義値を補完せず、条件ごとの集計と条件差で用いる試料数を併記する。

`outputs/sanity_checks/a0_m00_oof_visualization/`はB0・B1・A0・M00の診断図である。
全8条件の評価には主実験の集計を用いる。

## 全体fit・K8マップとスペクトル

出力先は`outputs/experiments/global_v1/results/figures/global_k8_v1/`である。

| ファイル・内容 | 用途 |
| --- | --- |
| `01_representative_samples_6x7.png` | 主要6条件と表示用7試料の比較 |
| `M11/labels/` | M11の全49試料および7試料のクラスタマップ |
| `M11/01_representative_spectra.png` | SNV・反射率・疑似吸光度の二次微分。試料等重み平均と試料間IQR |
| `M11/sample_spectra.csv` | 試料・表示クラスタ別の観測スペクトル平均 |
| `M11/spectrum_counts.csv` | スペクトル要約に寄与した画素数・除外数 |

マップは全49試料をfitに含めた記述的解析であり、OOF予測ではない。
主要6条件の表現は各1回学習・fitし、KMeans初期化を3回変える。
表示には初回を用い、条件間の色はM00を基準としたSNV代表線の対応によって整列する。
同じ色は化学状態の同一性を保証しない。

表示用7試料の列順はKYOw02752、KYOw02772、KYOw02777、KYOw02787、KYOw02790、KYOw16744、KYOw16750。
各樹種から選択した表示例であり、全試料や各化学状態の典型性を保証しない。
PCA座標と描画の手順は[全体fitのPCA](experiment_runbook.md#global-pca)を参照する。

## NIRスペクトルの解釈

`scripts/experiments/visualize_nir_interpretation.py`は保存済みのM11試料内平均CSVを読み、
`outputs/nir_interpretation/M11_equal_sample.png`と`each_nir_spectra/<sample_id>.png`を出力する。
試料別の比較クラスタと参照曲線は[選択設定](../scripts/experiments/nir_plot_selection.yaml)で指定する。
別試料の参照線は試料ID・クラスタIDを明示し、対象試料の集計へ混ぜない。

横軸は波数に変換するが、縦軸の二次微分は波長に関する値である。
疑似吸光度の画素除外、試料等重みの集計、帰属18帯域・表示9帯域の定義は
[NIR解析仕様](design/visualization_and_interpretation.md#nir-interpretation)に示す。
本リポジトリのNIR描画CLIはスペクトル曲線の出力を扱う。論文の帯域値集計・baseline棒グラフの生成コードは収録していない。

<a id="mask-rate-handoff"></a>

## Mask率の感度解析

| 解析 | 出力先・主要ファイル | 評価範囲 |
| --- | --- | --- |
| CV・25%/50%/75% | `outputs/experiments/production_v1/results/figures/mask_rate_oof_v1/`：`01_mask_rate_metrics_k_sweep.png`、`metrics_all_k.csv`、`metrics_k8.csv`、`report.json`、`completion.json` | 同じsplit・抽出画素による5-fold×3学習反復、全49試料・全7K |
| 全体fit・50%/75% | `outputs/experiments/global_v1/results/figures/global_mask_rate_k8_v1/`：`01_representative_samples_2x7.png`、`summary.csv/json`、`sample_metrics.csv`、`matching.csv`、`captions.csv`、`run.json`、`completion.json` | 各表現1回の学習後、K8のKMeans初期化だけを3回変える記述的比較 |

50%は主条件M11を共用し、図では`M11(-50)`と表示する。
全体fitの2×7図は初回の割当を表示し、上段50%・下段75%とする。
試料間SDと反復間SDを区別し、全体fitの反復間SDをNN再学習のばらつきとは解釈しない。

## 空間マップ再現性

出力先は`outputs/experiments/global_spatial_reproducibility_v2/`である。
M11・K8の3 encoder×3 KMeans初期化について、同一画素集合の9マップ・全36対を比較する。
`results/evaluation/`の対別・試料別・全体集計と`results/report/`のマップ・境界図を併読する。
入力として参照する学習済み重み・マップと、再利用する成果物の関係は[専用手順](spatial_map_reproducibility_runbook.md)に示す。
現行CLIは保存済み成果物からの再評価用であり、encoder 2・3の学習actionを持たない。
Gitに同梱しない学習済み重み・再利用マップが実行の前提となる。

## Raman Bacteria-ID

| 解析 | 出力先 | 主要成果物 |
| --- | --- | --- |
| Bacteria-4/6クラスタリング | `outputs/experiments/bacteria_id_table4_80_20_v1/` | `table4.md`、`table4_seed_scores.csv`、`table4_statistics.json`、`table4_seed_scores.png` |
| 30分離株の分類・HPO | `outputs/experiments/bacteria_id_table5_hpo_cv_v1/` | `table5.md`とHPO・採用設定・seed別評価記録 |
| 分類用の事前学習 | `outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/` | M00/M11のreference30事前学習記録 |

`table4`・`table5`は参照したSMAE論文の比較表に由来する成果物名である。
引用した文献値は本リポジトリで再実験した値とは区別する。
分割・前処理・評価条件の差と、設定選択の範囲は[Bacteria-ID仕様](design/bacteria_id_experiment_plan.md)と[HPO仕様](design/bacteria_id_hpo_protocol.md)を参照する。
