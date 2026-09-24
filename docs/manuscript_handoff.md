# 論文執筆への引き継ぎ

論文の章立て・原稿・執筆進捗は `C:\Users\PC_User\Python\Thesis` で管理する。
本リポジトリから研究条件、根拠、実験成果物を参照する。

2026-09-25のユーザー完了報告により、A1を含む主8条件CV・OOF図表、全体6条件のfit・K8マップと
スペクトル、6×7代表図・2×6 PCA図まで再生成済みである。残るToDoはmask ratio sweepと位置対応FT-IRのみ。
追加可視化や正式な目視評価は現在の実施対象に含めず、既存成果物から執筆を進める。
詳細な観察はM11・K8の試料内クラスタを中心に検討する。完了記録の確認範囲は[ToDo](../ToDo.md)を参照する。

## 1. 参照資料

| 内容 | 定義・参照先 |
| --- | --- |
| 問い・採用理由・主張の範囲 | [研究概要](research_overview.md)、[ChemoMAEの位置づけ](chemomae_positioning.md)、[関連研究](related_work.md) |
| 前処理・データ契約 | [前処理仕様](design/preprocessing.md) |
| モデル・学習・CV・seed・クラスタリング | [実験プロトコル](design/experiment_protocol.md) |
| 指標・未定義値・集計・比較 | [評価指標](design/evaluation_metrics.md) |
| 全体fit・スペクトル要約・表示・FT-IR計画 | [全体可視化と解釈](design/visualization_and_interpretation.md) |
| SNV・摂動・loss・表現幾何の導出 | [数理的補足](mathematical_notes.md) |
| 混合精度・train loss・inertia | [数値実装の補足](numerical_implementation_notes.md) |
| 完了範囲・残作業・成果物の確認方法 | [ToDo](../ToDo.md)、[runbook](experiment_runbook.md) |

## 2. 図表の引き渡し

必須図表は[評価の報告規約](design/evaluation_metrics.md#reporting)、全体マップと観測スペクトルは
[可視化設計](design/visualization_and_interpretation.md)に従う。掲載章・枚数は外部原稿で決める。

- Captionとともに元ファイル、生成コード・config・manifestの版、試料・画素・fold・反復・K・条件・手法・fit範囲、選択根拠を示す。
- 実測・要約・人工摂動・模式図を区別し、単位・凡例・matching基準を明記する。主要な反例や傾向の逆転も報告する。
- [固定7代表試料](design/visualization_and_interpretation.md#representative-samples)と全49試料の補足表示を区別する。
- [代表スペクトル](design/visualization_and_interpretation.md#representative-spectra)は試料内平均→試料間の等重み平均とする。変換順序、除外範囲、四分位範囲、寄与数をcaptionと対応づける。
- 個別試料の領域差は試料内クラスタ平均で確認する。全試料macro平均のIQRを、その試料内のばらつきとして使わない。
- OOF sanityのfold内B0基準・画素一致数による整列と、全体fitのM00基準・試料等重みSNV代表線のcosine類似度による整列を区別する。
- `outputs/`の元成果物を保持し、原稿での体裁調整と解析結果を区別する。Fixedの仕様を実装・解析済みとは扱わない。

2026-09-21に代表7試料を、各樹種から多様な化学状態が見られそうなものを目視で選んだ試料へ変更し、
列順を試料番号の昇順にした。[選定記録](design/visualization_and_interpretation.md#representative-samples)と
[上書き生成手順](experiment_runbook.md#representative-sample-update)を参照し、使用するPNGに新指定が反映されていることを確認する。

### 図表候補と用途

| 資料 | 出典と注意点 |
| --- | --- |
| 波長cutoff | [cutoff_decision.png](../outputs/preprocessing/production_v1/cutoff_decision.png)。Reference由来SNR proxy、閾値10、保持・除外範囲。描画境界2308.72 nmと補間上限2305.59 nmを区別する |
| 補間後の帯域分布 | [反射率](../outputs/preprocessing/production_v1/interpolated_reflectance_band_distribution.png)・[SNV](../outputs/preprocessing/production_v1/interpolated_snv_band_distribution.png)。最終入力の記述的な確認 |
| 前処理後の候補画素 | [final_snv_anomaly_candidates.png](../outputs/preprocessing/production_v1/final_snv_anomaly_candidates.png)。確認用の候補であり、劣化・異常の正解ラベルではない |
| TGN・shiftの例 | [TGN](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_noise_exact_angles_examples.png)・[shift](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_shift_exact_endpoints_examples.png)。同じ実測train SNV 3画素による説明図 |
| 摂動の数値要約 | [metrics.csv](../outputs/sanity_checks/augmentation_strengths_train_fold1/metrics.csv)・[summary.json](../outputs/sanity_checks/augmentation_strengths_train_fold1/summary.json)。fold 1のtrain 8試料×128画素による強度確認。CV指標による最適化ではない |
| B0・B1・A0・M00のOOF確認 | `outputs/sanity_checks/a0_m00_oof_visualization/`のPNG・CSV。[OOF sanity仕様](design/oof_sanity_visualization.md)に従う。全主条件の最終比較には代用しない |
| 主8条件のK依存性 | [01_main_metrics_k_sweep.png](../outputs/experiments/production_v1/results/figures/main_oof_v1/01_main_metrics_k_sweep.png)。単段幅2×2、上段LLA 3・5、下段LLA 9・LFR(TGN+FS) |
| 主8条件の試料別分布 | [02_k8_distributions.png](../outputs/experiments/production_v1/results/figures/main_oof_v1/02_k8_distributions.png)。$K_0=8$、試料別平均・macro平均・共通対象数 |
| 主要3比較＋追加ablationのpaired差 | [03_paired_k_sweep.png](../outputs/experiments/production_v1/results/figures/main_oof_v1/03_paired_k_sweep.png)。M11−B0・M11−B1・M11−M00・A1−A0・M11−A1。ARI差は試料内3反復対平均同士の差 |
| 主条件の表・出典 | `outputs/experiments/production_v1/results/figures/main_oof_v1/`のCSV 11個と[report.json](../outputs/experiments/production_v1/results/figures/main_oof_v1/report.json)。Occupancyと交互作用は表のみ。内容一覧・再生成は[runbook](experiment_runbook.md#oof-reporting) |
| 全体fit・K8の6条件比較 | [01_representative_samples_6x7.png](../outputs/experiments/global_v1/results/figures/global_k8_v1/01_representative_samples_6x7.png)。固定代表7試料を列、B0・B1・A0・A1・M00・M11を行に配置。CVの予測ではなく全体fitによる記述 |
| M11・K8の代表試料マップ | [07_representative_samples_1x7.png](../outputs/experiments/global_v1/results/figures/global_k8_v1/M11/labels/07_representative_samples_1x7.png)。同じ`labels/`の00〜06・08に全49試料を保存 |
| M11の全体代表スペクトル | [01_representative_spectra.png](../outputs/experiments/global_v1/results/figures/global_k8_v1/M11/01_representative_spectra.png)。上段SNV・反射率、下段疑似吸光度のSG二次微分。試料等重み平均・試料間IQR |
| M11の試料別スペクトル・寄与数 | [sample_spectra.csv](../outputs/experiments/global_v1/results/figures/global_k8_v1/M11/sample_spectra.csv)、[spectrum_counts.csv](../outputs/experiments/global_v1/results/figures/global_k8_v1/M11/spectrum_counts.csv)、[wavelengths.csv](../outputs/experiments/global_v1/results/figures/global_k8_v1/wavelengths.csv)。試料ID・表示Cluster ID・kindで選び、`band_000`〜`band_255`をnmへ対応づける。試料別詳細PNGは未実装 |
| K8のmatching・出典 | [report.json](../outputs/experiments/global_v1/results/figures/global_k8_v1/report.json)、[M11/matching.csv](../outputs/experiments/global_v1/results/figures/global_k8_v1/M11/matching.csv)。元番号とM00基準の表示番号を区別。親図表はPNG 67枚・CSV 44個 |
| 6条件のPCA補助図 | [01_pca_density_clusters.png](../outputs/experiments/global_v1/results/figures/global_k8_v1/pca-latent-2d/01_pca_density_clusters.png)。2行6列・共通401,408画素。B1は既存PC、他条件は別々のPCA。軸の対応・化学情報量・優劣を図の見栄えから推定しない。掲載は未定 |

摂動例の選定・再生成は[runbook](experiment_runbook.md#input-preparation)を参照する。
TGNの固定例は2.5・5・7.5度で採用上限5度を超える例を含み、shiftの符号は順に＋・−・＋である。
固定値の説明図と学習時の一様分布を区別する。
Cutoffのproxyはreferenceの検出器列間のばらつきに基づき、試料画素のSNRや時間方向の測定noiseを直接示さない。

### 執筆を進める順序

1. CVの主指標・分布・paired差から、どの条件・指標で改善やtrade-offがあるかを書く。LLAは補正後、LFRは低いほど割当が安定する指標として扱う。
2. 全体fit・K8のマップと観測スペクトルで、分割された領域を記述する。M11に注目する理由と、結果を見た後の探索であることを明示する。
3. 個別試料で解釈したい差を絞り、既存CSVで足りる確認と、新しい図が必要な確認を分ける。追加図の依頼には対象試料・領域・支える主張を添える。

空間的一貫性の改善と物理化学的な意味の同定は別の主張である。外観との対応、観測スペクトル差、
化学的帰属を区別して記載し、ARI等でbaselineを下回る比較や解釈できない例も残す。
mask率25%・50%・75%の補助実験は実施し、専用OOF集計後に感度解析として結果を追記する。
結果を待つ間も既存の主条件結果で執筆を進められる。帯域積分map・PCAの追加色分けは保留中の候補として扱う。

### 解析フロー図

- 学習と全帯域可視の抽出・マッピングを分け、復元targetはTGN・shiftより前の原SNVからlossへ分岐させる。
- 学習済みencoderを固定してマッピングに使う。通常のマッピング経路にdecoderを入れない。
- 16個のpatchとCLSを区別する。主比較の50% maskは可視8 patch＋CLS、全可視時は16 patch＋CLSである。
- train表現によるクラスタ中心の推定と、固定中心への割当を分ける。CVと全体fitのfit範囲も明記する。
- 座標はラベル配置にだけ使う。背景0・クラスタ1始まりとし、色を劣化順序にしない。
- 入力形状は「行数 × 列数 × 256 bands」とし、波長の単位とSNVの縦軸を示す。模式線・模式色を実測や実験のKと混同しない。

## 3. 原稿で確認する事項

- 試料由来・採取関係・状態、撮像装置・測定条件。解析49試料と取得試料全体を区別し、未確認情報を補完しない。
- Multi-Otsu、SNV、PCA、球面クラスタリング、Transformer、MAE、denoisingの原典・引用箇所・書誌。
- mask率補助実験の完了状況と、FT-IRの[Open事項](design/README.md#open-items)。正式な目視評価などの未採用案を実施済みと扱わない。
- A1とA1−A0・M11−A1は、既存7条件の結果を得た後の追加ablationであること。M11−A1はmaskとloss対象の両方が異なること。
- 採用runの実行環境・完了状況、図表の出典、提出書式。
- 指標名と保存キーの対応：補正前LLA（$\mathrm{LLA}^{\mathrm{raw}}$）は`lla`、LLA（$\mathrm{LLA}$）は`adjusted_lla`。
- 代表指標はLLA、LFR(TGN+FS)、ARI、Cosine-Silhouette、Cluster Occupancyの優先順。
  OOF集計完了後の2026-09-19のユーザー指定による報告規約であり、元snapshotと報告CSVの対応を
  [評価仕様](design/evaluation_metrics.md#reporting)で確認する。補正後LLAの交互作用は報告CSVを参照する。
- 記号を初出と添字省略時に定義する。SNVと潜在のnorm、中心化行列$P$と偶然一致確率$P_m$、学習時潜在と全可視潜在を区別する。

CVは未知試料でのマップの性質を比較する。全体fit・観測スペクトル・位置対応FT-IRは化学的対応を探索する。
主張の範囲は[解釈メモ](interpretation_notes.md)に従い、未実施の解析を結果として記載しない。
