# 論文執筆への引き継ぎ

論文の章立て・原稿・執筆進捗は `C:\Users\PC_User\Python\Thesis` で管理する。
本リポジトリから研究条件、根拠、実験成果物を参照する。

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
- OOF sanityのfold内B0基準・画素一致数による整列と、全体fitのM00基準・SNV類似度による整列を区別する。
- `outputs/`の元成果物を保持し、原稿での体裁調整と解析結果を区別する。Fixedの仕様を実装・解析済みとは扱わない。

### 図表候補と用途

| 資料 | 出典と注意点 |
| --- | --- |
| 波長cutoff | [cutoff_decision.png](../outputs/preprocessing/production_v1/cutoff_decision.png)。Reference由来SNR proxy、閾値10、保持・除外範囲。描画境界2308.72 nmと補間上限2305.59 nmを区別する |
| 補間後の帯域分布 | [反射率](../outputs/preprocessing/production_v1/interpolated_reflectance_band_distribution.png)・[SNV](../outputs/preprocessing/production_v1/interpolated_snv_band_distribution.png)。最終入力の記述的な確認 |
| 前処理後の候補画素 | [final_snv_anomaly_candidates.png](../outputs/preprocessing/production_v1/final_snv_anomaly_candidates.png)。確認用の候補であり、劣化・異常の正解ラベルではない |
| TGN・shiftの例 | [TGN](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_noise_exact_angles_examples.png)・[shift](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_shift_exact_endpoints_examples.png)。同じ実測train SNV 3画素による説明図 |
| 摂動の数値要約 | [metrics.csv](../outputs/sanity_checks/augmentation_strengths_train_fold1/metrics.csv)・[summary.json](../outputs/sanity_checks/augmentation_strengths_train_fold1/summary.json)。fold 1のtrain 8試料×128画素による強度確認。CV指標による最適化ではない |
| B0・B1・A0・M00のOOF確認 | `outputs/sanity_checks/a0_m00_oof_visualization/`のPNG・CSV。[OOF sanity仕様](design/oof_sanity_visualization.md)に従う。全主条件の最終比較には代用しない |

摂動例の選定・再生成は[runbook](experiment_runbook.md#input-preparation)を参照する。
TGNの固定例は2.5・5・7.5度で採用上限5度を超える例を含み、shiftの符号は順に＋・−・＋である。
固定値の説明図と学習時の一様分布を区別する。
Cutoffのproxyはreferenceの検出器列間のばらつきに基づき、試料画素のSNRや時間方向の測定noiseを直接示さない。

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
- vMF数値仕様、FT-IR、正式な目視評価の[Open事項](design/README.md#open-items)と実施状況。
- 採用runの実行環境・完了状況、図表の出典、提出書式。
- 指標名と保存キーの対応：補正前LLA（$\mathrm{LLA}^{\mathrm{raw}}$）は`lla`、LLA（$\mathrm{LLA}$）は`adjusted_lla`。
- 記号を初出と添字省略時に定義する。SNVと潜在のnorm、中心化行列$P$と偶然一致確率$P_m$、学習時潜在と全可視潜在を区別する。

CVは未知試料でのマップの性質を比較する。全体fit・観測スペクトル・位置対応FT-IRは化学的対応を探索する。
主張の範囲は[解釈メモ](interpretation_notes.md)に従い、未実施の解析を結果として記載しない。
