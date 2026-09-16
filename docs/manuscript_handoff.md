# 論文執筆への引き継ぎ

更新日: 2026-09-17

論文の執筆先は `C:\Users\PC_User\Python\Thesis` とする。このリポジトリでは研究の仕様・根拠・実施記録を管理する。
旧論文草稿から、研究文書に必要な導出・数値実装の記録と、外部での執筆に必要な注意点を残した。
外部執筆先への原稿・画像のコピーは、この整理では行っていない。章立て・原稿・執筆進捗は外部執筆先で管理する。

## 1. 残した内容と参照先

| 内容 | このリポジトリで参照する資料 |
| --- | --- |
| 研究の問い・採用理由・主張の範囲 | [研究概要](research_overview.md)、[ChemoMAEの位置づけ](chemomae_positioning.md)、[関連研究](related_work.md) |
| 前処理、cutoff、マスク、品質条件、SNV | [前処理仕様](design/preprocessing.md)、[本番前処理](../src/wood_degradation_map/preprocessing/production_preprocessing.py)、[波長grid](../src/wood_degradation_map/preprocessing/spectral_grid.py) |
| SNV・TGN・shift、masked loss、入力差・潜在差・残差の導出 | [数理的補足](mathematical_notes.md)。旧草稿のB番号を照合用に保持 |
| モデル・学習・CV・seed・クラスタリング | [実験プロトコル](design/experiment_protocol.md)、[モデルと抽出](../src/wood_degradation_map/experiments/neural.py)、[学習](../src/wood_degradation_map/experiments/training.py)、[クラスタリング](../src/wood_degradation_map/experiments/clustering.py) |
| 混合精度、学習loss履歴、inertiaの解釈 | [数値実装の記録](numerical_implementation_notes.md) |
| 補正前LLA・LLA、他の指標、集計・比較 | [評価指標](design/evaluation_metrics.md)。畳み込みによる定義と未定義条件も同書に保持 |
| 全体fit・スペクトル要約・表示・FT-IR計画 | [全体可視化と解釈](design/visualization_and_interpretation.md) |
| 実施済みの範囲・残作業・採用runの由来 | [ToDo](../ToDo.md)、[runbook](experiment_runbook.md)、[検証履歴](verification_history.md)、[決定記録](design/decisions.md) |

旧草稿の前処理・モデル・評価の定義は上記仕様と重なるため、本文一式をここへ複製しない。
数理的補足の記載は、対応する幾何診断を実装・実施したことを意味しない。診断一式は2026-09-13に実施計画から外したままとする。
旧草稿の実装照合は読み取りによるものであり、今回の整理でも学習・評価や実行時dtypeの計測を追加していない。

## 2. 図表を引き渡す際の確認

最終比較の必須図表は[評価指標第8.3節](design/evaluation_metrics.md#reporting)、全体マップと観測スペクトルは[全体可視化設計](design/visualization_and_interpretation.md)に従う。
掲載章・枚数は外部原稿で決め、図表の都合で条件・対象試料・指標を変更しない。

- Captionとともに、元ファイル、生成コード・config・manifestの版、試料・画素・fold・反復・K・条件・手法・fit範囲、選択根拠を記録する。
- 実測・要約・人工摂動・模式図を区別し、単位・凡例・matching基準を明記する。主要な反例や傾向の逆転は本文にも残す。
- 代表7試料は各樹種で保存有効画素数が最大の試料という固定規約を維持する。全49試料の補足表示と区別する。
- 代表スペクトルは2026-09-17の合意に従い、試料内平均→試料間の等重み平均で反射率・画素別SNVを要約する。
  画素別の疑似吸光度も同じ順で平均し、SciPyのSG二次微分（窓幅7点、多項式次数2、実波長間隔、interp）を比較する。
  変換不能画素の除外範囲、四分位範囲と寄与数は[集計仕様](design/visualization_and_interpretation.md#representative-spectra)に従う。
  この設計の確定を実装・解析済みと読み替えず、実行記録を別途確認する。
- B0・B1のOOF sanityはfold内B0基準・画素一致数による表示である。全体fitのM00基準・観測SNV代表線のcosine類似度による整列と混同しない。
- `outputs/`の元成果物は保持し、採用図の体裁調整と研究成果物の更新を区別する。

### 既存資料の候補と読み方

以下は旧草稿で候補として記録されていた資料であり、今回の整理で再生成・画像内容の再確認はしていない。

| 資料 | 出典と注意点 |
| --- | --- |
| 波長cutoff | [cutoff_decision.png](../outputs/preprocessing/production_v1/cutoff_decision.png)。元の256測定bandsのreference由来SNR proxy、閾値10、保持・除外範囲の単一panel。境界2308.72 nmは描画位置であり、補間上限は最後の保持波長2305.59 nm |
| 補間後の帯域分布 | [反射率](../outputs/preprocessing/production_v1/interpolated_reflectance_band_distribution.png)、[SNV](../outputs/preprocessing/production_v1/interpolated_snv_band_distribution.png)。最終入力の記述的な確認 |
| 前処理後の候補画素 | [final_snv_anomaly_candidates.png](../outputs/preprocessing/production_v1/final_snv_anomaly_candidates.png)。確認用の候補選択であり、劣化・異常の正解ラベルではない |
| TGN・shiftの例 | [TGN](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_noise_exact_angles_examples.png)、[shift](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_shift_exact_endpoints_examples.png)。同じ実測train SNV 3画素を使う。固定角度・固定shift幅の説明図と学習時の一様分布を区別 |
| 摂動の数値要約 | [metrics.csv](../outputs/sanity_checks/augmentation_strengths_train_fold1/metrics.csv)、[summary.json](../outputs/sanity_checks/augmentation_strengths_train_fold1/summary.json)。fold 1のtrain 8試料×128画素による強度sanityであり、outer-test・学習モデル・CV指標での最適化ではない |
| B0・B1の途中確認 | 旧草稿の参照先は `outputs/sanity_checks/b0_b1_oof_visualization/` 内の `labels/B0_representatives_k8_repeat1.png`、`labels/B1_representatives_k8_repeat1.png`、`silhouette_k_sweep.png`。今回の整理時にはこの3パスのファイルを確認できなかったため、採用時に保存先を確認する。全主条件の最終比較として提示しない |

摂動例の選定方法・軸範囲・再生成範囲は[runbook](experiment_runbook.md)と[生成スクリプト](../scripts/experiments/sanity_check_augmentation_strengths.py)を参照する。
TGNの固定例は2.5・5・7.5度で、採用上限5度を超える候補も含む。Shiftの3例の符号は順に＋・−・＋である。
Cutoffのproxyはreferenceの検出器列間のばらつきに基づく量であり、試料画素のSNRや時間方向の測定ノイズを直接示さない。

### 解析フロー図を再作図する際の注意

旧参考画像は画像生成による模式図で、実測データ・解析結果ではない。画像自体は旧原稿フォルダとともに削除する。
再作図で保持する科学的な対応は次のとおり。

- 上段を学習、下段を全帯域可視の抽出・マッピングとし、復元targetはTGN・shiftより前の原SNVからlossへ分岐させる。
- 学習済み重みの引継ぎはencoderから固定encoderへ接続する。通常のマッピング経路にdecoderを入れない。
- 16個のスペクトルpatchとCLSを区別する。主比較の50% maskは可視8 patch＋CLS、全可視時は16 patch＋CLSである。
- Cosine-KMeansのtrain表現による中心推定と固定中心への割当を分ける。CVと全体fitのfit範囲を区別する。
- 座標はencoder・クラスタリングの特徴へ入れず、最後のラベル配置にのみ使う。背景0、クラスタ1始まりとし、色を劣化順序にしない。
- 入力形状は「行数 × 列数 × 256 bands」とし、white referenceの記号Wとの衝突を避ける。SNVの縦軸と波長の単位を正しく示す。
- 模式線からSNVの数値的不変量を読み取らない。4色などの模式表示を実験の代表K=8と同一視しない。

## 3. 外部原稿で確認する事項

- 試料由来・採取関係・状態、撮像装置・測定条件。解析49試料と取得試料全体の数・採取関係は別に確認し、未整理情報を推測で補わない。
- Multi-Otsu、SNV、PCA、球面クラスタリング、Transformer、MAE、denoisingの原典・引用箇所・書誌。
- vMFの数値仕様、FT-IRの対象・位置対応・反復・前処理等のOpen事項。Fixed・実装済み・実施済みを区別する。
- 採用runの実行環境・完了状況、図表の出典、提出書式。最新状況は各runの記録とToDoで確認する。
- 指標名は「補正前LLA」と「LLA」。数式ではそれぞれ$\mathrm{LLA}^{\mathrm{raw}}$と$\mathrm{LLA}$であり、保存キーは`lla`と`adjusted_lla`に対応する。名称変更だけでは主評価・補助診断の位置づけは変更していない。
- 記号は各節の初出と添字省略時に定義する。SNVのnormと潜在の単位norm、中心化行列$P$と偶然一致確率$P_m$、学習時潜在と全可視潜在を区別する。

CVは未知試料へのマップの性質、全体fit・観測スペクトル・位置対応FT-IRは化学的対応の探索として扱う。
教師なし指標の改善だけで劣化精度・境界の正しさ・化学情報の保存を主張しない。CV開始後の判断を事前計画へ遡及させない。
