# 研究・実装・執筆 ToDo

更新日: 2026-09-12。実験の状態は同日までに確認・共有された記録に基づく。
今回の文書整理では学習・評価・成果物checkを再実行していない。

本書は残作業を管理する。[文書案内](docs/README.md)から、
[固定設計](docs/design/README.md)、[実行手順](docs/experiment_runbook.md)、[検証履歴](docs/verification_history.md)へ進む。
原稿と資料の対応は[修論ドラフト](thesis/README.md)・[執筆計画](thesis/writing_plan.md)を参照する。

## 現在の状態と次のrun

| 項目 | 確認済みの状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root | `outputs/experiments/production_v1/` |
| manifest | split・共通train座標・augmentation contractは現行仕様で確定済み |
| baseline | B1 PCAの5 fits、B0・B1の全5 folds×3反復のclustering・評価・checkが完了 |
| 主ニューラル条件 | A0の全5 folds、M11のfold 1–2、M00・M10・M01のfold 1の各repeat 1–3で学習・clustering・評価・checkが完了 |
| 主実験の完了数 | NN学習・clustering・評価は各30/75。B0・B1を含むclustering・評価は各60/105組合せ |
| OOF sanity | B0・B1のPNG 3枚・CSV 3つを生成済み。[表示仕様](docs/design/oof_sanity_visualization.md) |
| 実行中 | なし（最終確認時点） |
| 次のrun | M00・fold 2・repeat 1–3。同repeatの学習 → clean test map → clustering checkを順次実行し、3反復をまとめて全test評価 → evaluation checkまで完了する |
| 実行環境 | ChemoMAE v0.2.2。固定設定と環境確認は[runbook](docs/experiment_runbook.md)・[検証履歴](docs/verification_history.md) |

前処理・入力照合、学習と再開、clustering、評価、OOF数値集計の実装とpreflightは完了済み。
本文の代表7試料も固定済み。詳細な完了記録を本書へ重複掲載しない。

## 1. 主ニューラルCVの残り

各runは800 epochとし、正常完了した重みからclean test mapと評価を作り、各checkまで完了する。
[1 runの手順](docs/experiment_runbook.md#neural-run)を使用する。

- [x] A0の全5 folds・repeat 1–3を完了した（15/15 runs、2026-09-12）。
- [ ] M00のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
- [ ] M10のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
- [ ] M01のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
- [ ] M11のfold 3–5・repeat 1–3を完了する（6/15 runs完了。fold 1–2は完了）。
- [ ] 主7条件の5 folds×3反復の完全性を確認し、`main_oof_v1`を作成・checkする。
- [ ] 欠損・失敗・中断・未定義指標と理由がOOF集計に保持されていることを確認する。

## 2. Mask率補助実験

- [ ] M11-25の5 folds×3反復を学習し、clustering・評価・checkまで完了する（15 runs）。
- [ ] M11-75の5 folds×3反復を学習し、clustering・評価・checkまで完了する（15 runs）。
- [ ] 50%は主実験M11の結果を再利用し、3条件の完全性を確認して`mask_rate_oof_v1`を作成・checkする。

主条件とmask率のOOFは[runbookのOOF集計](docs/experiment_runbook.md#oof-aggregation)に従い、別snapshotにする。

## 3. vMF補助実験

主7条件×5 folds×3反復×7Kの735 fits。追加のNN学習・PCA fitは行わない。
範囲・利用版・退化成分の扱いは決定済みで、数値仕様と専用pipelineは未完了。
[実験プロトコル](docs/design/experiment_protocol.md#vmf-supplementary)と[評価規約](docs/design/evaluation_metrics.md#vmf-evaluation)に従う。

- [ ] v0.2.2の数値関数・公開helper・初期化・最終尤度・保存復元・退化成分を検証する。
  16次元・256次元の参照値比較、CPU小規模、chunk、GPU最小確認を含む。
- [ ] 数値精度・EM停止条件・集中度設定をvMFのtest結果を見る前に固定する。
- [ ] 専用のfit・評価・check・OOFと独立した出力先を設計・実装する。
- [ ] 本番CV後、既存の主7条件の重み・PCAと共通train画素で735 fitsを実施する。
- [ ] 同じtest全画素・共通摂動で評価し、完了・失敗・未定義値を保持してOOF集計する。
- [ ] 計画contrast・2×2交互作用・K依存性をCosine-KMeansと併記する。

## 4. 全CV完了後の図表生成

この節と次節の実装はAstraへ切り替えて進める予定（ユーザー指定）。
作業時のモデル選択であり、研究条件や成果物の再現性要件には含めない。
B0・B1 OOF sanityは生成済みだが、以下の最終報告用図表pipelineは未実装。
修論の結果章5.2〜5.4と付録Eへ渡す成果物として、[執筆計画の図表対応](thesis/writing_plan.md)と照合する。

- [ ] OOF snapshotを入力に、代表$K_0=8$の主表と、補正LLA・silhouette・ARI・occupancy・有効対象数の診断表を生成するpipelineを実装する。
- [ ] 全7Kの主指標・主要contrast・反復別曲線、mask率依存性、paired差、2×2交互作用の図表を実装する。主要比較はM11対B0・B1・M00とし、残りの計画比較も保持する。
- [ ] 未定義理由と共通対象数を保持し、試料間SD・反復間SD・ARIの集計を[報告規約](docs/design/evaluation_metrics.md#reporting)に合わせる。
- [ ] 元snapshotと図表のsource hash、条件・K・反復・集計対象、captionに必要な定義を保存し、原稿の図表から根拠をたどれるようにする。
- [ ] 完了した主条件・mask率の各OOF snapshotから図表を生成し、必須の表・図と原稿の掲載候補を照合する。vMFの比較図表は同補助実験の完了後に作成する。

## 5. 全体fitと解釈

対象はB0・B1・A0・M00・M11の5条件×Cosine-KMeans・vMFの2手法。
共通の表示番号は**A0＋Cosine-KMeans**を基準にする。[全体可視化設計](docs/design/visualization_and_interpretation.md)に従う。
全体解釈pipelineは未実装で、既存のCV用CLIをそのまま全体fitへ使わない。

- [ ] 全49試料の画素抽出・manifest・seed適用・実行記録・保存先・CLIを実装する。
- [ ] PCAの全体fit、A0・M00・M11の各1回（計3回）の全体学習を実装・実施する。モデル・学習条件は既存実装を再利用する。
- [ ] 同じ表現・抽出座標・$K_0=8$でCosine-KMeansを5 fits行う。
- [ ] 数値仕様確定・検証後、同じ表現を再利用してvMFを5 fits行う。CV補助実験の735 fitsとは分ける。
- [ ] 全49試料のhard label map、A0基準のmatching・overlap、occupancy、使用クラスタ数、潜在空間図を保存する。
- [ ] 観測反射率・SNVの代表線を試料内中央値→試料間中央値・四分位範囲で要約し、寄与試料・画素数と差スペクトルの引き算の向きを保存する。Decoderの復元値はこの観測スペクトル集計へ混ぜない。
- [ ] 固定7代表試料を本文表示に使い、行をCosine-KMeans・vMF、列をB0・B1・A0・M00・M11とする比較図、KYOw試料ID、共通の表示規約を確認する。
- [ ] 劣化との対応を探索的に記述し、CVの指標改善と化学的な対応を区別する。

これらは修論の結果章5.5と付録Eに対応する。表現幾何の補助診断は第6.2節で管理する。

## 6. 補助診断と未決定事項

### 6.1 既定計画に残るOpen事項

研究上のOpen事項は[研究設計の一覧](docs/design/README.md#open-items)で確認する。
任意の形状診断・vMF責務マップは、採用する場合だけ結果を見る前に定義を固定する。
位置対応FT-IRは測定計画があり、対象・位置対応・反復・前処理・指標などの詳細設計はOpen。

実行ごとの記録と保存は[runbook](docs/experiment_runbook.md#artifact-records)に従う。
全体fit対象の拡張やmatching基準などの決定時点は[決定記録](docs/design/decisions.md)に残す。

### 6.2 表現幾何の補助診断（方針合意・詳細Open・未実装）

MAE群M00・M10・M01・M11を軸に、入力のどの変動が潜在cosine幾何で拡大・縮小されるかを調べる。
クラスタ平均と画素対・同一画素への摂動の両方を扱う方針は合意済み。
数理は改訂した[付録B.5](thesis/appendices/mathematical_details.md#latent-decoder)、診断と主張の対応・実施順序は
[診断計画](docs/design/representation_geometry_diagnostics.md)を参照する。診断実装・実行と実測結果の原稿反映は未実施。

- [ ] 共通群・画素対の選び方、対象fold・反復、重み、摂動回数、数値閾値、保存物などの[Open事項](docs/design/representation_geometry_diagnostics.md#open-items)を確定する。
- [ ] 同じ画素・同じ重みで入力、画素ごとの潜在表現、残差を平均し、群間の方向差と群内の集中度を診断する。
- [ ] 共通画素対の入力距離と潜在距離、SVD方向別の入力差・残差差を対応づけ、拡大と縮小の双方を調べる。
- [ ] 同一画素への追加摂動による変動を平均のずれとばらつきに分け、全可視推論での安定性と対応づける。固定maskの診断は必要性を判断して追加する。
- [ ] 実測後、改訂済みB.5の数理と診断結果を照合し、結果・考察と補足図表へ反映する。

平均スペクトルをencoderへ入力した値で画素ごとの潜在平均を代用しない。
A1の追加や再学習はこの計画に含めず、対象モデル自身のクラスタによる診断と共通群での条件比較を区別する。
学習平均スペクトルとの復元比較、AMPとFP32の比較は、[解釈メモ](docs/interpretation_notes.md#decoder-residual-discussion)に残す別の未採用候補である。

## 7. 論文原稿と図表の整備

章ごとの論点は[構成案](thesis/outline.md)、資料・実装・図表の対応は[執筆計画](thesis/writing_plan.md)を正とする。
ここでは執筆の残作業と、実験・実装への依存を管理する。

- [x] 第3章と付録A〜Cの第1稿を作成し、`docs/`・`thesis/`の論点、本文・付録・執筆メモ、記号・参照を整理した（2026-09-12）。
- [ ] 第4章の比較条件・CV・反復・K・評価・集計と、付録Dを固定仕様に基づいて執筆する。vMFとFT-IRのOpen部分は未確定として残す。
- [ ] 解析フロー参考図を最終作図へ移し、摂動・モデル・CV・指標の説明図と条件表を整える。Caption・記号・図の出典を本文と照合する。
- [ ] 第1章の背景・関連研究・採用理由を、主張できる範囲と引用根拠が対応する文章へ整える。標準手法の原典・書誌を確認する。
- [ ] 第2章に必要な試料由来・採取関係・状態・撮像装置・測定条件を整理する。FT-IRは詳細決定・実施後に対応する記載を整える。
- [ ] 本書の第4〜5節の成果物と実行記録を確認して、第5章・付録Eの結果、第6章の考察、第7章の結論を執筆する。第6.2節の補助診断は、実施・確認できた範囲を反映する。
- [ ] 採用runの環境・重み・source hashと図表の出典、共通記号、引用、提出書式を照合し、LaTeXの最終稿へ移す。
