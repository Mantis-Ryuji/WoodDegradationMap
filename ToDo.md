# 研究・実装 ToDo

更新日: 2026-09-16。実験の状態は2026-09-15までに確認・共有された記録に基づく。
今回の文書整理では学習・評価・成果物checkを再実行していない。

本書は残作業を管理する。[文書案内](docs/README.md)から、
[固定設計](docs/design/README.md)、[実行手順](docs/experiment_runbook.md)、[検証履歴](docs/verification_history.md)へ進む。
論文は外部で執筆する。研究資料との対応と注意点は[執筆への引き継ぎ](docs/manuscript_handoff.md)を参照する。

## 現在の状態と次のrun

| 項目 | 確認済みの状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root | `outputs/experiments/production_v1/` |
| manifest | split・共通train座標・augmentation contractは現行仕様で確定済み |
| baseline | B1 PCAの5 fits、B0・B1の全5 folds×3反復のclustering・評価・checkが完了 |
| 主ニューラル条件 | A0・M00・M01の全5 folds、M11のfold 1–2、M10のfold 1の各repeat 1–3で学習・clustering・評価・checkが完了 |
| 主実験の完了数 | NN学習・clustering・評価は各54/75。B0・B1を含むclustering・評価は各84/105組合せ |
| OOF sanity | B0・B1のPNG 3枚・CSV 3つを生成済み。[表示仕様](docs/design/oof_sanity_visualization.md) |
| 実行中 | なし（最終確認時点） |
| 次のrun | M10・fold 2–5・repeat 1–3。各foldで3反復の学習 → clean test map → clustering checkを順次実行し、3反復をまとめて全test評価 → evaluation checkまで完了する |
| 実行環境 | ChemoMAE v0.2.2。固定設定と環境確認は[runbook](docs/experiment_runbook.md)・[検証履歴](docs/verification_history.md) |

前処理・入力照合、学習と再開、clustering、評価、OOF数値集計の実装とpreflightは完了済み。
本文の代表7試料も固定済み。詳細な完了記録を本書へ重複掲載しない。

## 1. 主ニューラルCVの残り

各runは800 epochとし、正常完了した重みからclean test mapと評価を作り、各checkまで完了する。
[1 runの手順](docs/experiment_runbook.md#neural-run)を使用する。

- [x] A0の全5 folds・repeat 1–3を完了した（15/15 runs、2026-09-12）。
- [x] M00の全5 folds・repeat 1–3を完了した（15/15 runs、2026-09-14）。
- [ ] M10のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
- [x] M01の全5 folds・repeat 1–3を完了した（15/15 runs、2026-09-15）。
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
論文へ渡す成果物は[評価指標の必須図表](docs/design/evaluation_metrics.md#reporting)と[引き継ぎ時の確認](docs/manuscript_handoff.md)に従う。

- [ ] OOF snapshotを入力に、代表$K_0=8$の主表と、LLA・silhouette・ARI・occupancy・有効対象数の診断表を生成するpipelineを実装する。
- [ ] 全7Kの主指標・主要contrast・反復別曲線、mask率依存性、paired差、2×2交互作用の図表を実装する。主要比較はM11対B0・B1・M00とし、残りの計画比較も保持する。
- [ ] 未定義理由と共通対象数を保持し、試料間SD・反復間SD・ARIの集計を[報告規約](docs/design/evaluation_metrics.md#reporting)に合わせる。
- [ ] 元snapshotと図表のsource hash、条件・K・反復・集計対象、captionに必要な定義を保存し、原稿の図表から根拠をたどれるようにする。
- [ ] 完了した主条件・mask率の各OOF snapshotから図表を生成し、必須の表・図と原稿の掲載候補を照合する。vMFの比較図表は同補助実験の完了後に作成する。

## 5. 全体fitと解釈

対象はB0・B1・A0・M00・M11の5条件×Cosine-KMeans・vMFの2手法。
共通の表示番号は**M00＋Cosine-KMeans**を基準にし、観測SNV代表線のcosine類似度によるHungarian matchingで直接整列する。[全体可視化設計](docs/design/visualization_and_interpretation.md)に従う。
代表スペクトルの試料内平均→試料間の等重み平均、疑似吸光度のSG二次微分、および同じSNV平均線によるmatchingは**確定設計（Fixed）**とする。以下の未完了項目は実装・実施の残作業であり、集計方法の再検討を意味しない。
全体解釈pipelineは未実装で、既存のCV用CLIをそのまま全体fitへ使わない。

- [ ] 全49試料の画素抽出・manifest・seed適用・実行記録・保存先・CLIを実装する。
- [ ] PCAの全体fit、A0・M00・M11の各1回（計3回）の全体学習を実装・実施する。モデル・学習条件は既存実装を再利用する。
- [ ] 同じ表現・抽出座標・$K_0=8$でCosine-KMeansを5 fits行う。
- [ ] 数値仕様確定・検証後、同じ表現を再利用してvMFを5 fits行う。CV補助実験の735 fitsとは分ける。
- [ ] 全49試料のhard label map、M00基準のSNV代表線cosine類似度行列・matching対応表、確認用contingency・overlap、occupancy、使用クラスタ数、潜在空間図を保存する。
- [ ] 観測反射率、画素別SNVおよび画素別疑似吸光度を、試料・クラスタごとの波長別算術平均→対象画素が存在する試料間の等重み平均で要約する。SNV変換と対数変換は平均より先に行い、Decoderの復元値は集計へ混ぜない。表示とmatchingには同じ観測SNV平均線を用いる。
- [ ] 疑似吸光度の平均代表線にSciPyのSG二次微分（窓幅7点、次数2、実波長間隔、interp）を適用する。全帯域で反射率が正かつ有限という追加条件は疑似吸光度・二次微分の集計に限り、反射率・SNVの集計とmatchingには適用しない。
- [ ] 試料別平均線から波長別四分位範囲を求め、代表線ごとの寄与試料数・試料ID・画素数と、疑似吸光度解析の追加除外画素数を保存する。二次微分の四分位範囲は各試料の平均疑似吸光度へ同じSG処理を施した後に求める。差スペクトルでは対象と引き算の向きを記録する。
- [ ] 固定7代表試料を本文表示に使い、行をCosine-KMeans・vMF、列をB0・B1・A0・M00・M11とする比較図、KYOw試料ID、共通の表示規約を確認する。
- [ ] 劣化との対応を探索的に記述し、CVの指標改善と化学的な対応を区別する。

これらは全体マップと観測スペクトルの解釈資料として外部原稿へ渡す。

## 6. 既定計画に残るOpen事項

研究上のOpen事項は[研究設計の一覧](docs/design/README.md#open-items)で確認する。
任意の形状診断・vMF責務マップは、採用する場合だけ結果を見る前に定義を固定する。
位置対応FT-IRは測定計画があり、対象・位置対応・反復・前処理・指標などの詳細設計はOpen。

実行ごとの記録と保存は[runbook](docs/experiment_runbook.md#artifact-records)に従う。
全体fit対象の拡張やmatching基準などの決定時点は[決定記録](docs/design/decisions.md)に残す。

数理的補足B.5に対応する表現幾何の補助診断一式は、2026-09-13の見直しで実施項目から外した。
既定CVの比較に必須ではなく、未実装・未実施の候補として[判断理由と旧案](docs/design/representation_geometry_diagnostics.md)に残す。
LLA・silhouette・ARI・occupancy等の既定診断は第4節に含まれる。

## 7. 外部執筆への資料提供

論文の章立て・本文・図の体裁・引用・執筆進捗は `C:\Users\PC_User\Python\Thesis` で管理する。
2026-09-12に作成した第3章・付録A〜C等の旧草稿から、研究に必要な情報を2026-09-16に[引き継ぎ資料](docs/manuscript_handoff.md)とその参照先へ整理した。
本リポジトリで行う実験・図表生成の残作業は第4〜6節、操作と出典の記録はrunbookに集約する。
