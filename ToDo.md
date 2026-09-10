# 実験実施 ToDo

更新日: 2026-09-11。進捗はこの日までに確認・共有された状態を示す。

本書は残作業を管理する。[文書案内](docs/README.md)から、
[固定設計](docs/design/README.md)、[実行手順](docs/experiment_runbook.md)、[検証履歴](docs/verification_history.md)へ進む。

## 現在の状態と次のrun

| 項目 | 確認済みの状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root | `outputs/experiments/production_v1/` |
| manifest | split・共通train座標・augmentation contractは現行仕様で確定済み |
| baseline | B1 PCAの5 fits、B0・B1の全5 folds×3反復のclustering・評価・checkが完了 |
| 主ニューラル条件 | A0・M00・M10・M01のfold 1、M11のfold 1–2の各repeat 1–3で学習・clustering・評価・checkが完了 |
| 主実験の完了数 | NN学習・clustering・評価は各18/75。B0・B1を含むclustering・評価は各48/105組合せ |
| OOF sanity | B0・B1のPNG 3枚・CSV 3つを生成済み。[表示仕様](docs/design/oof_sanity_visualization.md) |
| 実行中 | なし（最終確認時点） |
| 次のrun | A0・fold 2・repeat 1–3。学習は同repeatを順次実行し、clean test map → 全test評価 → 各checkまで完了する |
| 実行環境 | ChemoMAE v0.2.2。固定設定と環境確認は[runbook](docs/experiment_runbook.md)・[検証履歴](docs/verification_history.md) |

前処理・入力照合、学習と再開、clustering、評価、OOF数値集計の実装とpreflightは完了済み。
本文の代表7試料も固定済み。詳細な完了記録を本書へ重複掲載しない。

## 1. 主ニューラルCVの残り

各runは800 epochとし、正常完了した重みからclean test mapと評価を作り、各checkまで完了する。
[1 runの手順](docs/experiment_runbook.md#neural-run)を使用する。

- [ ] A0のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
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

- [ ] OOF snapshotから主表、補助表、K依存性、mask率依存性、paired差の図表を生成する。
- [ ] 図表のsource hash、captionに必要な定義、試料間SDと反復間SDの区別を保存する。
- [ ] [必須の表・図](docs/design/evaluation_metrics.md#reporting)に沿って、主要比較・ablation・有効対象数を確認する。

## 5. 全体fitと解釈

対象はB0・B1・A0・M00・M11の5条件×Cosine-KMeans・vMFの2手法。
共通の表示番号は**A0＋Cosine-KMeans**を基準にする。[全体可視化設計](docs/design/visualization_and_interpretation.md)に従う。
全体解釈pipelineは未実装で、既存のCV用CLIをそのまま全体fitへ使わない。

- [ ] 全49試料の画素抽出・manifest・seed適用・実行記録・保存先・CLIを実装する。
- [ ] PCAの全体fit、A0・M00・M11の各1回（計3回）の全体学習を実装・実施する。モデル・学習条件は既存実装を再利用する。
- [ ] 同じ表現・抽出座標・$K_0=8$でCosine-KMeansを5 fits行う。
- [ ] 数値仕様確定・検証後、同じ表現を再利用してvMFを5 fits行う。CV補助実験の735 fitsとは分ける。
- [ ] 全49試料のhard label map、A0基準のmatching・overlap、occupancy、代表・差スペクトル、潜在空間図を保存する。
- [ ] 固定7代表試料を本文表示に使い、KYOw試料IDと共通の表示規約を確認する。
- [ ] 劣化との対応を探索的に記述し、CVの指標改善と化学的な対応を区別する。

## 6. 未決定事項

研究上のOpen事項は[研究設計の一覧](docs/design/README.md#open-items)で確認する。
任意の形状診断・vMF責務マップは、採用する場合だけ結果を見る前に定義を固定する。
位置対応FT-IRは測定計画があり、対象・位置対応・反復・前処理・指標などの詳細設計はOpen。

実行ごとの記録と保存は[runbook](docs/experiment_runbook.md#artifact-records)に従う。
全体fit対象の拡張やmatching基準などの決定時点は[決定記録](docs/design/decisions.md)に残す。
