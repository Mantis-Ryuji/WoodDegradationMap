# 実験実施 ToDo

更新日: 2026-09-10

この文書は現在の実行状態と残作業だけを管理する。固定済みの研究条件は
[研究設計](docs/design/README.md)、CLIと完了判定は[実験runbook](docs/experiment_runbook.md)、
完了済みのテストとpreflightは[検証履歴](docs/verification_history.md)を参照する。

## 現在の状態

| 項目 | 状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root | `outputs/experiments/production_v1/` |
| manifest | split・共通train座標・augmentation contractは現行仕様で確定済み |
| 完了 | B0・B1の全5 folds × 3 repeats、およびA0・M00・M10・M01・M11のfold 1・repeat 1–3でclean-map clustering・全test評価と各checkが完了 |
| 実行中 | なし |
| 次の工程 | M11・fold 2・repeat 1の学習 → clean test map → 全test評価 |
| 実行環境 | ChemoMAE v0.2.2。noise $U(0,5^\circ)$、shift $U(-2,2)$ |
| 検証 | manifest・neural・clustering・evaluation・records関連テストは現行コードで全件passed |
| 本番結果 | 主ニューラル実験の学習・clustering・評価は各15/75完了。B0・B1を含む主7条件のclustering・評価は各45/105組合せ完了 |
| vMF補助実験 | 主7条件・5-fold・3反復・共通7Kの735 fits。修正版v0.2.2を導入済み、数値検証・設定確定は未完了 |

本番CVでは入力・manifest・研究条件を固定し、成果物に実行時の設定と環境を記録する。
環境変更とその確認結果は[検証履歴](docs/verification_history.md)で管理する。

## 完了済みの準備

- [x] 200 Hz本番前処理を `production_v1` として再生成した。
- [x] 補間後・SNV前の負の反射率496画素をtrain・test共通で背景化した。
- [x] 49試料とmetadata、品質表、確認図を照合した。
- [x] KYOw単位の5-fold splitと$q=8192$の共通train座標を固定した。
- [x] train/test間で同じKYOwが重複しないことを検証した。
- [x] B0、B1、全ニューラル条件の学習・再開・表現抽出・全K clusteringを実装した。
- [x] clean test map、LLA、LFR、silhouette、ARI、occupancy、試料macro、paired差を実装した。
- [x] OOF完全性検証と計画比較のsnapshot出力を実装した。
- [x] CPUテスト、GPU smoke、B0・fold 1の実データ全量preflightを完了した。
- [x] 本文代表例を各樹種の保存有効画素数最大の7試料に固定した。
- [x] `production_v1` のmanifestを新規作成し、preflightとの一致を確認した。

## 直近の完了と次のrun

M11・fold 1・repeat 1–3は、学習completionと重み・checkpointの整合確認、全7Kのclustering、
全test評価、各保存済み成果物のcheckまで完了した。続いてB1 PCAを5 foldsで各1回fit・checkし、
B0・B1の全5 folds × 3 repeats、計30組合せのclustering・評価と各checkを完了した。
次はM11・fold 2・repeat 1から主ニューラル実験を再開する。
完了済みrunの実測値とcheck結果は[検証履歴](docs/verification_history.md)にまとめる。

- [x] A0・M00のfold 1・repeat 1–3で、現行設定の全test評価とcheckを完了する。
- [x] M10・fold 1・repeat 1–3を800 epochまで学習し、completion、training history、重み・checkpointを照合する。
- [x] M10・fold 1・repeat 1–3の全7K clean test mapと全test評価を作成し、各checkを通す。
- [x] M01・fold 1・repeat 1–3を800 epochまで学習し、completion、training history、重み・checkpointを照合する。
- [x] M01・fold 1・repeat 1–3の全7K clean test mapと全test評価を作成し、各checkを通す。
- [x] M11・fold 1・repeat 1–3を800 epochまで学習し、全7K clean test mapと全test評価の各checkを通す。

新規runと中断再開のコマンド、完了判定は[実験runbook](docs/experiment_runbook.md)を使う。

## 本番CV

### Baseline

- [x] B1 PCAを5 foldsのtrain画素で各1回、計5回fit・checkする。
- [x] B0を5 folds × 3 repeatsでclustering・評価する（15組合せ）。
- [x] B1を5 folds × 3 repeatsでclustering・評価する（15組合せ）。
- [x] PCAの実solverとrepeat間再利用可否を各foldのfit記録へ保存する。

### 主ニューラル条件

各runは800 epochとし、正常完了した重みだけをclusteringへ渡す。

- [ ] A0: 5 folds × 3 repeats（fold 1の3/15 runsで学習・clustering・評価完了）
- [ ] M00: 5 folds × 3 repeats（fold 1の3/15 runsで学習・clustering・評価完了）
- [ ] M10: 5 folds × 3 repeats（fold 1の3/15 runsで学習・clustering・評価完了）
- [ ] M01: 5 folds × 3 repeats（fold 1の3/15 runsで学習・clustering・評価完了）
- [ ] M11: 5 folds × 3 repeats（fold 1の3/15 runsで学習・clustering・評価完了）
- [ ] 全75 runsでclean test mapのrun・checkを完了する（15/75組合せ完了）。
- [ ] 全75 runsで評価のrun・checkを完了する（15/75組合せ完了）。

### Mask率補助実験

- [ ] M11-25: 5 folds × 3 repeats（15 runs）
- [ ] M11-75: 5 folds × 3 repeats（15 runs）
- [ ] 全30 runsでclean test mapと評価のrun・checkを完了する。
- [ ] M11の50%結果は主実験から再利用し、再学習しない。

### OOF集計

- [ ] 主7条件の5 folds × 3 repeatsが完全であることを確認する。
- [ ] `main_oof_v1` を作成・checkする。
- [ ] M11-25、M11、M11-75の5 folds × 3 repeatsが完全であることを確認する。
- [ ] `mask_rate_oof_v1` を作成・checkする。
- [ ] 欠損、失敗、中断、未定義指標とその理由が集計に保持されていることを確認する。
- [ ] CV結果からbest条件やseedを事後選択しない。

### vMFクラスタリング補助実験

範囲・数値仕様は[実験プロトコル第5.2節](docs/design/experiment_protocol.md#vmf-supplementary)、
評価は[評価指標第8.4節](docs/design/evaluation_metrics.md#vmf-evaluation)を参照する。
既存のニューラル学習回数は増やさない。

- [x] 主7条件 × 5 folds × 3 repeats × 共通7K、計735 fitsの範囲を2026-09-07に確定した。
- [x] 静的レビューで指摘した問題の修正版ChemoMAE v0.2.2を採用・導入した。
- [x] 退化成分の扱いを確定した。責務ゼロ時は方向・集中度を保持し、重み付き和ゼロ時は方向を保持して`kappa_min`へ下げる。
- [ ] v0.2.2の数値関数・公開helper・初期化・最終尤度・保存復元・退化成分を検証する。
  16次元・256次元の参照値比較、CPU小規模、chunk、GPU最小確認を含む。
- [ ] 数値精度・EM停止条件・集中度設定をvMFのtest結果を見る前に固定する。
- [ ] 専用のfit・評価・check・OOFと独立した出力先を設計・実装する。
- [ ] 本番CV後、既存の主7条件の重み・PCAと共通train画素で735 fitsを実施する。
- [ ] 同じtest全画素・共通摂動で評価し、完了・失敗・未定義値を保持してOOF集計する。
- [ ] 既定の条件contrast・2×2交互作用・K依存性をCosine-KMeansと併記する。

## CV後に残る実装

この節の実装はAstraへ切り替えて進める予定（ユーザー指定）。これは作業時のモデル選択であり、
研究条件や成果物の再現性要件には含めない。

現行CV（Cosine-KMeans）の学習・評価・OOF数値集計は実装済みで、以下の図表生成と全体解釈pipelineは未実装である。
A0のモデル・学習条件は既存実装を再利用し、全体解釈pipelineの対象を最初から5条件とする。
全体学習用の画素抽出・manifest、seedの適用、実行記録・保存先・CLIは、このpipelineで扱う。

- [ ] AstraでOOF snapshotから主表、補助表、K依存性、mask率依存性、paired差の図表を生成する。
- [ ] 図表のsource hash、captionに必要な定義、試料間SDと反復間SDの区別を保存する。
- [ ] AstraでB0、B1、A0、M00、M11を全49試料でfitまたは学習する全体解釈pipelineを実装する。ニューラル学習はA0・M00・M11の各1回、計3回とする。
- [x] 2026-09-08、全体解釈にvMFを追加し、当時の4条件 × 2手法の比較を[可視化設計](docs/design/visualization_and_interpretation.md)へ反映した。
- [x] 2026-09-09、全体学習・可視化へA0を追加し、5条件 × 2手法、全体3学習、各手法5 fitsへ設計文書を更新した。CV開始後の追加決定として記録した。
- [ ] vMFの数値仕様確定・検証後、全体学習の同じ表現・抽出座標・$K_0=8$で5 fitsを行う。vMFのための追加の表現学習は0回とし、CVの735 fitsとは分ける。
- [ ] 全試料の5条件 × 2手法のhard label map、B0のCosine-KMeansを共通基準とするHungarian matching、overlap、occupancy、代表・差スペクトル、潜在空間図を保存する。
- [ ] 固定済み7代表試料を本文用表示に使い、全49試料のmapも保存する。
- [ ] 劣化との対応を探索的解釈として記述し、定量的な劣化検出性能として扱わない。

任意の最大posterior責務マップは採用・表示規約がOpenであり、[可視化設計第4.4節](docs/design/visualization_and_interpretation.md#vmf-responsibility-maps)で管理する。

任意の孤立label・連結成分shape診断は定義がOpenである。採用する場合だけ、結果を見る前に
connectivity、閾値、分母を決めて[評価指標](docs/design/evaluation_metrics.md)へ反映してから実装する。

## 実行ごとの記録

各runで次を残す。

- condition、fold、repeat、seedとmanifest・code・config hash
- status、epoch、attempted/optimizer updates、AMP skips
- 学習のepoch時間合計、clustering・評価のwall timeとGPU peak allocated/reserved、保存量
- checkpointから再開した場合のsource pathと整合確認
- clustering・評価のcompletionとcheck結果

重みとcheckpointはGit管理しない。config、manifest、数値結果、図、completion記録は
`outputs/experiments/production_v1/` に残す。

現行の本番学習completionはGPU peakを保存しないため、学習時の値を事後推定しない。学習GPU peakは
preflightの実測値だけを工学的参考値として扱い、本番CV中に記録項目を追加するコード変更は行わない。
