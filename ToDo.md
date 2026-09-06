# 実験実施 ToDo

更新日: 2026-09-06

この文書は現在の実行状態と残作業だけを管理する。固定済みの研究条件は
[研究設計](docs/design/README.md)、CLIと完了判定は[実験runbook](docs/experiment_runbook.md)、
完了済みのテストとpreflightは[検証履歴](docs/verification_history.md)を参照する。

## 現在の状態

| 項目 | 状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root | `outputs/experiments/production_v1/` |
| manifest | 新規作成・check済み。 `preflight_v1` の `complete.json` とSHA-256一致 |
| 完了 | A0・fold 1・repeat 1–3の学習、clustering、全test評価と各check |
| 次run | M00・fold 1・repeat 1を独立runとして開始する |
| 本番結果 | 3/75主ニューラルrun完了。OOF未完成のため性能比較はまだ行わない |

本番CV中は入力、manifest、config、実装を変更しない。ドキュメント整理は学習source hashの対象外で
あり、runのコードとデータには触れない。

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

## 直近の作業

- [x] A0・fold 1・repeat 1–3を800 epochまで学習し、完了条件とweights hashを確認した。
- [x] 3反復すべてで全7Kのclean test mapを作成し、既存成果物のcheckを通した。
- [x] 3反復すべてで全test評価を作成し、既存成果物のcheckを通した。
- [x] 実測時間、AMP skip、GPU peak、保存量とrepeat 2の再開を
  [検証履歴](docs/verification_history.md)へ記録した。
- [ ] M00・fold 1・repeat 1を新規学習し、完了条件とweights hashを確認する。
- [ ] 同runのclustering・評価を作成し、それぞれcheckを通す。

新規runと中断再開のコマンド、完了判定は[実験runbook](docs/experiment_runbook.md)を使う。

## 本番CV

### Baseline

- [ ] B1 PCAを5 foldsのtrain画素でfit・checkする。
- [ ] B0を5 folds × 3 repeatsでclustering・評価する（15組合せ）。
- [ ] B1を5 folds × 3 repeatsでclustering・評価する（15組合せ）。
- [ ] PCAの実solverとrepeat間再利用可否を各foldのfit記録から保存する。

### 主ニューラル条件

各runは800 epochとし、正常完了した重みだけをclusteringへ渡す。

- [ ] A0: 5 folds × 3 repeats（3/15 runs。fold 1の学習・clustering・評価が完了）
- [ ] M00: 5 folds × 3 repeats（0/15 runs。次runはfold 1・repeat 1）
- [ ] M10: 5 folds × 3 repeats（15 runs）
- [ ] M01: 5 folds × 3 repeats（15 runs）
- [ ] M11: 5 folds × 3 repeats（15 runs）
- [ ] 全75 runsでclean test mapのrun・checkを完了する（3/75組合せ完了）。
- [ ] 全75 runsで評価のrun・checkを完了する（3/75組合せ完了）。

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

## CV後に残る実装

この節の実装はAstraへ切り替えて進める予定（ユーザー指定）。これは作業時のモデル選択であり、
研究条件や成果物の再現性要件には含めない。

- [ ] AstraでOOF snapshotから主表、補助表、K依存性、mask率依存性、paired差の図表を生成する。
- [ ] 図表のsource hash、captionに必要な定義、試料間SDと反復間SDの区別を保存する。
- [ ] AstraでB0、B1、M00、M11を全49試料でfitする全体解釈pipelineを実装する。
- [ ] 全試料の$K_0=8$ map、Hungarian matching、代表・差スペクトル、潜在空間図を保存する。
- [ ] 固定済み7代表試料を本文用表示に使い、全49試料のmapも保存する。
- [ ] 劣化との対応を探索的解釈として記述し、定量的な劣化検出性能として扱わない。

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
