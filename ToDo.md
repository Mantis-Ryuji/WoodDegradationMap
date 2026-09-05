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
| 進行中 | A0・fold 1・repeat 1、800 epoch（2026-09-06にユーザーが開始） |
| 本番結果 | 現runの正常終了報告待ち。性能値はまだ確認していない |

進行中の学習に対して入力、manifest、config、実装を変更しない。ドキュメント整理は学習source hashの
対象外であり、runのコードとデータには触れない。

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

- [ ] A0・fold 1・repeat 1の学習を正常終了させる。
- [ ] `completion.json` で `training_completed`、800 epoch、249,600 attempted updatesを確認する。
- [ ] `optimizer_updates + amp_skips == attempted_updates`、weights hash、実時間を記録する。
- [ ] 同runのclean test mapを作成し、`cluster_representations.py check` を通す。
- [ ] 同runの全test評価を作成し、`evaluate_representations.py check` を通す。
- [ ] 実測した学習時間、GPU peak、保存量を[検証履歴](docs/verification_history.md)へ追記する。

正常終了後のコマンドは次のとおり。学習が中断した場合は、先に
[runbookの再開手順](docs/experiment_runbook.md#中断からの再開)を使う。

```powershell
uv run python scripts/experiments/cluster_representations.py run --condition A0 --fold 1 --repeat 1 --experiment-dir outputs/experiments/production_v1
uv run python scripts/experiments/cluster_representations.py check --condition A0 --fold 1 --repeat 1 --experiment-dir outputs/experiments/production_v1
uv run python scripts/experiments/evaluate_representations.py run --conditions A0 --fold 1 --repeats 1 --experiment-dir outputs/experiments/production_v1
uv run python scripts/experiments/evaluate_representations.py check --conditions A0 --fold 1 --repeats 1 --experiment-dir outputs/experiments/production_v1
```

## 本番CV

### Baseline

- [ ] B1 PCAを5 foldsのtrain画素でfit・checkする。
- [ ] B0を5 folds × 3 repeatsでclustering・評価する（15組合せ）。
- [ ] B1を5 folds × 3 repeatsでclustering・評価する（15組合せ）。
- [ ] PCAの実solverとrepeat間再利用可否を各foldのfit記録から保存する。

### 主ニューラル条件

各runは800 epochとし、正常完了した重みだけをclusteringへ渡す。

- [ ] A0: 5 folds × 3 repeats（15 runs。fold 1・repeat 1を実行中）
- [ ] M00: 5 folds × 3 repeats（15 runs）
- [ ] M10: 5 folds × 3 repeats（15 runs）
- [ ] M01: 5 folds × 3 repeats（15 runs）
- [ ] M11: 5 folds × 3 repeats（15 runs）
- [ ] 全75 runsでclean test mapのrun・checkを完了する。
- [ ] 全75 runsで評価のrun・checkを完了する。

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
- wall time、GPU peak allocated/reserved、保存量
- checkpointから再開した場合のsource pathと整合確認
- clustering・評価のcompletionとcheck結果

重みとcheckpointはGit管理しない。config、manifest、数値結果、図、completion記録は
`outputs/experiments/production_v1/` に残す。
