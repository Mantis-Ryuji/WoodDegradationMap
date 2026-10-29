# Bacteria-ID Table 4の実行手順

本書は入力準備と、80/20分割によるTable 4の事前学習・クラスタリングを扱う。
Table 5の事前学習・分類は[専用runbook](bacteria_id_hpo_runbook.md)、
条件の定義と比較上の制約は[実験仕様](design/bacteria_id_experiment_plan.md)を参照する。
以下はリポジトリrootから、構築済み環境でPowerShellを使って実行する。
各コマンドの終了codeが非0なら後続工程へ進まない。

## 入力と準備

必要な原本は`data/raw/bacteria_id/`の`X_reference.npy`・`y_reference.npy`・
`X_finetune.npy`・`y_finetune.npy`・`X_test.npy`・`y_test.npy`である。
`wavenumbers.npy`があれば、軸の形状・単調性・範囲を記録する。
採用入力の波数軸は1792.40から381.98 cm⁻¹の降順であり、チャネル順は反転しない。
ラベル原本のfloat64値は有限な整数であることを検証してから整数IDへ変換する。

入力は[bacteria-SANetの公開ミラー案内](https://github.com/DenglinGo/bacteria-SANet)経由で取得した。
取得URL、アーカイブ、取り出したファイルの記録は`data/raw/bacteria_id/acquisition.json`にある。
公式配布物とのバイト単位の同一性は未確認である。原本は書き換えない。

未作成の出力先にSNV入力と分割を準備する。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id.py prepare
```

準備先は`data/processed/bacteria_id_v1/`である。既存入力は再生成せず、そのmanifestと
`splits.npz`をTable 4の分割準備へ渡す。分割の定義と由来は
[実験仕様](design/bacteria_id_experiment_plan.md#table4-split)を参照する。

## Table 4の分割準備

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py prepare
```

保存済みのtrain・validation indexを結合し、test indexとその順序を保持して、
`outputs/experiments/bacteria_id_table4_80_20_v1/`に80/20分割を作る。
原本・SNV・入力分割は変更しない。読み取りはindexとラベルが主で、全スペクトルの再前処理はしない。
出力先が既に存在する場合は停止する。

`split_sizes`はBacteria-4がtrain 6,400/test 1,600、Bacteria-6がtrain 9,600/test 2,400、
各クラスtrain 1,600/test 400であり、`planned_runs`が20であることを確認する。
準備コードは分割の非重複・対象全件の網羅・test indexの完全一致を検証する。
`manifest.json`には入力manifest・分割のhash、件数、run identity、用途別seedを保存する。

## 事前学習・クラスタリング

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py all --device 0
```

Bacteria-4/6 × M00/M11 × seed 0–4の20 runを単一CUDA GPUで順次実行する。
各runはtrainだけで初期化から800 epoch・batch size 1024で学習する。
`drop_last=True`によりBacteria-4は6 batch/epoch・4,800更新試行、
Bacteria-6は9 batch/epoch・7,200更新試行となる。AMP skip後の実更新数は`completion.json`を参照する。
長時間のGPU処理であり、checkpointと評価結果を出力先へ保存する。

`all`は完了済みrunを検証してスキップし、中断runは同じrunの`last.pt`から再開する。
既知の未完成クラスタ出力（空のresults directoryと`centroids.npz`のみ）は除去して
クラスタリングを再実行するが、他の不完全出力は停止して原因の確認を求める。
Table 5のcheckpointは読み込まない。20件の評価が揃うと主表・seed別CSV・統計JSON・seed別図を作る。

個別実行の例：

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py pretrain --corpus bacteria4 --condition M00 --seed 0 --device 0
if ($LASTEXITCODE -ne 0) { throw 'Table 4 pretraining failed' }
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py cluster --corpus bacteria4 --condition M00 --seed 0 --device 0
```

中断した同じ事前学習runの再開例：

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py pretrain --corpus bacteria4 --condition M00 --seed 0 --device 0 --resume outputs/experiments/bacteria_id_table4_80_20_v1/checkpoints/pretrain/bacteria4/M00/seed_0/checkpoints/last.pt
```

## 集計と確認

個別実行で20件を揃えた場合、または保存済み評価から表を作る場合は次を使う。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py report
```

出力先の`table4.md`を、`table4_seed_scores.csv`の20行全件と
`table4_statistics.json`の平均・標本SD（`ddof=1`）に照合する。
`table4_seed_scores.png`は各seed値と平均±標本SDを示す。
`outputs/`とcheckpointはGit管理対象外である。

入力・分割・評価処理を変更した場合の小規模テストは次のとおり。実データやGPUは使用しない。

```powershell
uv run --no-sync python -m pytest tests/experiments/test_bacteria_table4_80.py
```
