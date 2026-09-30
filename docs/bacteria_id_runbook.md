# Bacteria-ID補助実験 runbook

> Table 5の分類学習は[HPO runbook](bacteria_id_hpo_runbook.md)を参照する。
> 本書は入力準備と、新しい80/20 Table 4の事前学習・クラスタリングを扱う。

条件の根拠と比較上の留保は[固定計画](design/bacteria_id_experiment_plan.md)を参照する。
新Table 4の実行CLIは[`scripts/experiments/bacteria_id_table4_80.py`](../scripts/experiments/bacteria_id_table4_80.py)。以下はリポジトリrootから
PowerShellで実行する。各コマンドの終了codeが非0なら、後続工程へ進まない。

## 入力と準備

必要な原本は`data/raw/bacteria_id/`の`X_reference.npy`・`y_reference.npy`・
`X_finetune.npy`・`y_finetune.npy`・`X_test.npy`・`y_test.npy`である。
`wavenumbers.npy`は存在すれば軸の形状・単調性・範囲を記録する。
今回取得したファイルのヘッダーは6つの入力・ラベル配列について計画上のshapeと一致し、
波数軸の端点は先頭1792.40、末尾381.98 cm⁻¹で降順だった。処理でチャネル順を反転しない。
ラベル原本はfloat64保存のため、準備時に有限な整数値であることを確認してから整数IDへ変換する。
原本は書き換えない。2026-09-28には[公式案内](https://github.com/csho33/bacteria-ID/blob/3c00a712a6dbad9aefa19ac878a1e6db20590ca9/data/data.md)の
Stanford BoxがHTTP 404だったため、[bacteria-SANetの公開ミラー](https://github.com/DenglinGo/bacteria-SANet)が案内する
Dropboxから取得した。取得アーカイブ、取り出した7ファイル、URLは`data/raw/bacteria_id/acquisition.json`へ保存済み。
公式配布物とのバイト単位の同一性は未確認であるため、論文中で公式Boxから直接取得したと記さない。

小規模のコード確認は次で行う。実データやGPUを使用しない。

```powershell
uv run --no-sync python -m pytest tests/experiments/test_bacteria_table4_80.py
```

既存のSNV・旧分割はすでに`data/processed/bacteria_id_v1/`へ準備済みである。
新Table 4ではSNVを作り直さない。
旧manifestには旧train/validation/testとTable 5の入力契約が残る。

## 新Table 4の分割準備と検証

次のコマンドは旧分割のindexとラベルを読み、旧train∪validationの新trainと、順序を保った旧testを
`outputs/experiments/bacteria_id_table4_80_20_v1/`に保存する。原本・SNV・旧分割は変更しない。
読み取りはindexとラベルが主であり、全スペクトルの再前処理はしない。出力先が既に存在すれば停止する。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py prepare
```

出力された`split_sizes`はBacteria-4がtrain 6,400/test 1,600、Bacteria-6がtrain 9,600/test 2,400、
`planned_runs`が20であることを確認する。各クラスはtrain 1,600/test 400。
準備コードは旧splitの非重複と対象全件の網羅、新train/testの非重複・網羅、旧test配列との完全一致を検証する。
新`manifest.json`に旧入力manifest・旧splitのhash、分割件数、旧test index hash、20 runと用途別seedを保存する。

## 20 runの新規事前学習・クラスタリング

次の一括コマンドはBacteria-4／6 × M00／M11 × seed 0–4を順に実行する。
各runは新trainだけで初期化から800 epoch・batch size 1024、既存AdamW・学習率schedule・
`drop_last=True`を使用する。Bacteria-4は6 batch/epoch・予定4,800更新、Bacteria-6は
9 batch/epoch・予定7,200更新。AMP skip後の実更新回数は各runの`completion.json`に記録する。
単一CUDA GPUを長時間占有し、checkpointと20件の結果を新出力先に生成する。
Table 5の処理も同じGPUを使う場合は、ユーザー側で実行順序を調整する。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py all --device 0
```

`all`は新出力先の同一runの完了記録を検証してスキップし、中断runはそのrunの`last.pt`からのみ再開する。
先頭runのクラスタリングでscore metadata生成前に失敗した場合は、空のresults directoryと
`centroids.npz`だけを含む未完成出力を`all`が除去し、そのrunのクラスタリングだけを再実行する。
完了済みの800 epoch事前学習は再実行しない。この評価処理修正前に終了した
`bacteria4/M00/seed_0`の旧ソースhashは保持し、集計CSV/JSONでrun別に追跡する。
旧60%学習・reference30・Table 5のcheckpointは読み込まない。新規開始時に出力先が存在するrunは上書きしない。
20件の評価が揃うと、Table 4主表・seed別CSV・統計JSON・seed別図を作る。

個別に実行するときは、例えば次の順序を使う。中断後の再開には同じ新runのcheckpointを明示する。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py pretrain --corpus bacteria4 --condition M00 --seed 0 --device 0
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py cluster --corpus bacteria4 --condition M00 --seed 0 --device 0
```

中断した事前学習runの再開例：

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py pretrain --corpus bacteria4 --condition M00 --seed 0 --device 0 --resume outputs/experiments/bacteria_id_table4_80_20_v1/checkpoints/pretrain/bacteria4/M00/seed_0/checkpoints/last.pt
```

個別実行で20件を揃えた場合の集計コマンドは次のとおり。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py report
```

主結果は新出力先の`table4.md`とし、`table4_seed_scores.csv`の20行全件と
`table4_statistics.json`の平均・標本SD（`ddof=1`）を照合する。
`table4_seed_scores.png`は各seed値と平均±標本SDを示す。
旧60/20/20 Table 4は`outputs/experiments/bacteria_id_table4_60_20_20_legacy_v1/`に
過去実験として保持する。新80/20 Table 4の20件と主表は完了した。
Table 5の事前学習・HPO・分類は[専用記録](bacteria_id_hpo_runbook.md)に従う。
成果物の移動と照合は[整理記録](bacteria_id_artifact_reorganization_plan.md)を参照する。
