# 本番実験runbook

## 1. 役割

この文書は、固定済みの研究設計を `production_v1` で実行する手順をまとめる。
研究条件の定義は[design/README.md](design/README.md)以下を正とし、この文書のコマンドを使って
条件、seed、split、評価方法を変更しない。現在の進捗は[../ToDo.md](../ToDo.md)、実行済みの
工学的確認は[verification_history.md](verification_history.md)を参照する。

すべてのコマンドはリポジトリrootからPowerShellで実行し、Python環境には `uv` を使用する。
各CLIの終了後に `$LASTEXITCODE -eq 0` を確認し、非0なら後続工程へ進まない。JSONのstatus確認は
終了codeの確認に加えて行う。

## 2. 固定パス

| 用途 | パス |
| --- | --- |
| 本番入力 | `data/processed/production_v1/` |
| 前処理確認図 | `outputs/preprocessing/production_v1/` |
| preflight | `outputs/experiments/preflight_v1/` |
| 本番実験 | `outputs/experiments/production_v1/` |
| metadata | `data/metadata/古材メタデータ.csv` |

smokeやpreflightの成果物を本番rootへコピーしない。本番開始後はmanifestを作り直さず、
`outputs/experiments/production_v1/manifests/` を同じ実験系列の固定入力として扱う。

## 3. 本番manifest

本番開始時に限り、manifestを新規作成して検証する。2026-09-06の本番開始では次の処理を行い、
preflightと本番の `complete.json` が同じSHA-256であることを確認してから学習を開始した。

```powershell
uv run python scripts/experiments/prepare_manifests.py create --experiment-id production_v1
uv run python scripts/experiments/prepare_manifests.py check --experiment-id production_v1

$preflightManifestHash = (Get-FileHash 'outputs/experiments/preflight_v1/manifests/complete.json' -Algorithm SHA256).Hash
$productionManifestHash = (Get-FileHash 'outputs/experiments/production_v1/manifests/complete.json' -Algorithm SHA256).Hash
if ($preflightManifestHash -ne $productionManifestHash) {
    throw 'preflightと本番manifestが一致しません'
}
```

以後は `create` を再実行しない。既存manifestの確認には `check` だけを使用する。

```powershell
uv run python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
```

## 4. ニューラルネットの1 run

PowerShell変数には値を直接代入する。 `Read-Host '値'` は値の代入ではなく入力promptの表示なので、
run IDや条件の固定には使わない。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'A0'
$fold = 1
$repeat = 1

uv run python scripts/experiments/train_neural.py train `
    --condition $condition `
    --fold $fold `
    --repeat $repeat `
    --experiment-dir $experimentDir
```

ニューラル条件は `A0`、`M00`、`M10`、`M01`、`M11`、`M11-25`、`M11-75` である。
各runは800 epochで、fold 1–4は249,600回、fold 5は256,000回のbatch試行を予定する。

正常終了後は次の `completion.json` を確認する。

```powershell
$completionPath = Join-Path $experimentDir "results/neural/$condition/fold_$fold/repeat_$repeat/completion.json"
$completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
$completion |
    Select-Object status, completed_epochs, attempted_updates, optimizer_updates, nonzero_lr_updates, amp_skips, training_seconds

if (-not (Test-Path -LiteralPath $completion.weights_file)) {
    throw 'weights fileが存在しません'
}
$actualWeightsHash = (Get-FileHash -LiteralPath $completion.weights_file -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualWeightsHash -ne $completion.weights_sha256) {
    throw 'weights hashがcompletion記録と一致しません'
}
```

完了条件は次のとおり。

- `status` が `training_completed`
- `completed_epochs` が800
- `attempted_updates` が対象foldの予定数と一致
- `optimizer_updates + amp_skips == attempted_updates`
- `weights_file` が存在し、実ファイルのSHA-256が `weights_sha256` と一致

`amp_skips` は実測値として保存し、0と仮定しない。正常完了したrunへ同じ `train` を再実行しない。
`training_seconds` は各epochの処理時間の合計で、CLI全体のwall timeではない。中断再開したrunでも、
完了した800 epochを1回ずつ合計した値になる。

### 中断からの再開

中断または失敗した同じrunだけを、明示した `last.pt` から再開する。manifest、condition、fold、repeat、
configを変えない。

```powershell
$resumePath = Join-Path $experimentDir "checkpoints/neural/$condition/fold_$fold/repeat_$repeat/checkpoints/last.pt"

uv run python scripts/experiments/train_neural.py train `
    --condition $condition `
    --fold $fold `
    --repeat $repeat `
    --experiment-dir $experimentDir `
    --resume $resumePath
```

checkpointがない、またはsource hash・config・run identityが一致しない場合は、別runとして扱う前に
原因を確認する。一致検証を回避して継続しない。

### Windowsでcheckpoint記録の置換に失敗した場合

epoch末に `checkpoint.json.tmp` から `checkpoint.json` への置換だけが `PermissionError` になった場合、
既存ファイルを削除、移動、上書きしない。まず最新の `last.pt` と一時記録、training historyが同じ
完了epochを表していることをread-onlyで確認する。

```powershell
$resultDir = Join-Path $experimentDir "results/neural/$condition/fold_$fold/repeat_$repeat"
$resumePath = Join-Path $experimentDir "checkpoints/neural/$condition/fold_$fold/repeat_$repeat/checkpoints/last.pt"
$pendingPath = Join-Path $resultDir 'checkpoint.json.tmp'

$pending = Get-Content -LiteralPath $pendingPath -Raw | ConvertFrom-Json
$history = @(Get-Content -LiteralPath (Join-Path $resultDir 'training_history.json') -Raw | ConvertFrom-Json)
$actualCheckpointHash = (Get-FileHash -LiteralPath $resumePath -Algorithm SHA256).Hash.ToLowerInvariant()

if ($actualCheckpointHash -ne $pending.checkpoint_sha256 -or
    $history.Count -ne $pending.completed_epochs -or
    $history[-1].epoch -ne $pending.completed_epochs) {
    throw 'checkpoint、一時記録、training historyが一致しません'
}
```

一致した場合だけ、前節の `--resume $resumePath` で同じrunを再開する。再開loaderによる内部state、
run identity、config、manifest、source hashの検証を省略しない。同じ置換エラーが再発する場合は、
対象JSONを開いているeditorやpreviewを閉じ、原因を確認してから再開する。

## 5. PCA baseline

B0は学習済み変換を必要としない。B1は各foldのtrain画素だけでPCAをfitする。まずrepeat 1をfitし、
保存・再読込と由来を確認する。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$fold = 1

uv run python scripts/experiments/fit_baselines.py fit --fold $fold --repeat 1 --experiment-dir $experimentDir
uv run python scripts/experiments/fit_baselines.py check --fold $fold --repeat 1 --experiment-dir $experimentDir
```

`fit.json` の `pca_reusable_across_repeats` が `true` なら、B1のrepeat 2・3では
`cluster_representations.py run` へ `--pca-repeat 1` を明示する。 `false` ならrepeatごとに
PCAをfitする。KMeansはPCAを再利用する場合も各repeatでfitする。

## 6. clean test map

学習またはbaseline fitが完了したcondition・fold・repeatについて、全事前固定KのKMeansと
clean test mapを作成し、CPUの `check` で保存物を検証する。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'A0'
$fold = 1
$repeat = 1

uv run python scripts/experiments/cluster_representations.py run `
    --condition $condition --fold $fold --repeat $repeat --experiment-dir $experimentDir
uv run python scripts/experiments/cluster_representations.py check `
    --condition $condition --fold $fold --repeat $repeat --experiment-dir $experimentDir
```

B1でrepeat 1のPCAを再利用する例は次のとおり。

```powershell
uv run python scripts/experiments/cluster_representations.py run `
    --condition B1 --fold 1 --repeat 2 --pca-repeat 1 `
    --experiment-dir outputs/experiments/production_v1
```

runでは `status=clean_test_maps_completed` かつ `checks_passed=true`、checkでは
`status=validated_existing_clustering` を確認する。test試料数・画素数・全7Kが揃うことも確認する。

## 7. 評価

clean test mapが揃った組合せを評価する。 `run` はGPUを使用し、`check` は保存済み結果をCPUで
検証する。条件とrepeatはまとめて渡せるが、既に評価済みの組合せを再指定して上書きしない。

```powershell
$experimentDir = 'outputs/experiments/production_v1'

uv run python scripts/experiments/evaluate_representations.py run `
    --conditions A0 --fold 1 --repeats 1 `
    --experiment-dir $experimentDir
uv run python scripts/experiments/evaluate_representations.py check `
    --conditions A0 --fold 1 --repeats 1 `
    --experiment-dir $experimentDir
```

同じfoldの未評価3反復をまとめる例は次のとおり。

```powershell
uv run python scripts/experiments/evaluate_representations.py run `
    --conditions B0 B1 M00 M10 M01 M11 --fold 1 --repeats 1 2 3 `
    --experiment-dir outputs/experiments/production_v1
```

runでは各組合せの `status=full_test_evaluation_completed` と `checks_passed=true`、checkでは
`status=validated_existing_evaluation` を確認する。未定義指標は理由付きの `null` として扱い、
0へ置換しない。

## 8. 実行matrix

| 区分 | 条件 | 必要な組合せ |
| --- | --- | ---: |
| baseline | B0、B1 | 各5 folds × 3 repeats |
| 主実験の学習 | A0、M00、M10、M01、M11 | 75 runs |
| mask率補助学習 | M11-25、M11-75 | 30 runs |

各組合せについてclean mapと評価を完了する。ニューラル学習は合計105 runsで、B0・B1のfitや
KMeans、評価処理はこの数に含めない。3反復はseed選別に使わず、すべてOOF集計へ含める。

## 9. OOF集計

指定する全conditionについて5 folds × 3 repeatsの評価が揃ってから実行する。snapshot名は一度だけ
使用し、既存snapshotは `check` で読む。

```powershell
uv run python scripts/experiments/aggregate_oof.py run `
    --conditions B0 B1 A0 M00 M10 M01 M11 `
    --snapshot main_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
uv run python scripts/experiments/aggregate_oof.py check `
    --snapshot main_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
```

mask率補助実験は別snapshotにする。

```powershell
uv run python scripts/experiments/aggregate_oof.py run `
    --conditions M11-25 M11 M11-75 `
    --snapshot mask_rate_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
uv run python scripts/experiments/aggregate_oof.py check `
    --snapshot mask_rate_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
```

runでは `status=oof_aggregation_completed` と `checks_passed=true`、checkでは
`status=validated_existing_oof` を確認する。欠損run、失敗run、不完全な試料・画素対応を無視して
集計しない。

## 10. 保存とGit

`outputs/experiments/production_v1/` のconfig、manifest、数値結果、図、completion記録は保存する。
`checkpoints/`、`weights/`、`*.pt`、`*.pth`、`*.safetensors` は `.gitignore` により
Git管理対象外である。重みを削除する場合も、論文・再解析に必要なrunの由来とhashを数値記録に残す。

実行中にコード、設計条件、入力データ、manifestを変更しない。変更が必要になった場合はrunを止め、
影響範囲を確認し、`production_v1` と混合しない新しい実験系列として扱う。
