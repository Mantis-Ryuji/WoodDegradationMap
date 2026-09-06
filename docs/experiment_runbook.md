# 本番実験runbook

## 1. 役割

この文書は、固定済みの研究設計を `production_v1` で実行する手順をまとめる。
研究条件の定義は[design/README.md](design/README.md)以下を正とし、この文書のコマンドを使って
条件、seed、split、評価方法を変更しない。現在の進捗は[../ToDo.md](../ToDo.md)、実行済みの
工学的確認は[verification_history.md](verification_history.md)を参照する。

[vMF補助実験](design/experiment_protocol.md#vmf-supplementary)は、
数値検証・設定確定とpipeline実装が必要であり、CLIはまだない。本書のclustering・評価・OOFコマンドは
Cosine-KMeansの主実験・mask率補助実験用である。vMFの実施順序は第8.1節を参照する。

実行環境はChemoMAE v0.2.2とする。

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

本番開始時、出力先が存在しない場合に限りmanifestを新規作成して検証する。
既存の`production_v1`では次の`create`を実行せず、保存済みmanifestの`check`を使う。

```powershell
uv run python scripts/experiments/prepare_manifests.py create --experiment-id production_v1
if ($LASTEXITCODE -ne 0) { throw 'manifest creation failed' }
uv run python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
if ($LASTEXITCODE -ne 0) { throw 'manifest check failed' }
```

以後は `create` を再実行しない。既存manifestの確認には `check` だけを使用する。

```powershell
uv run python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
```

## 4. ニューラルネットの1 run

対象はToDoの未完了runから選び、PowerShell変数へ直接代入する。以下はM00・fold 1・repeat 2の例である。
完了済みrunは再学習せず、保存済み成果物の確認には各工程の`check`を使う。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M00'
$fold = 1
$repeat = 2

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
$condition = 'M00'
$fold = 1
$repeat = 2

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
    --conditions M00 --fold 1 --repeats 2 `
    --experiment-dir $experimentDir
uv run python scripts/experiments/evaluate_representations.py check `
    --conditions M00 --fold 1 --repeats 2 `
    --experiment-dir $experimentDir
```

同じfoldの3反復をまとめる場合は、指定する全組合せのclean test mapが揃い、すべて未評価であることを確認する。

```powershell
uv run python scripts/experiments/evaluate_representations.py run `
    --conditions B0 B1 M10 M01 M11 --fold 1 --repeats 1 2 3 `
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

### 8.1 vMF補助実験の準備と実施

1. 実験プロトコル第5.2.3節の数値仕様を確定し、v0.2.2の修正内容と小規模CPU・GPU動作を検証する。
2. 元の成果物の検証、独立した保存先、fit・評価・check・OOFを実装する。具体的なCLIは実装時に追記する。
3. 本番CV後、同じ表現・train画素・Kを使って735 fitsを行い、同じtest全画素・共通摂動で評価する。
4. 完了・失敗・未定義値を保持し、全組合せの完全性を確認して独立にOOF集計・報告する。

ニューラル学習とPCA fitは追加しない。研究条件は[実験プロトコル第5.2節](design/experiment_protocol.md#vmf-supplementary)、
指標と比較の定義は[評価指標第8.4節](design/evaluation_metrics.md#vmf-evaluation)に従う。
vMF用の設定・結果・完了記録は主実験から分け、元の成果物との対応とsource hashを保存する。

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

本番CVではコード、設計条件、入力データ、manifestを固定する。
変更が必要な場合は影響範囲と実験系列の扱いを事前に確認し、変更内容と検証結果を記録する。
