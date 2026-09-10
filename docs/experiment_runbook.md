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
本番実行例は`uv run --no-sync`とし、環境構築・更新をrun開始時の処理から分ける。

- [入力確認](#input-preparation)・[manifest](#3-本番manifest)
- [NNの1 runと再開](#neural-run)・[PCA](#5-b1-pca-fitとbaseline変換の検証)
- [clean test map](#6-clean-test-map)・[評価](#7-評価)
- [OOF集計](#oof-aggregation)・[B0/B1 sanity図](#oof-sanity)
- [未実装の全体fit](#global-fit-pipeline)・[保存規約](#artifact-records)

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
| B0・B1 OOF sanity | `outputs/sanity_checks/b0_b1_oof_visualization/` |

smokeやpreflightの成果物を本番rootへコピーしない。本番開始後はmanifestを作り直さず、
`outputs/experiments/production_v1/manifests/` を同じ実験系列の固定入力として扱う。

<a id="input-preparation"></a>

### 2.1 前処理と入力確認

本番前処理は生成済みである。再生成が必要な別runで、両出力先が新規または空の場合に限り、
[前処理の固定仕様](design/preprocessing.md)に従って実行する。

```powershell
uv run --no-sync python scripts/preprocess/run_production_preprocessing.py
```

既存の品質表から、各試料で8192画素の非復元抽出が可能か確認する場合は次を使う。
スペクトルやHDF5は読み込まず、データ・manifestを変更しない。

```powershell
uv run --no-sync python scripts/preprocess/check_sampling_pixels.py --q 8192
```

試料別の保存有効画素数、抽出率、不足数と全体の最小・中央値・最大を表示する。
全試料で抽出可能なら終了code 0、不足があれば1。実際の抽出とsplitの作成は次節で行う。

## 3. 本番manifest

本番開始時、出力先が存在しない場合に限りmanifestを新規作成して検証する。
既存の`production_v1`では次の`create`を実行せず、保存済みmanifestの`check`を使う。

```powershell
uv run --no-sync python scripts/experiments/prepare_manifests.py create --experiment-id production_v1
if ($LASTEXITCODE -ne 0) { throw 'manifest creation failed' }
uv run --no-sync python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
if ($LASTEXITCODE -ne 0) { throw 'manifest check failed' }
```

以後は `create` を再実行しない。既存manifestの確認には `check` だけを使用する。

```powershell
uv run --no-sync python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
```

<a id="neural-run"></a>

## 4. ニューラルネットの1 run

対象はToDoの未完了runから選び、PowerShell変数へ直接代入する。以下はA0・fold 2・repeat 1の例である。
完了済みrunは再学習せず、保存済み成果物の確認には各工程の`check`を使う。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'A0'
$fold = 2
$repeat = 1

uv run --no-sync python scripts/experiments/train_neural.py train `
    --condition $condition `
    --fold $fold `
    --repeat $repeat `
    --experiment-dir $experimentDir
```

同一foldの未着手repeatを連続実行する場合も、CLIは1 runずつ呼び出す。
以下はA0・fold 2・repeat 1–3を並列化せずに順次実行する例である。
実行時は`$repeats`に未着手runだけを列挙し、完了済みrunや中断したrunを含めない。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'A0'
$fold = 2
$repeats = 1..3

foreach ($repeat in $repeats) {
    uv run --no-sync python scripts/experiments/train_neural.py train `
        --condition $condition `
        --fold $fold `
        --repeat $repeat `
        --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) {
        throw "$condition fold $fold repeat $repeat training failed"
    }
}
```

非0終了でloopは停止する。中断時は完了済みrepeatを再実行せず、
対象repeatだけを次節の手順で明示的に再開する。

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

uv run --no-sync python scripts/experiments/train_neural.py train `
    --condition $condition `
    --fold $fold `
    --repeat $repeat `
    --experiment-dir $experimentDir `
    --resume $resumePath
```

checkpointがない、またはsource hash・config・run identityが一致しない場合は、別runとして扱う前に
原因を確認する。一致検証を回避して継続しない。
固定configを変更したrunは旧checkpointから再開しない。明示的に無効と判断した未完了runの
resultsとcheckpointsだけを対象パスの照合後に除き、新規runとして開始する。

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

## 5. B1 PCA fitとbaseline変換の検証

B0は学習済み変換を必要とせず、fitするパラメータを持たない。B1はfoldごとにtrain集合が異なるため、
各foldのtrain画素だけでPCAを1回ずつ、5 foldsで計5回fitする。同じfold内では決定的なPCA変換を
repeat 1～3で共有し、PCAを15回fitしない。KMeansはPCAを共有してもrepeatごとにfitする。

CLI名の`fit_baselines.py fit`と保存先`results/baselines/`は、B0・B1をまとめて検証する工程を表す。
本書でいう「baseline fit」の実質はB1 PCA fitであり、B0について行うのは無パラメータ変換の仕様保存と
probe検証だけである。完了済みのfoldで`fit`を再実行しない。

```powershell
$experimentDir = 'outputs/experiments/production_v1'

foreach ($fold in 1..5) {
    uv run --no-sync python scripts/experiments/fit_baselines.py fit `
        --fold $fold --repeat 1 --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) { throw "B1 PCA fit failed: fold $fold" }

    uv run --no-sync python scripts/experiments/fit_baselines.py check `
        --fold $fold --repeat 1 --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) { throw "baseline check failed: fold $fold" }
}
```

各foldで`fit`は`status=fitted_and_roundtrip_checked`、`pca_reusable_across_repeats=true`、
`check`は`status=validated_existing_baselines`を確認する。production_v1では5 foldsすべてで
repeat間再利用可否が`true`だったため、B1のrepeat 2・3では`cluster_representations.py run`へ
`--pca-repeat 1`を明示する。

`results/baselines/fold_<fold>/repeat_1/b0.json`はB0変換仕様、同じ場所の`fit.json`は主にB1 PCAの
fit由来とB0・B1のprobe診断である。実際のPCAパラメータは
`checkpoints/baselines/fold_<fold>/repeat_1/pca.npz`に保存する。

## 6. clean test map

ニューラル学習、またはB1で必要なfold別PCA fitが完了したcondition・fold・repeatについて、
全事前固定KのKMeansとclean test mapを作成し、CPUの`check`で保存物を検証する。
B0には前段のfitはない。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11'
$fold = 2
$repeat = 1

uv run --no-sync python scripts/experiments/cluster_representations.py run `
    --condition $condition --fold $fold --repeat $repeat --experiment-dir $experimentDir
if ($LASTEXITCODE -ne 0) { throw 'clustering failed' }
uv run --no-sync python scripts/experiments/cluster_representations.py check `
    --condition $condition --fold $fold --repeat $repeat --experiment-dir $experimentDir
```

B1でrepeat 1のPCAを再利用する例は次のとおり。

```powershell
uv run --no-sync python scripts/experiments/cluster_representations.py run `
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
$condition = 'M11'
$fold = 2
$repeat = 1

uv run --no-sync python scripts/experiments/evaluate_representations.py run `
    --conditions $condition --fold $fold --repeats $repeat `
    --experiment-dir $experimentDir
if ($LASTEXITCODE -ne 0) { throw 'evaluation failed' }
uv run --no-sync python scripts/experiments/evaluate_representations.py check `
    --conditions $condition --fold $fold --repeats $repeat `
    --experiment-dir $experimentDir
```

複数条件・反復をまとめる場合は`--conditions`と`--repeats`へ空白区切りで列挙する。
指定した直積の全組合せでclean test mapが揃い、すべて未評価であることを事前に確認する。

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

<a id="global-fit-pipeline"></a>

### 8.2 全体fitと解釈（未実装）

全体解釈はB0・B1・A0・M00・M11の5条件を対象とする。全49試料の共通抽出画素でPCAをfitし、
A0・M00・M11を各1回、計3回学習する。この3学習はCVの105学習とは別に行う。
得られた各条件の同じ表現に、$K_0=8$でCosine-KMeansとvMFを各1回fitする。
vMFはCV補助実験と同じ数値仕様の確定・検証後に実施し、全体学習用のPCA・encoder・共通抽出座標を再利用する。
全体解釈用のvMF 5 fitsは、CV補助実験の735 fitsとは別枠で管理する。
fitと表示の規約は[全体可視化設計](design/visualization_and_interpretation.md)に従い、成果物をCVのOOF集計へ含めない。
全体解釈pipelineと具体的なCLIは未実装であり、実装時に実行手順を追記する。
既存の`train_neural.py`は`--fold`を必須とするCV用CLIであり、そのまま全体学習には使用できない。

<a id="oof-aggregation"></a>

## 9. OOF集計

指定する全conditionについて5 folds × 3 repeatsの評価が揃ってから実行する。snapshot名は一度だけ
使用し、既存snapshotは `check` で読む。

```powershell
uv run --no-sync python scripts/experiments/aggregate_oof.py run `
    --conditions B0 B1 A0 M00 M10 M01 M11 `
    --snapshot main_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'OOF aggregation failed' }
uv run --no-sync python scripts/experiments/aggregate_oof.py check `
    --snapshot main_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
```

mask率補助実験は別snapshotにする。

```powershell
uv run --no-sync python scripts/experiments/aggregate_oof.py run `
    --conditions M11-25 M11 M11-75 `
    --snapshot mask_rate_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'OOF aggregation failed' }
uv run --no-sync python scripts/experiments/aggregate_oof.py check `
    --snapshot mask_rate_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
```

runでは `status=oof_aggregation_completed` と `checks_passed=true`、checkでは
`status=validated_existing_oof` を確認する。欠損run、失敗run、不完全な試料・画素対応を無視して
集計しない。

<a id="oof-sanity"></a>

### 9.1 B0・B1 OOF sanity可視化

B0・B1の全5 folds×3反復のclustering・評価が完了した成果物をCPUで読み、
[専用の表示仕様](design/oof_sanity_visualization.md)に従ってPNG 3枚とCSV 3つを生成する。
全主条件のOOF snapshot作成や、モデルの再fitは不要である。

```powershell
# 既定の出力先が存在しない場合のみ実行できる
uv run --no-sync python scripts/experiments/visualize_b0_b1_oof.py
```

既定の出力先は生成済みなので、再生成では`--output-dir`へ新規パスを指定する。
以下のプレースホルダーを、まだ存在しない出力先に置き換える。

```powershell
uv run --no-sync python scripts/experiments/visualize_b0_b1_oof.py `
    --experiment-dir outputs/experiments/production_v1 `
    --output-dir '<新規出力先>'
```

終了code 0と保存物を確認する。B0/B1別の代表7試料図でKYOw名が各試料の下にあり、
silhouetteに下段subplotがないことを確認する。LLA・LFR・occupancyと未定義理由はCSVで読む。
sanity出力にはログやcompletion JSONを追加しない。

<a id="artifact-records"></a>

## 10. 保存とGit

`outputs/experiments/production_v1/` のconfig、manifest、数値結果、図、completion記録は保存する。
`checkpoints/`、`weights/`、`*.pt`、`*.pth`、`*.safetensors` は `.gitignore` により
Git管理対象外である。重みを削除する場合も、論文・再解析に必要なrunの由来とhashを数値記録に残す。

各runで保存する記録は次のとおり。schemaと由来の扱いは[実験プロトコル第11.2節](design/experiment_protocol.md#execution-records)に従う。

- condition、fold、repeat、seed、manifest・code・config hash
- status、epoch、attempted/optimizer updates、AMP skips
- 学習のepoch時間合計、clustering・評価のwall timeとGPU peak allocated/reserved、保存量
- 再開した場合のsource pathと整合確認
- clustering・評価のcompletionとcheck結果

本番学習completionはGPU peakを保存しない。学習時の値を事後推定せず、preflightの実測値は
工学的参考値として区別する。本番CV中にこの不足を補う記録コードの変更は行わない。
上記は本番実験の記録規約であり、PNG・CSVだけを保存するOOF sanityとは分ける。

本番CVではコード、設計条件、入力データ、manifestを固定する。
変更が必要な場合は影響範囲と実験系列の扱いを事前に確認し、変更内容と検証結果を記録する。
