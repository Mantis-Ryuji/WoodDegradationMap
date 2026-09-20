# 本番実験runbook

## 1. 役割

固定設計を`production_v1`で実行する手順を示す。条件・seed・split・評価方法の定義は
[研究設計](design/README.md)、現在の進捗は[ToDo](../ToDo.md)を参照する。

[vMF補助実験](design/experiment_protocol.md#vmf-supplementary)は、
数値検証・設定確定とpipeline実装が必要であり、CLIはまだない。本書のclustering・評価・OOFコマンドは
Cosine-KMeansの主実験・mask率補助実験用である。vMFの実施順序は第8.1節を参照する。

ChemoMAE v0.2.2の環境で、リポジトリrootからPowerShellで実行する。
本番実行例は`uv run --no-sync`とし、環境構築・更新をrun開始時の処理から分ける。

- [入力確認](#input-preparation)・[manifest](#3-本番manifest)
- [NNの1 runと再開](#neural-run)・[PCA](#5-b1-pca-fitとbaseline変換の検証)
- [clean test map](#6-clean-test-map)・[評価](#7-評価)
- [OOF集計](#oof-aggregation)・[OOF sanity図](#oof-sanity)・[主条件OOF図表](#oof-reporting)
- [全体fitの準備・一括学習](#global-fit-pipeline)・[保存規約](#artifact-records)

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
| B0・B1・A0・M00 OOF sanity | `outputs/sanity_checks/a0_m00_oof_visualization/` |
| 主条件OOF snapshot | `outputs/experiments/production_v1/results/oof/main_oof_v1/` |
| 主条件OOF図表 | `outputs/experiments/production_v1/results/figures/main_oof_v1/` |

smokeやpreflightの成果物を本番rootへコピーしない。本番開始後はmanifestを作り直さず、
`outputs/experiments/production_v1/manifests/` を同じ実験系列の固定入力として扱う。

<a id="input-preparation"></a>

### 2.1 前処理と入力確認

本番前処理は生成済みである。再生成が必要な別runで、両出力先が新規または空の場合に限り、
[前処理の固定仕様](design/preprocessing.md)に従って実行する。

```powershell
uv run --no-sync python scripts/preprocess/run_production_preprocessing.py
```

既存結果の確認図だけを更新する場合は、次を実行する。前者は保存済みの品質表から前処理のPNG 4枚を
上書きし、`reflectance_l2_norm/`は読み書きしない。後者は既存のaugmentation出力にある例図2枚と
`summary.json`の可視化記録を更新し、noise・shiftの箱ひげ図PNGを取り除く。
後者は保存済み`selection.csv`が指定するtrain画素（8試料×128画素）だけをHDF5から読み、
各試料の波長ごとの中央値にRMS距離が最も近い実測画素を1つずつ選ぶ。その8候補から、
画素間の最小RMS距離が最大となる3試料の組を選ぶ。同点は距離の和、試料順で決める。
noise・shiftは同じ3画素・例順を共有し、描画するclean値には平滑化・平均化・再標準化を行わない。
選定方法と試料ID・HDF5行・画素座標を`summary.json`の可視化記録に残す。
前処理データ、train画素の`selection.csv`・`metrics.csv`と数値要約は再生成しない。

```powershell
.venv\Scripts\python.exe scripts/preprocess/redraw_production_reports.py
.venv\Scripts\python.exe scripts/experiments/sanity_check_augmentation_strengths.py --plots-only
```

終了code 0と、[共通の縦軸規約](design/visualization_and_interpretation.md#figure-style)を確認する。
CutoffはSNR proxyだけ、候補画素図は個別のSNV panelであることを確認する。
Shift例はExample 1（＋）、Example 2（−）、Example 3（＋）の3行×2列とし、noise図と同じサイズであることを確認する。

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

既存manifestの確認:

```powershell
uv run --no-sync python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
```

<a id="neural-run"></a>

## 4. ニューラルネットの1 run

対象はToDoの未完了runから選び、PowerShell変数へ直接代入する。以下はmask率補助条件M11-25・fold 1・repeat 1の例である。
完了済みrunは再学習せず、保存済み成果物の確認には各工程の`check`を使う。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11-25'
$fold = 1
$repeat = 1

uv run --no-sync python scripts/experiments/train_neural.py train `
    --condition $condition `
    --fold $fold `
    --repeat $repeat `
    --experiment-dir $experimentDir
```

同一foldの未着手repeatを連続実行する場合も、CLIは1 runずつ呼び出す。
以下はM11-25・fold 1・repeat 1–3を並列化せずに順次実行する例である。
実行時は`$repeats`に未着手runだけを列挙し、完了済みrunや中断したrunを含めない。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11-25'
$fold = 1
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

B1 PCAは各foldのtrain画素だけで1回ずつ、計5回fitする。同じfold内のrepeat 1～3でPCAを共有し、
KMeansはrepeatごとにfitする。`fit_baselines.py fit`では、パラメータを持たないB0の変換仕様保存と
probe検証も行う。完了済みfoldで`fit`を再実行しない。

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
`check`は`status=validated_existing_baselines`を確認する。PCAを共有するB1のrepeat 2・3では、
`cluster_representations.py run`へ`--pca-repeat 1`を明示する。

`results/baselines/fold_<fold>/repeat_1/b0.json`はB0変換仕様、同じ場所の`fit.json`は主にB1 PCAの
fit由来とB0・B1のprobe診断である。実際のPCAパラメータは
`checkpoints/baselines/fold_<fold>/repeat_1/pca.npz`に保存する。

## 6. clean test map

ニューラル学習、またはB1で必要なfold別PCA fitが完了したcondition・fold・repeatについて、
全事前固定KのKMeansとclean test mapを作成し、CPUの`check`で保存物を検証する。
B0には前段のfitはない。以下は第4節と同じM11-25・fold 1・repeat 1の例である。

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11-25'
$fold = 1
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
$condition = 'M11-25'
$fold = 1
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

2026-09-20の指定により低優先度とし、主条件のOOF図表と、第8.2節のCosine-KMeansによる
潜在空間・空間map・観測スペクトルの解析・図表・解釈の整理を一通り終えた後に回す。
その後、数値仕様の確定・v0.2.2の検証・共通fit処理実装 → 全体fit用5 fits・手法間比較 →
CV専用pipeline実装 → 735 fits・評価 → 独立OOF集計・比較図表の順に進める。
数値検証・実装も第8.2節の先行解析の前提にはしない。
NN学習・PCA fitは追加せず、既存の表現・train画素・Kと同じtest全画素・共通摂動を使う。
設定・結果・完了記録は主実験から分け、元成果物との対応とsource hash、失敗・未定義値を保持する。

条件は[実験プロトコル](design/experiment_protocol.md#vmf-supplementary)、比較は
[評価規約](design/evaluation_metrics.md#vmf-evaluation)、実装の残作業は[ToDo第5節](../ToDo.md#5-vmf補助実験)を参照する。

<a id="global-fit-pipeline"></a>

### 8.2 全体fitと解釈

主条件のOOF図表と全体fitの準備・NN学習は完了した。手順5〜6の専用CLIも実装し、次は本番のクラスタリングと図表生成を実行する。
mask率sweepとvMF（数値検証・全体fit用5 fitsを含む）より先に、本節の解析・図表・解釈を一通り完了する。

[全体可視化設計](design/visualization_and_interpretation.md)に従い、全49試料の共通抽出画素で
B1 PCAとA0・M00・M11をfitする。B0を加えた5条件の表現で、$K_0=8$のCosine-KMeansを5 fits行う。
vMFの5 fitsは第8.1節の後続解析へ回し、数値仕様の確定・検証後に同じPCA・encoder・抽出座標を再利用する。
全体学習の3 runsと全体クラスタリングは、CVの105学習・vMF 735 fitsとは別枠であり、OOF集計に含めない。

実装の残作業は[ToDo第3節](../ToDo.md#3-全体fitと解釈)を参照する。
`global_fit.py`にmanifest・B0/PCA・A0/M00/M11の一括学習・再開・checkを実装した。
合成データのCPU検証済み。本番manifestの作成・checkに続き、修正後のPCA fit・GPU smoke・
A0/M00/M11の各800 epochが完了した。2026-09-20に保存記録を確認し、`training-check`の完了はユーザー報告による。
後続処理の実装時には合成データでCPU検証し、本番のGPU checkや全データ処理は再実行していない。
既存の`train_neural.py`は`--fold`必須のCV用であり、全体fitには使わない。

実装・実施は次の順序とする。手順1〜4は完了し、CLIは再現・再開用の参照として下記に残す。
手順5〜6は実装・CPU小規模検証済み、本番実行は未完了。手順7は設定・帯域の確定後に実装する。
実行コマンドとその後の順序は[学習完了後の作業](#global-post-fit)を参照する。

1. **全体runの実行契約を固定する（確定・実装済み）。** ROOT_SEED=20260905・SHA-256方式を維持し、
   fold位置を`global`、反復IDを1とする。抽出・PCA・NNと後続の$K_0=8$クラスタリング用のseed計55個を保存する。
   保存rootは`outputs/experiments/global_v1/`。2026-09-19のユーザー確認による。
2. **共通入力と専用pipelineを実装・小規模検証する。** 49試料から各8,192画素、計401,408画素を抽出し、
   全5条件・両手法で座標を共有する。fit画素と推論対象の全有効画素を区別し、CPU小規模と必要最小限のGPU検証で、
   入出力・seed・保存復元・完了判定を確認する。CVのmanifestと成果物は全体fitから分離する。
3. **B0・B1を準備する。** B0は固定のSNV変換、B1は共通fit画素でPCAを1回fitする。
   全体fit用の入力・表現抽出・保存復元を確認してから長時間のNN学習へ進む。
4. **A0・M00・M11を各800 epochで1回ずつ学習する。** CVと同じ条件別recipeを使用し、
   最終重み・実際の更新回数・実行環境・seed・所要時間を保存する。学習は合計3 runs。
5. **Cosine-KMeansを5 fits行う。** 各条件の共通fit画素の表現で$K_0=8$をfitし、
   モデル・中心を固定して全49試料の全有効画素を予測する。試料ごとの再fitは行わない。
6. **matching・スペクトル集計・可視化を行う。** 観測SNVの試料等重み平均線で
   M00＋Cosine-KMeansへ直接Hungarian matchingする。全49試料のマップ、対応表・類似度・overlap・occupancyと、
   反射率・SNV・疑似吸光度のSG二次微分の代表線・四分位範囲・寄与数を保存する。
   固定7代表試料はCosine-KMeansの5条件の比較図で確認する。
7. **潜在空間・連続map・観測スペクトルの対応を詳しく解析する。**
   [可視化案](design/visualization_and_interpretation.md#latent-spectral-maps)に従い、代表二次微分曲線から
   帯域を選び、残る数値設定を確定する。共通画素のcosine UMAP・連続スペクトル指標map・対応の詳細図をPNGで保存する。
   図表の元数値はCSVに残し、B0・B1・A0・M00・M11で読み取れる領域差と化学的解釈を比較する。
8. **探索的解釈と残件を記録する。** CVの未知試料評価と、全体fitの記述的なマップ・スペクトルを区別する。
   解析・図表・知見・限界を一通り整理した後に、低優先度のmask率sweep・vMFの必要性と工数を再確認する。

vMF数値仕様は引き続きOpen。5条件・共通画素数・各1回・800 epoch・$K_0=8$の方針は維持する。

#### 準備・PCA・GPU smoke

リポジトリrootのPowerShellで実行する。以下は新規出力先を準備する場合の初回作成用であり、完了済みの`global_v1`には再実行しない。
`create`は座標・maskを読み、PCAは共通401,408行のSNV（FP32行列だけで392 MiB）をCPUでfitする。
`smoke`は実寸model・batch size 1024で、A0 → M00 → M11の順に各2 epoch×2 batchと第2 epochの再開を確認する。
3条件合計18 batchをGPUで実行し、raw重みの保存復元、全可視16次元表現、再開時の入力・LR・AMP判断・重みを照合する。
smoke重みは本番へ引き継がない。

```powershell
foreach ($step in @("create", "check", "baseline-fit", "baseline-check", "smoke")) {
    uv run --no-sync python scripts/experiments/global_fit.py $step
    if ($LASTEXITCODE -ne 0) { throw "Global fit preparation failed: $step" }
}
```

初回作成後に既存manifestを検証する場合は`check`、PCAを検証する場合は`baseline-check`を使う。
`create`・`baseline-fit`は既存出力を上書きしない。smokeは日時別の独立directoryを使用する。
途中失敗後は成功済みのcreate/fitを繰り返さず、失敗した段階を確認する。completionのない部分出力は自動再利用しない。
PCAの復元では保存時のC/F配列配置を保持する。配置を変えるとFP32積和の丸めが変わるため、
保存復元の許容差は`1e-6`のまま、同じ配置で照合する。復元checkは一時directoryで行い、合格後に本番出力先へ置く。
初回の`Global PCA roundtrip mismatch`で残った2ファイルは`global_v1/recovery/pca_roundtrip_<timestamp>/`へ退避した。
その後の`baseline-fit`・`baseline-check`・`smoke`は完了した。既存PCAを確認する際は`baseline-check`を使う。

#### 3条件の一括学習と完了check

以下は学習手順の参照である。既存`global_v1`の3条件は完了済みのため、新規学習として再実行しない。
新規出力先で行う場合は上の全段階が正常終了してから実行し、1 GPUでA0 → M00 → M11を直列に各800 epoch学習する。
392 batch/epoch、313,600 attempted updates/run、計940,800 attempted updatesが予定値である。
AMP overflowによるskipと実optimizer更新数は別途記録し、実更新数を予定値と同一とは仮定しない。

```powershell
uv run --no-sync python scripts/experiments/global_fit.py train --conditions A0 M00 M11
if ($LASTEXITCODE -ne 0) { throw "Global training failed; inspect the checkpoint before resuming" }
uv run --no-sync python scripts/experiments/global_fit.py training-check
if ($LASTEXITCODE -ne 0) { throw "Global training completion check failed" }
```

一括実行は最初の失敗で停止する。中断後は次の1コマンドで、完了済み条件を検証してskip、
未完了条件を同じrunの`last.pt`から再開、未着手条件を新規学習する。途中epochは最後の保存境界から再実行する。
run/config/manifest/code/runtimeが一致しない場合や、既存runにcheckpointがない場合は停止する。
GPUを変更する場合は初回から一貫して`--device`を指定する（既定0）。

```powershell
uv run --no-sync python scripts/experiments/global_fit.py train --conditions A0 M00 M11 --resume
if ($LASTEXITCODE -ne 0) { throw "Global training resume failed" }
uv run --no-sync python scripts/experiments/global_fit.py training-check
if ($LASTEXITCODE -ne 0) { throw "Global training completion check failed" }
```

#### 全体fitの成果物

すべて`outputs/experiments/global_v1/`配下。全体fit成果物をCVのOOF snapshotへ加えない。

| 保存先 | 内容 |
| --- | --- |
| `config/experiment.json`・`config/seeds.json` | 全体fit契約・55 seeds（後続クラスタリング用を含む） |
| `manifests/samples.parquet`・`manifests/fit_pixels.parquet` | 全49試料と共通401,408座標。fold列なし |
| `manifests/inputs.json`・`manifests/complete.json` | 元入力のfingerprint・manifestのhashと予定更新数 |
| `results/baselines/b0.json`・`completion.json` | B0変換契約、PCA由来・solver・復元probe・hash |
| `checkpoints/baselines/pca.npz` | PCA係数（pickleなし） |
| `results/neural/{condition}/repeat_1/` | `run.json`・history・checkpoint記録・attempt・`completion.json` |
| `checkpoints/neural/{condition}/repeat_1/last_model.pt` | epoch 800の最終raw重み |
| `checkpoints/neural/{condition}/repeat_1/checkpoints/last.pt` | optimizer・AMP scaler・各RNG・epoch境界を含む再開用state |
| `results/neural_smoke/{timestamp}/{condition}/repeat_1/smoke.json` | 短いGPU検証の合否と再開誤差。対応するcheckpointは`checkpoints/neural_smoke/` |

`training-check`は保存checkpointを実際に読み、epoch・更新数・manifest/config/code/runtime・重みhashと、
checkpoint中の重みと最終raw重みの一致を確認する。全3条件の完了確認後に、手順5の全体Cosine-KMeansへ進む。

#### 2026-09-20に確認した全体fitの完了記録

B0/PCAの[完了記録](../outputs/experiments/global_v1/results/baselines/completion.json)は
`fitted_and_roundtrip_checked`。PCAは全49試料の共通401,408画素で16成分をfitし、
solverは`covariance_eigh`、保存復元probeの最大絶対誤差は0である。
GPU smokeはA0・M00・M11のすべてで`checks_passed=true`、再開時の重み・全可視潜在の最大絶対誤差は0。

| 条件・完了記録 | epoch | attempted updates | optimizer updates | AMP skips |
| --- | ---: | ---: | ---: | ---: |
| [A0](../outputs/experiments/global_v1/results/neural/A0/repeat_1/completion.json) | 800 | 313,600 | 313,483 | 117 |
| [M00](../outputs/experiments/global_v1/results/neural/M00/repeat_1/completion.json) | 800 | 313,600 | 313,483 | 117 |
| [M11](../outputs/experiments/global_v1/results/neural/M11/repeat_1/completion.json) | 800 | 313,600 | 313,481 | 119 |

全3条件の`completion.json`は`training_completed`、attempt記録は`completed`である。
表は保存記録の値を示し、`training-check`の完了は同日のユーザー報告に基づく。
PCA・NNのfitが完了した段階であり、全体クラスタリング・全画素予測・解釈図が生成済みという意味ではない。

<a id="global-post-fit"></a>

#### 学習完了後の作業

保存済みモデルを利用する**表現抽出・Cosine-KMeans・全画素予測・check**を`global_cluster.py`へ、
**matching・観測スペクトル集計・PNG/CSV生成・check**を`visualize_global.py`へ分離して実装した。
`global_fit.py`は引き続き準備・baseline・smoke・NN学習とcheckを担当する。
既存の`cluster_representations.py`は`--fold`必須のCV用なので、`global_v1`へそのまま実行しない。
以下の手順1〜3は実装・合成データのCPU検証済みで、本番の5 fitsと全画素処理はユーザーが実行する。

1. 保存済みPCA・最終NN重み・共通global manifestを読み、B0・B1・A0・M00・M11の表現を抽出する。
   B0は256次元SNV、B1・NNは16次元で、既定のL2正規化・全可視抽出を用いる。
   既存の学習・PCAを再fitせず、共通401,408画素の表現で固定seedのCosine-KMeansを各1回、$K_0=8$でfitする。
2. 各条件の固定モデル・中心を全49試料の全3,902,250有効画素へ適用し、画素座標と対応したラベルを保存する。
   保存復元・対象数・ラベル範囲・出典のcheckを行う。試料ごとのクラスタリング再fitは行わない。
3. 観測SNVの試料等重み平均線でM00＋Cosine-KMeansへラベルを整列する。
   クラスタmap・occupancy・対応表と、反射率・SNV・疑似吸光度のSG二次微分の代表線・ばらつきをPNG・CSVにする。
4. 二次微分の代表線を見て帯域を選び、平滑化・積分・UMAPの未確定設定を決める。
   cosine UMAP、クラスタに依存しない連続スペクトル指標map、空間位置とスペクトルの対応図を作成し、化学的解釈を整理する。

帯域選択やUMAP設定の確定は手順1〜3の着手条件ではない。まずクラスタmapと代表二次微分スペクトルを確認できる状態にする。
mask率sweep・vMFは、この解析・図表・解釈を一通り終えた後の低優先度の計画として維持する。

##### 実行コマンド：クラスタリングと可視化

リポジトリrootのPowerShellで次を順番に実行する。
クラスタリングはB0 → B1 → A0 → M00 → M11の順に1 GPUで処理し、最初の失敗で停止する。
条件ごとの完了後に保存中心・全map・出典をcheckする。可視化も保存後にcheckするため、下記2コマンドで手順1〜3を実行できる。

```powershell
uv run --no-sync python scripts/experiments/global_cluster.py run
if ($LASTEXITCODE -ne 0) { throw "Global clustering failed" }

uv run --no-sync python scripts/experiments/visualize_global.py run
if ($LASTEXITCODE -ne 0) { throw "Global visualization failed" }
```

`global_cluster.py run`はCUDA必須（既定`--device 0`）で、表現抽出と全画素予測のchunkは既定1,024画素。
fitには共通401,408画素の表現全体を一度に使用し、chunkごとの再fitは行わない。
B0のfit用FP32配列だけで392 MiB、B1・NNの16次元配列は24.5 MiBで、KMeansの作業領域とGPU転送先は別途必要となる。
NNはepoch 800の最終raw重みを全可視・FP32で使う。既存PCA・NNのfitを繰り返さず、追加のrestartも行わない。
表現配列は処理中のみ保持し、今回の保存対象は中心・ラベル・由来と診断記録である。

`visualize_global.py run`はCPUで保存済み全5条件のラベルと観測スペクトルを読み、KMeans fitやNN推論を行わない。
観測スペクトルは既定2,048画素のchunkで読み、全49試料を集計する。PNGは既定240 dpi（`--dpi`で変更可能）。
先に全5条件のクラスタリングを完了する。可視化だけを一部条件で生成する設定は設けない。

完了後に保存物だけを再検証する場合は次を使う。GPU不要で、出典・hash・有効maskとラベル範囲・成果物数などを検証する。

```powershell
uv run --no-sync python scripts/experiments/global_cluster.py check
if ($LASTEXITCODE -ne 0) { throw "Global clustering check failed" }
uv run --no-sync python scripts/experiments/visualize_global.py check
if ($LASTEXITCODE -ne 0) { throw "Global visualization check failed" }
```

途中中断後は`global_cluster.py run --resume`を使う。出力のある条件をcheckしてからskipし、未着手条件を実行する。
通常の処理例外では作業用directoryを片付け、当該条件を最初から再実行できる。KMeansの途中反復からの再開はしない。
強制終了で正式保存先の片側だけが残った場合や、既存出力の出典・hashが不一致の場合は停止し、自動削除・再利用しない。
`visualize_global.py run --resume`は既存の完成した図表をcheckしてskipする用途で、再描画や途中からの集計再開は行わない。
既存出力がなければ通常の生成を行う。どちらも既存の正式保存先を無条件に上書きしない。

##### 保存物と確認点

以下はすべて`outputs/experiments/global_v1/`からの相対path。

| 保存先 | 内容 |
| --- | --- |
| `checkpoints/clustering/{condition}/repeat_1/centers_k8.npz` | 固定K=8中心、global seed・fit画素数・設定・fit診断 |
| `results/clustering/{condition}/repeat_1/maps/{sample_id}.npz` | 全有効画素の元ラベル。`labels_k8`、背景0・クラスタ1〜8 |
| `results/clustering/{condition}/repeat_1/run.json`・`completion.json` | 表現の由来、code/runtime/hash、保存復元、試料別画素数・occupancy・完了記録 |
| `results/figures/global_k8_v1/` | 下記のPNG・CSVと`report.json`・`completion.json` |

図表directory直下のPNGは次の9枚で、全試料の個別マップ49枚と合わせて58枚となる。

| PNG | 内容 |
| --- | --- |
| `01_representative_maps.png` | 固定7代表試料×5条件。列a〜eはB0・B1・A0・M00・M11 |
| `02_reflectance_spectra.png` | 反射率の代表線・試料間IQR |
| `03_snv_spectra.png` | 観測SNVの代表線・試料間IQR |
| `04_second_derivative_spectra.png` | 疑似吸光度SG二次微分の代表線・試料間IQR |
| `05_snv_similarity.png` | M00基準への観測SNV cosine類似度 |
| `06_contingency.png` | M00クラスタごとに正規化した同一画素の対応。matchingの目的関数には使わない |
| `07_reflectance_differences.png` | 各条件 − M00の反射率代表線差 |
| `08_snv_differences.png` | 各条件 − M00のSNV代表線差 |
| `09_second_derivative_differences.png` | 各条件 − M00の二次微分代表線差 |

スペクトル図は表示クラスタ1〜8の2行4列で、5条件（差はM00以外の4条件）の線を重ねる。
代表線の帯は試料間IQRであり信頼区間ではない。差スペクトルは代表線どうしの差で、構成試料が異なり得るためpaired差ではない。
`maps/{sample_id}_k8.png`は全49試料の5条件比較、`labels/{sample_id}.npz`は条件名をkeyとする整列済みラベル。
元ラベル・中心は変更しない。`spectral_summary.npz`にも波長・代表線・IQR・表示番号対応を保存する。

CSVは9個：`wavelengths.csv`、`sample_spectra.csv`、`spectrum_counts.csv`、`representative_spectra.csv`、
`difference_spectra.csv`、`matching.csv`、`matching_matrices.csv`、`occupancy.csv`、`captions.csv`。
試料別スペクトルの`band_000`〜`band_255`は`wavelengths.csv`でnmへ対応する。
代表線は試料内平均→試料間等重み平均。疑似吸光度の全帯域正値判定・除外数と、曲線ごとの寄与試料数・ID・画素数を保持する。
未定義曲線はNPZでNaN、CSVで空欄。SNV代表線が欠損・非有限・ゼロnormの場合は条件・クラスタをエラーに示して停止し、対応を捏造しない。
今回、空間平滑化と帯域積分は行わない。`04_second_derivative_spectra.png`と寄与・除外数のCSVを確認してから手順4へ進む。

CPU小規模検証では固定seedで1回だけfitすること、保存復元、端数chunk・座標対応、試料等重み集計、
画素別対数変換、SGのnm単位、SNV matching、PNG/CSV出力、改変検出、再開時のcheckと停止を確認した。
既存の全体fitテストも合格。本番の全画素処理・GPU実行・科学的な図の読み取りは未確認である。

<a id="oof-aggregation"></a>

## 9. OOF集計

指定する全conditionについて5 folds × 3 repeatsの評価が揃ってから実行する。snapshot名は一度だけ
使用し、既存snapshotは `check` で読む。

`main_oof_v1`は作成・check完了済みである。保存された完了記録は49試料・105 source runs・72,030 score records。
以下は新規作成からの手順例であり、現在のsnapshotの再確認には`check`だけを実行する。

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

mask率補助実験は低優先度とし、主条件の図表生成・全体fit・第8.2節の解析と解釈の整理を一通り終えた後に実施する。別snapshotにする。

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

### 9.1 B0・B1・A0・M00 OOF sanity可視化

B0・B1・A0・M00の全5 folds×3反復のclustering・評価が完了した成果物をCPUで読み、
[専用の表示仕様](design/oof_sanity_visualization.md)に従ってPNG 5枚とCSV 3つを生成する。
全主条件のOOF snapshot作成や、モデルの再fitは不要である。

保存済み図は`outputs/sanity_checks/a0_m00_oof_visualization/`を参照する。
再生成時は4条件と新規出力先を明示する。以下のプレースホルダーを、まだ存在しない出力先に置き換える。

```powershell
uv run --no-sync python scripts/experiments/visualize_b0_b1_oof.py `
    --conditions B0 B1 A0 M00 `
    --experiment-dir outputs/experiments/production_v1 `
    --output-dir '<新規出力先>'
```

終了code 0と保存物を確認する。条件別の代表7試料図でKYOw名が各試料の下にあり、
silhouetteに下段subplotがないことを確認する。補正前LLA・LFR・occupancyと未定義理由はCSVで読む。
sanity出力にはログやcompletion JSONを追加しない。

<a id="oof-reporting"></a>

### 9.2 主条件OOF図表（実装・生成済み）

`main_oof_v1`を出典に、主7条件・全49試料・全7KのPNG 3枚とCSV 11個を生成する。
代表指標はLLA（保存キー`adjusted_lla_*`）、LFR(TGN+FS)、ARI、Cosine-Silhouette、Cluster Occupancyの順。
LLAの窓3・5・9を個別に示し、Occupancyと交互作用はCSVだけにする。
LFRはCSVにTGN+FS・TGN単独・FS単独を保存し、PNGはLFR(TGN+FS)のみ表示する。
`source_metric`は元の保存キー`lfr_both`・`lfr_noise`・`lfr_shift`を保持する。
各PNGのLLA 3パネルは共通の縦軸範囲とし、01・02の縦軸目盛りは全パネル0.1刻みとする。
03のLLAは既存の目盛り間隔を維持し、異なる指標の縦軸範囲は個別に設定する。

```powershell
uv run --no-sync python scripts/experiments/visualize_main_oof.py `
    --experiment-dir outputs/experiments/production_v1 `
    --snapshot main_oof_v1
if ($LASTEXITCODE -ne 0) { throw 'OOF reporting failed' }
```

既定の保存先は`outputs/experiments/production_v1/results/figures/main_oof_v1/`。
`--output-dir`で変更でき、`--dpi`の既定値は240である。
再生成時はこのrendererが管理する出力を上書きし、統合前のPNG・Occupancy図・paired分布図・交互作用図を削除する。
元のOOF snapshot、clustering・評価成果物と、出力先の無関係なファイルは保持する。
失敗時は非0終了とする。入力検証後、出力の上書きを始める時点で旧`completion.json`を除去し、
上書き途中の失敗・中断を更新前の成功記録で完了扱いにしない。

| ファイル | 内容 |
| --- | --- |
| `01_main_metrics_k_sweep.png` | 2行3列。上段LLA 3・5・9、下段LFR(TGN+FS)・ARI・Cosine-Silhouette。主7条件の全K曲線 |
| `02_k8_distributions.png` | 同じ2行3列・指標順の試料別分布。$K_0=8$、黒線はmacro平均、`n`は共通対象数 |
| `03_paired_k_sweep.png` | 同じ2行3列にM11−B0・M11−B1・M11−M00の差を表示 |
| `metrics_all_k.csv`、`metrics_k8.csv` | 代表指標・LFR 3種類・clean testのOccupancy、平均・試料間SD・反復間SD・対象数 |
| `paired_all_k.csv`、`paired_k8.csv` | 計画済み10比較。LLA・LFR 3種類・ARI・Cosine-Silhouetteのpaired差 |
| `interaction_all_k.csv`、`interaction_k8.csv` | LLA 3・5・9とLFR 3種類の2×2交互作用 |
| `sample_values.csv` | 条件別・paired・交互作用の試料別値、反復値、共通対象への採否 |
| `availability.csv` | 反復ごとの定義済み対象数と未定義の試料・理由 |
| `ari_pairs.csv` | ARIの3反復対の値・未定義理由・使用クラスタ数・退化flag |
| `occupancy_samples.csv`、`occupancy_folds.csv` | clean testの試料別分布、fold・反復別の画素pool分布と試料macro分布 |
| `report.json` | 出典hash・生成コードhash・対象・指標定義・図の配置・captionに必要な情報 |
| `completion.json` | `status=oof_figures_completed`と管理対象出力のhash |

LLA・LFR(TGN+FS)・Cosine-Silhouetteの曲線は実線がmacro平均、破線・点線・一点鎖線が反復1・2・3。
ARIは試料内の3反復対平均を試料macro平均し、paired ARIはこの試料平均の条件間差を共通対象で集計する。
ARIとその差には反復別曲線・反復間SDを付けない。詳細は[報告規約](design/evaluation_metrics.md#reporting)を参照する。

補正後LLAの交互作用とpaired ARIは保存済みscoreから報告pipelineで求める。
rawスペクトル・重み・mapは読み込まず、学習・推論・CV評価を再実行しない。
消費するsnapshot・元成果物のmetadata hashと対象対応を確認するが、OOFの完全な`check`の代用にはしない。
終了code 0、PNGの01〜03、CSV 11個、completion status、文字・凡例・対象数の表示を確認する。

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
