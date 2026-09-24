# 本番実験runbook

## 1. 役割

固定設計を`production_v1`で実行する手順を示す。条件・seed・split・評価方法の定義は
[研究設計](design/README.md)、現在の進捗は[ToDo](../ToDo.md)を参照する。

本書のclustering・評価・OOFコマンドはCosine-KMeansの主実験・mask率補助実験用である。
mask率補助実験の範囲と実施順序は[第8.1節](#mask-rate-sweep)を参照する。

ChemoMAE v0.2.2の環境で、リポジトリrootからPowerShellで実行する。
本番実行例は`uv run --no-sync`とし、環境構築・更新をrun開始時の処理から分ける。

2026-09-25のユーザー完了報告により、A1を含む主8条件のCV・OOF・図表、全体6条件のfit・K8図表・PCAまで完了した。
主条件は`main_oof_v1`、全体図表・PCAは`global_k8_v1`へ上書き再生成済み。
A1追加時の操作記録は[追補runbook](a1_extension_runbook.md)を参照する。今回の文書更新では再実行・成果物再検証は行っていない。
残るToDoはmask ratio sweepと位置対応FT-IRのみ。以下の既存成果物のコマンドは参照・再生成用である。

現在は主条件CV・OOF、全体fit・K8図表、PCA一枚まで生成済みである。
執筆開始時は[引き継ぎ資料](manuscript_handoff.md)から既存成果物を参照し、本書のコマンドは再生成・再検証が必要な場合に使う。

- [入力確認](#input-preparation)・[manifest](#3-本番manifest)
- [NNの1 runと再開](#neural-run)・[PCA](#5-b1-pca-fitとbaseline変換の検証)
- [clean test map](#6-clean-test-map)・[評価](#7-評価)
- [OOF集計](#oof-aggregation)・[OOF sanity図](#oof-sanity)・[主条件OOF図表](#oof-reporting)
- [全体fitの準備・一括学習](#global-fit-pipeline)・[PCA一枚](#global-pca)・[保存規約](#artifact-records)

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

ニューラル条件は `A0`、`A1`、`M00`、`M10`、`M01`、`M11`、`M11-25`、`M11-75` である。
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
| 主実験の学習 | A0、A1、M00、M10、M01、M11 | 90 runs |
| mask率補助学習 | M11-25、M11-75 | 30 runs |

各組合せについてclean mapと評価を完了する。ニューラル学習は合計120 runsで、B0・B1のfitや
KMeans、評価処理はこの数に含めない。3反復はseed選別に使わず、すべてOOF集計へ含める。

<a id="mask-rate-sweep"></a>

### 8.1 mask率補助実験の実施

M11のmask率25%・50%・75%を比較する補助実験を実施する。
50%は完了済みの主条件M11を再利用し、M11-25・M11-75を各5 folds×3反復、計30 runs追加する。
800 epoch、split、抽出画素、seed、augmentation強度、全7Kと評価指標は主実験の固定条件を継承する。
最良mask率の選択ではなく感度解析として報告し、主条件M11を事後的に置き換えない。

1. 第4節の学習・再開手順でM11-25・M11-75を実行し、各runをcheckする。
2. 第6〜7節に従い、各runでCosine-KMeans・全test画素の予測・評価・checkを完了する。
3. M11-25・M11・M11-75の3条件を[OOF集計](#oof-aggregation)の`mask_rate_oof_v1`へまとめる。主条件のsnapshotは保持する。
4. mask率比較の図表生成対応を追加し、LLA・LFR・ARI・Cosine-Silhouette・occupancyとK依存性を報告する。

主条件の図表用CLIを、そのままmask率snapshotへ流用しない。mask率図表への対応は未実装である。
条件の定義は[実験プロトコル](design/experiment_protocol.md#mask-rate-sweep)、
残作業は[ToDo第4節](../ToDo.md#4-mask率補助実験)を参照する。
この実験の着手に、追加の潜在空間図や物理化学的解釈の完了を要求しない。

<a id="global-fit-pipeline"></a>

### 8.2 全体fitと解釈

主条件OOF図表、全体fit、K8クラスタリング・図表、PCA一枚は生成済みである。
残作業はmask率sweep（[第8.1節](#mask-rate-sweep)）と位置対応FT-IRとし、執筆はThesisで管理する。

[全体可視化設計](design/visualization_and_interpretation.md)に従い、全49試料の共通抽出画素で
B1 PCAとA0・A1・M00・M11をfitする。B0を加えた6条件の表現で、$K_0=8$のCosine-KMeansを6 fits行う。
全体学習の4 runsと全体クラスタリングは、補助条件を含むCVの計画120学習とは別枠であり、OOF集計に含めない。

完了範囲は[ToDo第3節](../ToDo.md#3-全体fitと解釈)を参照する。
`global_fit.py`にmanifest・B0/PCA・A0/A1/M00/M11の一括学習・再開・checkを実装した。
既存A0・M00・M11の検証履歴は下記の2026-09-20の記録を参照する。
2026-09-25のユーザー報告によりA1の800 epoch学習・K8クラスタリングと、全6条件の図表・PCA再生成まで完了した。
既存の`train_neural.py`は`--fold`必須のCV用であり、全体fitには使わない。

以下は処理工程の参照である。手順1〜7は完了済み。手順8の執筆はThesis側で管理し、追加可視化案は現在の残ToDoに含めない。
実行コマンドとその後の順序は[学習完了後の作業](#global-post-fit)を参照する。

1. **全体runの実行契約を固定する（確定・実装済み）。** ROOT_SEED=20260905・SHA-256方式を維持し、
   fold位置を`global`、反復IDを1とする。抽出・PCA・NNと後続の$K_0=8$クラスタリング用のseed計55個を保存する。
   保存rootは`outputs/experiments/global_v1/`。2026-09-19のユーザー確認による。
2. **共通入力と専用pipelineを実装・小規模検証する。** 49試料から各8,192画素、計401,408画素を抽出し、
   全6条件で座標を共有する。fit画素と推論対象の全有効画素を区別し、CPU小規模と必要最小限のGPU検証で、
   入出力・seed・保存復元・完了判定を確認する。CVのmanifestと成果物は全体fitから分離する。
3. **B0・B1を準備する。** B0は固定のSNV変換、B1は共通fit画素でPCAを1回fitする。
   全体fit用の入力・表現抽出・保存復元を確認してから長時間のNN学習へ進む。
4. **A0・A1・M00・M11を各800 epochで1回ずつ学習する。** CVと同じ条件別recipeを使用し、
   最終重み・実際の更新回数・実行環境・seed・所要時間を保存する。学習は合計4 runs。
5. **Cosine-KMeansを6 fits行う。** 各条件の共通fit画素の表現で$K_0=8$をfitし、
   モデル・中心を固定して全49試料の全有効画素を予測する。試料ごとの再fitは行わない。
6. **matching・スペクトル集計・可視化を行う。** 試料等重みの観測SNV代表線のcosine類似度で
   M00＋Cosine-KMeansへ直接Hungarian matchingする。全49試料のマップ、対応表・類似度・overlap・occupancyと、
   反射率・SNV・疑似吸光度のSG二次微分の代表線・四分位範囲・寄与数を保存する。
   固定7代表試料はCosine-KMeansの6条件の比較図で確認する。
7. **共通画素のPCAを保存する。** [第8.3節](#global-pca)の設定で6条件の座標と2行6列PNG一枚を保存する。
   図は表現の補助表示とし、条件の優越性や化学的妥当性を図の分離だけで判断しない。
8. **既存結果を書き、必要な詳細解析を絞る。** CVの未知試料評価と、全体fitの記述的なマップ・スペクトルを区別する。
   M11・K8の試料内クラスタの観察を中心に検討し、追加図・帯域指標は採否と仕様を決めてから実装する。

6条件・共通画素数・各1回・800 epoch・$K_0=8$の方針は維持する。

#### 準備・PCA・GPU smoke

リポジトリrootのPowerShellで実行する。以下は新規出力先を準備する場合の初回作成用であり、完了済みの`global_v1`には再実行しない。
`create`は座標・maskを読み、PCAは共通401,408行のSNV（FP32行列だけで392 MiB）をCPUでfitする。
`smoke`は実寸model・batch size 1024で、A0 → A1 → M00 → M11の順に各2 epoch×2 batchと第2 epochの再開を確認する。
4条件合計24 batchをGPUで実行し、raw重みの保存復元、全可視16次元表現、再開時の入力・LR・AMP判断・重みを照合する。
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

#### 4条件の一括学習と完了check

以下は学習手順の参照である。既存`global_v1`の4条件は完了済みのため、新規学習として再実行しない。
以下の一括resume・`training-check`は、同じ現行configで学習したrunが対象となる。
A1追加時に記録を更新した旧完了学習のoptimizer checkpointは更新していないため、
その旧runの再開・checkpoint照合には使わない。既存の最終重み・ラベルの確認は`global_cluster.py check`を使う。
新規出力先で行う場合は上の全段階が正常終了してから実行し、1 GPUでA0 → A1 → M00 → M11を直列に各800 epoch学習する。
392 batch/epoch、313,600 attempted updates/run、計1,254,400 attempted updatesが予定値である。
AMP overflowによるskipと実optimizer更新数は別途記録し、実更新数を予定値と同一とは仮定しない。

```powershell
uv run --no-sync python scripts/experiments/global_fit.py train --conditions A0 A1 M00 M11
if ($LASTEXITCODE -ne 0) { throw "Global training failed; inspect the checkpoint before resuming" }
uv run --no-sync python scripts/experiments/global_fit.py training-check
if ($LASTEXITCODE -ne 0) { throw "Global training completion check failed" }
```

一括実行は最初の失敗で停止する。中断後は次の1コマンドで、完了済み条件を検証してskip、
未完了条件を同じrunの`last.pt`から再開、未着手条件を新規学習する。途中epochは最後の保存境界から再実行する。
run/config/manifest/code/runtimeが一致しない場合や、既存runにcheckpointがない場合は停止する。
GPUを変更する場合は初回から一貫して`--device`を指定する（既定0）。

```powershell
uv run --no-sync python scripts/experiments/global_fit.py train --conditions A0 A1 M00 M11 --resume
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
checkpoint中の重みと最終raw重みの一致を確認する。全4条件の完了確認後に、手順5の全体Cosine-KMeansへ進む。

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
追加A1は2026-09-25のユーザー報告で完了とする。A1のoptimizer updates・AMP skipsは今回再確認していないため、この表へ推定値を補わない。
この表は学習段階の記録である。後続のK8マップ・図表とPCAの現状・操作は以下の節で扱う。

<a id="global-post-fit"></a>

#### 学習完了後の作業

保存済みモデルを利用する**表現抽出・Cosine-KMeans・全画素予測・check**を`global_cluster.py`へ、
**matching・観測スペクトル集計・PNG/CSV生成・check**を`visualize_global.py`へ分離して実装した。
`global_fit.py`は引き続き準備・baseline・smoke・NN学習とcheckを担当する。
既存の`cluster_representations.py`は`--fold`必須のCV用なので、`global_v1`へそのまま実行しない。
以下の手順1〜3はA1を含めて完了済み（2026-09-25ユーザー報告）。既存の6 fits・全画素ラベルを再利用できる。

1. 保存済みPCA・最終NN重み・共通global manifestを読み、B0・B1・A0・A1・M00・M11の表現を抽出する。
   B0は256次元SNV、B1・NNは16次元で、既定のL2正規化・全可視抽出を用いる。
   既存の学習・PCAを再fitせず、共通401,408画素の表現で固定seedのCosine-KMeansを各1回、$K_0=8$でfitする。
2. 各条件の固定モデル・中心を全49試料の全3,902,250有効画素へ適用し、画素座標と対応したラベルを保存する。
   保存復元・対象数・ラベル範囲・出典のcheckを行う。試料ごとのクラスタリング再fitは行わない。
3. 試料内クラスタ平均を試料間で等重み平均した観測SNV代表線のcosine類似度＋HungarianでM00＋Cosine-KMeansへラベルを整列する。
   クラスタmap・occupancy・対応表と、反射率・SNV・疑似吸光度のSG二次微分の代表線・ばらつきをPNG・CSVにする。
4. [PCA一枚](#global-pca)は生成済み。以降の詳細図は、執筆で不足した根拠に応じて選ぶ。
   試料別スペクトルCSVを先に参照でき、帯域積分mapや追加のPCA色分けは必須工程ではない。

帯域選択・平滑化・積分の数値設定は未確定である。採用時の定義は[可視化設計](design/visualization_and_interpretation.md#spectral-band-selection)を参照する。
mask率sweepは[第8.1節](#mask-rate-sweep)に従って実施する。

##### 実行コマンド：クラスタリングと可視化

以下は新規の全体クラスタリング出力先に対する手順。既存の`global_v1`はクラスタリング済みなので、
可視化を再生成する際は2番目の`visualize_global.py run`だけを実行する。
新規出力先ではリポジトリrootのPowerShellで次を順番に実行する。
クラスタリングはB0 → B1 → A0 → A1 → M00 → M11の順に1 GPUで処理し、最初の失敗で停止する。
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

`visualize_global.py run`はCPUで保存済み全6条件のラベルと観測スペクトルを読み、KMeans fitやNN推論を行わない。
観測スペクトルは既定2,048画素のchunkで読み、全49試料を集計する。PNGは既定240 dpi（`--dpi`で変更可能）。
先に全6条件のクラスタリングを完了する。可視化だけを一部条件で生成する設定は設けない。

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
可視化の通常の`run`は図表を再生成し、完成後に指定の`global_k8_v1`を丸ごと置き換える。
旧PNG・CSV・ラベル表示用NPZは残さず、新旧の構成を混在させない。途中失敗時は元の可視化を保持する。
独立管理の`pca-latent-2d/`は引き継ぎ、PCA座標・NN・クラスタリング成果物は置換対象に含めない。
クラスタリングの`run`は既存成果物を上書きしない。

<a id="representative-sample-update"></a>

##### 代表7試料の変更を既存図へ反映する

2026-09-21に代表試料を、各樹種で多様な化学状態が見られそうなものを目視で選んだ7試料へ変更した。
試料番号順に`KYOw02752`（ヒノキ）、`KYOw02772`（マツ）、`KYOw02777`（ケヤキ）、
`KYOw02787`（ツガ）、`KYOw02790`（クリ）、`KYOw16744`（スギ）、`KYOw16750`（モミ）とする。
旧指定との差分と選定の位置づけは[表示例の選択](design/visualization_and_interpretation.md#representative-samples)を参照。
コード・文書の更新だけでは保存済みPNGは変わらないため、以下をリポジトリrootのPowerShellで実行する。

raw BMPと反射率L2 normの代表1×7は、保存済み画像をCPUで読み、次の1コマンドで両方を同名上書きする。
既存の全49試料の7×7図は変更しない。前処理・スペクトル計算は行わない。

```powershell
uv run --no-sync python scripts/preprocess/create_sample_overviews.py --representatives-only --overwrite
if ($LASTEXITCODE -ne 0) { throw "Representative sample overviews failed" }
```

対象は`outputs/sample_overviews/raw_bmp_representatives_1x7.png`と
`outputs/sample_overviews/reflectance_l2_norm_representatives_1x7.png`。

全体fitの条件別代表1×7と6条件×7試料図は、次の1コマンドで同時に更新する。
既存の全49試料のラベル・観測スペクトルをCPUで読み、図表一式（PNG 67枚・CSV 44個と関連記録）を
再生成して同じ`results/figures/global_k8_v1/`へ置き換えるため、代表図2枚の合成より処理量が多い。
学習・クラスタリングは再実行せず、独立した`pca-latent-2d/`は保持する。

```powershell
uv run --no-sync python scripts/experiments/visualize_global.py run
if ($LASTEXITCODE -ne 0) { throw "Global representative visualization failed" }
```

対象の代表図は`{B0,B1,A0,A1,M00,M11}/labels/07_representative_samples_1x7.png`と
rootの`01_representative_samples_6x7.png`。`report.json`の`representative_sample_ids`と
`captions.csv`も新しい7試料・並び順へ更新される。既存図の検証だけを行う`--resume`は付けない。
終了code 0、図中のIDと列順、`report.json`の代表ID、実行末尾の`checks_passed=true`を確認する。
既存のOOF sanity図はこの2コマンドの更新対象に含まれない。

##### 保存物と確認点

以下はすべて`outputs/experiments/global_v1/`からの相対path。

| 保存先 | 内容 |
| --- | --- |
| `checkpoints/clustering/{condition}/repeat_1/centers_k8.npz` | 固定K=8中心、global seed・fit画素数・設定・fit診断 |
| `results/clustering/{condition}/repeat_1/maps/{sample_id}.npz` | 全有効画素の元ラベル。`labels_k8`、背景0・クラスタ1〜8 |
| `results/clustering/{condition}/repeat_1/run.json`・`completion.json` | 表現の由来、code/runtime/hash、保存復元、試料別画素数・occupancy・完了記録 |
| `results/figures/global_k8_v1/` | 下記のPNG・CSVと`report.json`・`completion.json` |

図表directoryを`B0/`・`B1/`・`A0/`・`A1/`・`M00/`・`M11/`へ分ける。各条件のPNGは11枚。
rootの`01_representative_samples_6x7.png`を加え、全体で67枚となる。
この比較図は固定7代表試料を列、上からB0・B1・A0・A1・M00・M11を行にする。試料IDは最下段だけに表示する。
次のpathは各条件directoryからの相対path。

| PNG | 内容 |
| --- | --- |
| `labels/00_samples_01-07_1x7.png`〜`06_samples_43-49_1x7.png` | 試料ID昇順で7試料ずつ。49試料を7枚へ分割 |
| `labels/07_representative_samples_1x7.png` | 固定代表7試料。rootの6条件×7試料図と同じ試料・並び順 |
| `labels/08_all_samples_7x7.png` | 同じ順序の全49試料、7×7 |
| `01_representative_spectra.png` | 上段SNV・反射率、下段全面SG二次微分。各panelを8クラスタの色で比較 |
| `02_snv_cosine_matrix.png` | 縦M00・横当該条件の8×8 SNV cosine類似度。表示番号順、値域−1〜1 |

マップは`outputs/sample_overviews`の余白・大きな太字の試料IDに合わせる。色は最近傍補間で保持し、colorbarは`Cluster ID`。
スペクトルはマップと共通のクラスタ色。平均線の帯は試料間IQRであり信頼区間ではない。
波長の主目盛100 nm・副目盛50 nmにグリッド線を引く。端点の数値は小数2桁を優先し、隣接30 nm未満の数字は省く。
SNVは縦軸−2〜2、反射率は0〜1、二次微分は6条件共通の自動範囲とする。
各条件の`label_maps.npz`は試料IDをkeyとする整列済みラベル。元ラベル・中心は変更しない。
rootの`spectral_summary.npz`には全条件の波長・代表線・IQR・元番号から表示番号への対応を保存する。

CSVは44個：各条件に`sample_spectra.csv`、`spectrum_counts.csv`、`representative_spectra.csv`、
`difference_spectra.csv`、`matching.csv`、`matching_matrices.csv`、`occupancy.csv`の7個、rootに`wavelengths.csv`と`captions.csv`。
差は各条件 − M00の代表線差で、構成試料が異なり得るためpaired差ではない。M00の差表はheaderのみ。
matching表は`snv_cosine_similarity`、同じ基準クラスタに対する他候補との類似度差、一致画素数、元番号と表示番号を保持する。
補助値として`iou`も保存する。IoUは全共通画素poolであり試料等重みではなく、SNVの割当には使用しない。
試料別スペクトルの`band_000`〜`band_255`は`wavelengths.csv`でnmへ対応する。
代表線は試料内平均→試料間等重み平均。疑似吸光度の全帯域正値判定・除外数と、曲線ごとの寄与試料数・ID・画素数を保持する。
未定義曲線はNPZでNaN、CSVで空欄。全試料を通して空のクラスタ、非有限またはノルム0のSNV代表線がある場合はmatchingを停止する。
既存図表には空間平滑化・帯域積分を適用していない。
個別試料の解釈には`sample_spectra.csv`と寄与・除外数のCSVを参照し、全試料macro平均と区別する。

CPU小規模検証では固定seedで1回だけfitすること、保存復元、端数chunk・座標対応、試料等重み集計、
画素別対数変換、SGのnm単位、SNV matching、PNG/CSV出力、改変検出、失敗時の旧可視化保持と完成時の置換を確認した。
SNV変更後に関連CPUテスト6件と静的検査が合格した。空間的な重なりとSNVの対応が異なる合成例、
試料等重み集計、欠落・空・ノルム0の代表線、符号付き類似度、描画代表線との一致を確認した。
2026-09-25のユーザー報告により、A1を含む49試料・PNG 67枚・CSV 44個の再生成まで完了した。
上記のCPU検証は過去の履歴であり、今回の文書整理ではテスト・成果物checkを再実行していない。
図表生成の完了と、領域差の物理化学的な解釈の完了は区別する。

<a id="global-pca"></a>

### 8.3 潜在空間のPCA：2行6列一枚

既存のホスト環境で可視化用PCAとPNG描画を行う。
左からB0・B1・A0・A1・M00・M11、上段は画素数hexbin（turbo・共通の対数色範囲）、
下段はSNV整列後のCluster IDと、所属画素のPC1・PC2座標の算術平均を示すcentroid。
上段の各panel上に条件名（22 pt）を付け、PC名・寄与率・少数の数値目盛りを表示する。
軸名・colorbar名は18 pt、目盛りは15 pt、centroid番号は14 pt。配置と定義は`captions.csv`にも保存する。

既存global fitの49試料×8,192＝401,408画素を全条件で共有し、B1以外の5条件の
クラスタリング用L2正規化済み表現へ独立に2次元PCAをfitする。
中心化し、追加の列標準化・whitening・投影後のL2正規化は行わない。
FP64の`PCA(n_components=2, svd_solver="covariance_eigh", whiten=False)`を使い、
座標・平均・主成分係数・固有値・寄与率・runtime・入力hashを保存する。
B1は保存済みbaseline PCAの特異値の降順に上位2成分を選び、L2正規化前の得点を直接表示する。
寄与率も元のSNV全分散に対する保存値を使う。B1の再PCA・baselineの再学習・クラスタリングのやり直しは行わない。
条件間で座標尺度が異なるため表示範囲は条件別とし、同じ条件の上下段では一致させる。hexbinの色範囲は全条件で共通。
詳細は[設計書](design/visualization_and_interpretation.md#latent-spectral-maps)を参照。

**A1を含む入力・6条件の投影・2×6 PNG一枚とCSV 5個は再生成済み（2026-09-25ユーザー完了報告）。**
今回の文書整理では回帰テストや全成果物checkは再実行していない。コード変更時の小規模CPUテストは次のとおり。

```powershell
uv run --no-sync pytest tests/experiments/test_global_pca.py -q
if ($LASTEXITCODE -ne 0) { throw 'PCA CPU tests failed' }
uv run --no-sync pytest tests/experiments/test_global_analysis.py -q -k "pca_export or report_png_csv or publish_report"
if ($LASTEXITCODE -ne 0) { throw 'PCA export/report tests failed' }
```

出力先の`global_k8_v1/pca-latent-2d/`は親のクラスタ図表と独立してcheckする。
親の図表の再生成時には既存`pca-latent-2d/`を引き継ぎ、親のPNG 67枚・CSV 44個の件数へ加算しない。
入力準備には、現行コードと整合した親図表・matchingが必要である。
既存成果物は揃っているため、通常は末尾のcheckまたは描画だけを使う。
入力・座標を新規に用意する場合は、親図表の生成後、リポジトリrootのPowerShellで次を実行する。

```powershell
& {
    # 保存済みモデルから共通画素の表現を抽出（ホストCUDA）、PCAをCPUでfit。
    foreach ($step in @("prepare", "fit")) {
        uv run --no-sync python scripts/experiments/global_pca.py $step --resume
        if ($LASTEXITCODE -ne 0) { throw "PCA failed: $step" }
    }

    # 固定した座標からPNG一枚を描画・check。
    uv run --no-sync python scripts/experiments/visualize_global_pca.py run
    if ($LASTEXITCODE -ne 0) { throw 'PCA plotting failed' }
}
```

`prepare`でのNN表現抽出のみCUDAを使用する（既定device 0、chunk 1,024）。
`fit`とPNG描画はCPU処理である。保存する入力配列は約517 MBと画素CSV、
B0のFP64化だけで約822 MBを要するため、PCA時は入力・作業領域を含むRAMを確保する。
PCAはB1以外の5条件で一度だけ全共通画素を使い、B1は既存PCを取り出す。描画時には再fitしない。
`--resume`は既存の完成した入力・条件を検証してskipする。
入力・投影を同じv1保存先へ再生成する場合は、`prepare`・`fit`の両方で`--resume`の代わりに`--overwrite`を使う。
PCA fitに乱数seedは使わず、scatterの重なり順だけROOT_SEEDで固定する。

| 保存先（`outputs/experiments/global_v1/`からの相対path） | 内容 |
| --- | --- |
| `checkpoints/pca_projection/global_k8_v1/inputs/` | 条件別FP32配列（B1は上位2成分の未正規化得点）、`B1_projection.npz`に既存係数・寄与率・元成分index、画素CSV、出典 |
| `results/pca/global_k8_v1/{condition}/` | `projection.npz`、`coordinates.npy`・`coordinates.csv`、`components.csv`、`explained_variance.csv`、実行・完了記録 |
| `results/figures/global_k8_v1/pca-latent-2d/01_pca_density_clusters.png` | 指定の2行6列PNG一枚 |
| `results/figures/global_k8_v1/pca-latent-2d/`内のCSV | `pixels.csv`・`hexbin_counts.csv`・`centroids.csv`・`explained_variance.csv`・`captions.csv` |

描画だけの変更時は`visualize_global_pca.py run`だけを実行する。新しい図が完成してから`pca-latent-2d/`を置換する。
checkだけを行う場合は次を使う。

```powershell
uv run --no-sync python scripts/experiments/global_pca.py check
if ($LASTEXITCODE -ne 0) { throw 'PCA coordinate check failed' }
uv run --no-sync python scripts/experiments/visualize_global_pca.py check
if ($LASTEXITCODE -ne 0) { throw 'PCA figure check failed' }
```

metadata・帯域指標による追加の色分けは保留中であり、執筆時に必要性を判断する。

<a id="oof-aggregation"></a>

## 9. OOF集計

指定する全conditionについて5 folds × 3 repeatsの評価が揃ってから実行する。
既存snapshotの確認は`check`、同じ保存先への再集計は明示的な`--overwrite`を使う。

`main_oof_v1`は主8条件で再集計・check完了済み（2026-09-25ユーザー報告）。対象は49試料・120 source runs・82,320 score records。
以下は同じv1への再生成例であり、現在のsnapshotの再確認には`check`だけを実行する。

```powershell
uv run --no-sync python scripts/experiments/aggregate_oof.py run `
    --conditions B0 B1 A0 A1 M00 M10 M01 M11 `
    --snapshot main_oof_v1 --overwrite `
    --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'OOF aggregation failed' }
uv run --no-sync python scripts/experiments/aggregate_oof.py check `
    --snapshot main_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
```

mask率補助実験は、M11-25・M11-75の全fold・全反復の評価完了後に、完了済みのM11を加えて集計する。
主条件のsnapshotを保持し、3条件を別snapshotにする。

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

`main_oof_v1`を出典に、主8条件・全49試料・全7KのPNG 3枚とCSV 11個を生成する。
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
| `01_main_metrics_k_sweep.png` | 単段幅2×2。上段LLA 3・5、下段LLA 9・LFR(TGN+FS)。主8条件の全K曲線 |
| `02_k8_distributions.png` | 2×3。上段LLA 3・5・9、下段LFR(TGN+FS)・ARI・Cosine-Silhouette。$K_0=8$、黒線はmacro平均、`n`は共通対象数 |
| `03_paired_k_sweep.png` | 分布図と同じ2×3にM11−B0・M11−B1・M11−M00・A1−A0・M11−A1の差を表示 |
| `metrics_all_k.csv`、`metrics_k8.csv` | 代表指標・LFR 3種類・clean testのOccupancy、平均・試料間SD・反復間SD・対象数 |
| `paired_all_k.csv`、`paired_k8.csv` | 既存10比較＋追加ablation 2比較。LLA・LFR 3種類・ARI・Cosine-Silhouetteのpaired差 |
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
