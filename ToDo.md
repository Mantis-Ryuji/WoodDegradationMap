# 実験実施 ToDo

更新日: 2026-09-05

第1段階のfixtureテストと入力照合はユーザー実行ログで成功を確認済み。
第2段階は20テストと本番manifestの生成・再読込成功をユーザー実行ログで確認済み。
第3段階のloader・B0/PCAはユーザー実行で検証済み（本番PCAはfold 1）。
現在はChemoMAEの共通部品・全可視抽出を実装し、新規CPUテストの結果待ち。
第1段階の残る品質表・空間図確認は、本学習前までに完了する。
この文書は確定済み設計を実行へ移すための作業順序であり、詳細仕様は以下の設計文書を正とする。
チェックは実装しただけでなく、各項目に必要な確認結果が得られてから付ける。

| 参照先 | 内容 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 作業範囲、実行・変更の規約 |
| [研究設計の概要](docs/design/README.md) | 主実験・補助実験・探索的解釈の区分 |
| [実験プロトコル](docs/design/experiment_protocol.md) | 分割、抽出、モデル、学習、K、実行記録 |
| [評価指標](docs/design/evaluation_metrics.md) | 指標の定義、未定義値、集約と計画比較 |
| [可視化と解釈](docs/design/visualization_and_interpretation.md) | 全体学習、ラベル整列、スペクトル要約 |
| [前処理](docs/design/preprocessing.md) | 本番入力とHDF5 schema |

プログラム・テスト・学習・評価の実行は、原則としてユーザーが行う。
Codexは依頼された段階を実装し、実在するコマンド、出力先、確認点を提示する。
Codexへの明示的な実行依頼がある場合はAGENTS.mdの実行規約に従う。
以下の未実装工程について、架空のCLIや実行済み結果は記載しない。

## 0. 引き継ぎ時点の状態

- [x] 主実験・mask率補助実験の研究条件を設計文書へ反映した。
- [x] 前処理コードと画素数確認コードが存在する。
  対象は `src/wood_degradation_map/preprocessing/` と `scripts/preprocess/`。
- [x] ユーザーの実行結果により、現行49試料すべてで8,192画素の非復元抽出が可能と確認した。
  全有効画素数3,902,746、試料別最小26,249・中央値73,259・最大161,735、抽出率5.07–31.21%。
- [ ] 学習・表現抽出・クラスタリング・評価・集計の実験pipelineを実装し、検証する。
- [ ] 本実験を実行する。実験結果やモデル性能はまだ確認していない。

画素数確認は `sample_quality.parquet` に対する確認であり、スペクトルの網羅性やモデル性能の検証ではない。
同じデータでこの確認を繰り返す必要はない。試料集合や前処理が変わった場合の再確認コマンドは次のとおり。

```powershell
uv run python scripts/preprocess/check_sampling_pixels.py --q 8192
```

## 確定済み条件の早見表

詳細な引数・数値処理・例外規則は設計文書に従う。以下の値を次のセッションで候補へ戻さない。

| 項目 | 固定条件 |
| --- | --- |
| 入力 | 200 Hz本番前処理済み、256チャネルSNV |
| split | 試料単位のランダム5-fold、樹種層化なし。同一splitで3反復 |
| train画素 | 各試料8,192画素、一様ランダム・非復元。同じfoldでは全条件・全K・全反復で共通 |
| test画素 | 各test試料の全有効画素 |
| K | $\{2,4,6,8,10,12,14\}$、主表・全体表示は $K_0=8$ |
| 表現 | B0は256次元、PCA・AE・MAE系は16次元。いずれもクラスタリング前に行ごとにL2正規化 |
| PCA | `PCA(n_components=16)`、他は既定値。trainでfit、射影後にL2正規化 |
| ChemoMAE | seq_len=256、d_model=256、nhead=8、num_layers=8、dim_feedforward=1024、n_patches=16 |
| 潜在・decoder | latent_dim=16、latent_normalize=True、decoder_num_layers=1の線形写像 |
| 初期化・dropout | ChemoMAE v0.2.1既定初期化、dropout=0.0 |
| 学習 | 800 epoch、単一GPU、batch size=1024、勾配蓄積なし、AdamW、peak lr=$6\times10^{-4}$ |
| schedule | 40 epoch warmup、残り760 epoch cosine decay。batch処理前にlr更新 |
| 学習精度等 | FP16 AMP + GradScaler、FP32 weights、gradient clippingなし、EMAなし、最終raw weights使用 |
| Aug | noise角度 $U(0,2.5^\circ)$、shift $U(-2,2)$ チャネル。学習時は有効な操作ごとに適用確率0.5 |
| 抽出・評価 | 全可視、FP32、抽出AMPなし、評価TF32無効。通常抽出のAugなし |
| Cosine-KMeans | 初期化1回、max_iter=500、tol=1e-4、他は参照版既定。K・seedは共通計画に従う |
| 主評価 | LLA-3/5/9、noise・shift・両方のLFR。評価摂動は各種類 $R=5$、対象操作の適用確率1 |
| 補助報告 | cosine-silhouette、補正LLA、3反復間ARI、occupancyと使用クラスタ数 |
| 集約 | 試料macro。試料間SDと3反復間SDを区別し、同一試料のpairedな条件差を報告 |

## 1. データと実装単位の確認

- [x] `git status` と対象階層のAGENTS.mdを確認し、既存差分を保持する。
- [x] 本番入力 `data/processed/preprocessing/200hz_snr10_linear256/` の
  `manifest.parquet`、`sample_quality.parquet`、`config.json` とmetadataの対応を確認する。
  metadataは `data/metadata/古材メタデータ.csv`。採用試料ID・件数を明示する。
- [x] 同一原材など上位の採取関係を確認し、確認できた内容と不明点を記録する。
  新たな依存関係が判明した場合は、split作成前にユーザーと扱いを決める。
  2026-09-05ユーザー決定: 現存49試料で進める。異なるKYOw間の原材関係は不明。
  KYOw単位で分割し、同一KYOwがtrain/testをまたがないことを必ず検証する。
- [ ] 既存の前処理診断図と品質記録を確認する。保存済み本番データの再生成を前提にしない。
- [x] HDF5の `snv`、`reflectance`、`pixel_row_col`、`valid_spectrum_mask`、`wavelength_nm` の
  対応を、小規模な読み取りで確認できる検証処理を用意する。schemaやraw dataを変更しない。
  ユーザー実行ログでテスト26件成功、49試料・計392行のprobe成功を確認した。
- [x] 既存構成に沿って、実験用コード・config・manifest・checkpoint・数値結果・図の配置を整理する。
  新規配置は実装前に明示し、既存の保存規約・ignore規則と整合させる。
  数値データと図を分け、本番前処理を上書きしない。

### 第1段階の実装・確認記録（2026-09-05）

- 作業開始時のGit差分なし。依存関係・前処理データ・Git indexは変更していない。
- metadataの52件とHDF5のファイル名49件を照合した。49件すべてにmetadataが対応する。
  metadataだけにあるのは `KYOw02782`、`KYOw16743`、`KYOw16746`。
  metadataの数値KYOwを5桁ゼロ埋めし、`KYOw` prefixを付けて照合する。
- 採用対象の49 ID（ユーザー共有の入力検証CLIログでmanifestとの照合成功を確認済み）:

  ```text
  KYOw02702 KYOw02707 KYOw02708 KYOw02709 KYOw02715 KYOw02716 KYOw02717
  KYOw02719 KYOw02720 KYOw02751 KYOw02752 KYOw02754 KYOw02756 KYOw02758
  KYOw02759 KYOw02760 KYOw02762 KYOw02763 KYOw02764 KYOw02766 KYOw02767
  KYOw02768 KYOw02769 KYOw02770 KYOw02771 KYOw02772 KYOw02773 KYOw02774
  KYOw02775 KYOw02776 KYOw02777 KYOw02780 KYOw02783 KYOw02784 KYOw02787
  KYOw02788 KYOw02789 KYOw02790 KYOw16662 KYOw16666 KYOw16700 KYOw16702
  KYOw16711 KYOw16714 KYOw16716 KYOw16719 KYOw16737 KYOw16744 KYOw16750
  ```

- 保存済み `config.json` は200 Hz・schema 2・256点線形補間・SNV ddof=1。
  `preprocessing_summary.json` は49試料・3,902,746画素・除外0を記録している。
  ユーザー共有ログではCLIのstatusは `passed_table_and_sampled_row_checks`、
  試料数49、全有効画素数3,902,746、未検証HDF5試料0。
  49試料×8行のSNV再計算最大絶対誤差は `1.66173969695649e-7`。
- 既存のcutoff図とSNV異常候補図を目視した。上位候補にはspikeが見られるが、
  順位図から発生率や採用可否を判断しない。全試料の空間図・品質表の確認は未完了。
- 配置:

  | 用途 | 配置 |
  | --- | --- |
  | 実験実装・CLI・テスト | `src/wood_degradation_map/experiments/`、`scripts/experiments/`、`tests/experiments/` |
  | 実験config・seed計画 | `outputs/experiments/<experiment_id>/config/` |
  | fold・抽出manifest | `outputs/experiments/<experiment_id>/manifests/` |
  | 数値結果・run台帳 | `outputs/experiments/<experiment_id>/results/` |
  | 図 | `outputs/experiments/<experiment_id>/figures/` |
  | 重み・optimizer等の再開状態 | `outputs/experiments/<experiment_id>/checkpoints/` |

  実験出力ディレクトリは各工程の実装時に作成する。`outputs/`の一括ignoreを解除し、
  `checkpoints/`・`weights/`と既存の重み拡張子はignoreする。commitやLFS設定は行っていない。
  `git check-ignore`でcheckpoint内の任意拡張子・重み拡張子の除外と、図・CSVが除外されないことを確認した。
  追跡済みファイルの差分は `git diff --check` で問題なし。

ユーザーが実行済みの検証（2026-09-05共有ログ。Codexでの再実行はしていない）:

```powershell
uv run pytest tests/experiments/test_input_validation.py
uv run python scripts/experiments/check_inputs.py
```

CLIは既存のconfig、manifest、sample_quality、metadataと、各HDF5の既定8行を読む。
スペクトル全件走査・GPU処理・入力変更・出力ファイル生成は行わない。
圧縮HDF5では指定行の読み取りにも該当chunkの展開が必要となる。
確認点は終了code 0、`candidate_sample_count=49`、上記49 IDとmetadata-only 3 ID、
全49試料の `probes`、座標・mask・波長・SNVの検証成功。
このprobeは全画素でのmask網羅性や座標重複の不存在を保証しない。
fixtureテストは `26 passed in 4.49s`。同じ入力での再実行は不要。

完了条件: 採用データと座標の契約、分割上の確認状況、実装・保存先を説明できる。
49試料の採用確定と、画素数確認に49試料が載っていたことを区別する。

## 2. 共通config・seed・manifestの実装

- [x] 固定条件を共通configへ転記し、条件別の差分を明示する。
  B0、B1、A0、M00、M10、M01、M11とM11のmask率25%・75%を識別できるようにする。
- [x] split、画素抽出、モデル構築、画素順序、mask、学習Aug、PCA、KMeans、評価摂動の
  用途別seedと3反復IDの対応を事前固定・保存する。乱数消費順への依存で共通入力が変わらないようにする。
- [x] 試料単位5-foldの生成・読み込みと、試料内8,192画素の抽出・読み込みを実装する。
  同じfoldの抽出座標は全条件・全K・全反復へ共有し、epochごとには再抽出しない。
- [x] fold割当、試料ID、抽出seed、HDF5行との対応・元座標をmanifestへ保存する。
  既存manifestを暗黙に作り直さず、再開時にも同じ集合を利用する。
- [x] 小規模fixtureで、train/testの試料重複なし、各試料が一度だけtestになること、
  抽出数・非復元・有効座標・共有集合・再現性を検証する。
- [x] 確認済み実装で本番split・抽出manifestを生成し、全foldの件数を確認する。
  現行49試料ならtrainは39または40試料、319,488または327,680画素となる。

完了条件: configとmanifestから全runの入力・乱数設定を追跡できる。

### 第2段階の実装・確認記録

- `src/wood_degradation_map/experiments/config.py`:
  採用済み49 ID、主実験7条件・mask率補助2条件、固定recipe、用途別seed導出を実装。
  本番生成時に試料集合がこの49 IDと異なる場合は停止し、試料を暗黙に追加・除外しない。
  基準seedは `20260905`。数値は結果を見る前に固定し、探索による選び直しを行わない。
- `src/wood_degradation_map/experiments/manifests.py`:
  KYOw単位5-fold、非復元抽出、保存・再読込・検証を実装。
  生成は既存出力ディレクトリがある場合に停止する。再読込は既存のfold・画素集合を使う。
- `tests/experiments/test_manifests.py`:
  KYOwリーク、試料/画素の欠落・重複、無効座標、入力順序や他用途の乱数からの独立性、
  保存/再読込、既存出力の保護、入力・manifest変更の検出を対象とする。
  ユーザー共有ログで `20 passed in 3.70s` を確認した。
- 評価摂動seedは試料・摂動種類・摂動反復に対応し、条件・K・学習反復に依存しない。
  seed計画だけで摂動入力の一致を確認済みとは扱わない。実際の共通入力生成・再利用は第4段階で検証する。
- ソースHDF5の全件hashやスペクトル全件走査は行わない。
  生成・確認CLIは1試料ずつ座標とmaskを読み、抽出画素について対応を検証する。
  入力追跡には小規模なconfig/manifest/品質表/metadataのSHA-256とHDF5の容量・mtimeを使う。
  HDF5の容量・mtimeだけでスペクトル内容の完全な同一性を保証しない。

次のコマンドはユーザー実行済みで、テスト成功後の本番manifest生成・再読込に成功した。
CLIはGPU・学習を使用せず、`create`が新規のconfigとmanifestを保存し、`check`は読み取りだけを行う。

```powershell
uv run pytest tests/experiments/test_manifests.py
uv run python scripts/experiments/prepare_manifests.py create
uv run python scripts/experiments/prepare_manifests.py check
```

保存先は `outputs/experiments/cv_200hz_snr10_linear256_v1/`。

```text
config/experiment.json                 固定条件
config/seeds.json                      用途別seedの数値一覧
manifests/inputs.json                  入力の識別記録
manifests/folds.parquet                各KYOwのtest fold（各試料1行）
manifests/train_pixels/fold_<1..5>.parquet
                                      train試料・抽出seed・HDF5行・元座標
manifests/complete.json                完了記録・成果物hash・fold件数・library version
```

確認点は `create` のstatus `created`、`check` のstatus `validated_existing_manifest`、
試料数49、test試料数9または10、train試料数39または40、train画素数319,488または327,680。
同一KYOwは1つのtest foldにだけ割り当て、当該foldのtrain画素manifestには存在しない。
ユーザー共有ログで `created` と `validated_existing_manifest` を確認した。

| fold | train試料 | test試料 | train画素 | batch/epoch | 予定更新/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–4（各fold） | 39 | 10 | 319,488 | 312 | 249,600 |
| 5 | 40 | 9 | 327,680 | 320 | 256,000 |

このmanifestを再生成しない。予定更新数は学習完了回数ではない。

## 3. 学習・表現抽出・クラスタリングの実装

- [ ] 必要な画素をchunkで読むloaderを実装し、SNVを入力・clean targetとして扱う。
  ニューラルネット学習・PCA fit・KMeans fitに同じtrain抽出集合を渡す。
  `data.py` の共通loaderとPCAへの接続を検証済み。NN・KMeansへの接続と実行確認は未完了。
- [x] B0とPCAのfit/transform・正規化・保存/再読込を実装する。
  PCAのfit対象をtrainに限定し、実solverと乱数管理を記録する。
  ユーザーが新規テスト・fold 1の本番fit・checkの成功を報告。保存記録も確認した。
- [ ] ChemoMAEの共通構成と条件別Aug・mask・lossを実装する。
  A0は `n_mask=0` と `loss_region="all"`、MAEは `loss_region="masked"`。
  mask率25/50/75%の `n_mask` は4/8/12とする。
  `neural.py` に共通モデル・Aug・可視mask・clean targetのMSEを追加済み。CPUテスト待ち。
- [ ] 実験プロトコル第4.2節の学習recipeを実装する。
  ChemoMAEの既定Trainer・optimizer・schedulerとの次の差を解消する。
  - `amp_dtype="fp16"`、`grad_clip=None`、`use_ema=False`を明示する。
  - weight decayは0.05、bias・正規化層のみ0とし、既定helperによるCLS・位置埋め込みの除外をそのまま使わない。
  - 初回lr=0から、batch処理前に公式recipeのlr列を設定する。
  - 独立runは `resume_from=None`。再開は同じrunのcheckpointを明示する。
- [ ] 800 epochの最終raw weightsと、再開に必要な学習状態を保存する。
  test指標によるearly stopping・checkpoint選択を入れない。
- [ ] 全可視16次元抽出を実装する。`ExtractorConfig(amp=False)` または全可視maskを明示したencoderを使う。
  `eval()`だけを指定した通常の `forward()` でランダムmaskを残さない。
  `extract_full_visible` を追加済み。正規化前の潜在診断を含むCPUテスト待ち。
- [ ] trainだけでCosine-KMeansをfitし、中心を固定してtestをpredictする。
  同じ学習済み表現を全Kで再利用し、中心・seed・停止状況を保存する。
- [ ] 元座標へ予測を戻し、背景0・クラスタ1からKのラベル規約を検証する。
  非有限値・ゼロnorm・単位norm誤差の記録を実装し、条件別に画素を無言で落とさない。

完了条件: 小規模入力でfitから保存・再読込・全可視推論・元座標への対応まで確認できる。

### 第3段階のloader・B0/PCAの実装・確認記録

- `src/wood_degradation_map/experiments/data.py`:
  `FoldData`がmanifestの選択行をコピーして保持し、train抽出行とtest全行を区別して読む。
  `SpectrumBatch`は試料ID・HDF5行・元座標・clean SNVを持つ。augmentationは別のcopyへ適用する。
  SNVはFP32のまま、既定2,048行以下のsource windowで読む。trainで未選択の行は返さない。
  testの末尾chunkも除外しない。非有限値・ゼロSNV・座標/mask不一致は行情報付きで停止する。
- `src/wood_degradation_map/experiments/baselines.py`:
  B0の256次元とPCA後の16次元をChemoMAEの既存helper（eps=1e-6）でL2正規化する。
  非有限表現・非有限norm・ゼロnorm・epsilonにclampされる微小normは診断付きエラーとする。
  正常行の単位norm誤差は診断値として返し、経験的な新しい除外閾値は設けない。
  PCAはtrain行列だけで `PCA(n_components=16)` をfitし、実solverと用途別seedを記録する。
  `random_state=None`を維持し、同期的なfit中だけNumPy乱数を設定して終了/例外時に復元する。
  NumPyのglobal RNGを消費する他threadとの同時fitは行わない。
  transformでは追加fitを行わず、FP32の射影後に正規化する。
  `full`・`covariance_eigh`は反復間の再利用が可能な状態として記録する。
  確率的solverのcheckpointを別反復へ読み込もうとした場合は停止する。
- `tests/experiments/test_baselines.py`:
  共通train集合、test全行、chunk上限、test/未抽出train行がfitに入らないこと、
  RNG復元、B0/PCAの次元・正規化・保存/再読込、無効値の扱いを小規模fixtureで検証する。
  ユーザーからテスト成功の報告を受けた。件数・所要時間のログは未共有。

ユーザー実行済みのテスト（同じ内容での再実行は不要）:

```powershell
uv run pytest tests/experiments/test_baselines.py
```

ユーザー実行済みのfold 1のB0/PCA保存・再読込:
`fit`はCPUでtrain全抽出行を使った本番PCA fitを行い、train/testの先頭source windowの
最大8行ずつでtransformと保存/再読込を確認する。全test評価やクラスタリングは行わない。
PCA入力行列だけでfold 1–4は312 MiB、fold 5は320 MiBを必要とし、内部作業領域が別途必要。
圧縮HDF5では抽出行を含むchunkの展開が必要なので、実I/O量は返す画素数より大きくなる。
実行時間・最大メモリ使用量は未計測。

```powershell
uv run python scripts/experiments/fit_baselines.py fit --fold 1
uv run python scripts/experiments/fit_baselines.py check --fold 1
```

既存の実験root `outputs/experiments/cv_200hz_snr10_linear256_v1/` の下へ保存する。

| 保存先 | 内容 | Git |
| --- | --- | --- |
| `results/baselines/fold_1/repeat_1/b0.json` | B0の変換契約（学習パラメータなし） | 対象 |
| `results/baselines/fold_1/repeat_1/fit.json` | train ID・画素数・seed・実solver・version・probe診断・manifest/係数hash | 対象 |
| `checkpoints/baselines/fold_1/repeat_1/pca.npz` | PCA平均・基底など。pickleなしで保存/再読込 | 除外 |

確認点は `fitted_and_roundtrip_checked` / `validated_existing_baselines`、
train 39試料・319,488行、B0 256次元・B1 16次元、数値異常の各件数0。
保存済みmanifestと共通configは変更しない。既存baseline出力への上書きも行わない。
保存済み `fit.json` で、train 39試料・319,488行、実solver `covariance_eigh`、
scikit-learn 1.9.0・NumPy 2.4.4、PCA再読込後のprobe最大絶対誤差
`1.6391277313232422e-7`、数値異常の各件数0を確認した。
別途ユーザーから `check --fold 1` もエラーなく終了したと報告を受けた。
このprobe成功を全test推論・クラスタリングの成功とは扱わない。

### 第3段階のChemoMAE共通部品の実装記録（検証待ち）

- `src/wood_degradation_map/experiments/neural.py`:
  固定configからCPU FP32モデルを構築し、同じfold・反復の初期化を条件間で共有する。
  参照版の初期化を変更せず、モデル構築後に外部のCPU乱数状態を復元する。
  Augは既存 `SpectraAugmenter` を使用し、clean targetのcopyへFP32で適用する。
  参照版のpatch mask生成によりTrue=可視のmaskを作り、条件別mask数とloss領域を検証する。
  mask・Augのglobal RNGを用途別状態へ一時切替し、画素順序は別のCPU Generatorで管理する。
  例外時にも外部の乱数状態を復元する。同じprocess内で競合するthreadから同時使用しない。
  epochごとにshuffleし、固定batch size=1024で末尾をdropする。画素集合の再抽出は行わない。
- AdamWのgroup分けとbatch処理前に使うlr計算を追加した。
  bias・LayerNormのみweight decay=0、CLS・位置埋め込みには0.05を適用する。
  800 epochの実行loop、CUDA FP16 AMP + GradScaler、checkpoint保存・再開との接続は未実装。
- 全可視抽出は `model.eval()` と全要素Trueのmaskを指定してencoderだけを呼ぶ。
  FP32・autocast無効・TF32無効とし、終了/例外時にmode・精度設定を復元する。
  `to_latent` の正規化前出力を一時hookで検査し、参照版のeps=1e-12にclampされる
  微小norm・ゼロnorm・非有限値を行番号付き診断で停止する。抽出後の単位norm誤差も返す。
- `tests/experiments/test_neural.py`:
  固定モデルの構成・初期値共有・parameter数、条件別mask/loss、clean target保持、
  乱数分離・状態再生・例外時復元、weight decay区分、lr境界、全可視FP32抽出を対象とする。
  forward/backwardと抽出のfixtureだけTransformer幅・層数を縮小し、CPUで検証する。
  本番モデル構築は固定サイズで検査する。実データ・GPU・800 epoch学習・ファイル出力は使わない。

次の確認コマンド（Codexでは未実行）:

```powershell
uv run pytest tests/experiments/test_neural.py
```

共通部品を確認した後、train loaderとの接続・実学習loop・checkpoint・KMeansを実装する。
本番batch size=1024のGPUメモリ適合性とCUDA乱数の再開再現性は未検証。
保存済みconfig・manifest・PCA係数は変更していない。

## 4. 評価・集計の実装

- [ ] 評価文書の式に従いLLA-3/5/9と補正LLAを実装する。
  有効近傍対の計数、背景・中心画素の除外、coverage、単一クラスタ時の未定義を確認する。
- [ ] LFR用に、SNV入力へnoise・shift・両方を各5回適用する。
  同一画素の15入力を全条件・全K・全学習反復で共有できる生成・再利用処理を実装する。
  評価Augは明示的にtraining mode、encoderは推論mode・全可視とする。
  摂動後にPCAや中心をfitし直さず、各5回のLFRと平均を保存する。
- [ ] `silhouette_samples_cosine_gpu` でfoldの全test画素のscoreを計算し、試料ごとに平均する。
  FP32、singleton、使用クラスタ数1、ゼロ距離などの規約を確認する。
- [ ] train fit・clean test・摂動後のoccupancy、使用クラスタ数、最大占有率を保存する。
  試料別とfold全体を区別し、未整列クラスタ番号をfold間で平均しない。
- [ ] 同一試料の3反復間ARIを3対計算し、退化flagとともに保存する。Hungarian matchingは使用しない。
- [ ] 試料・fold・条件・K・反復・摂動種別を追跡できる数値出力とrun台帳を実装する。
  未定義理由、利用可能数、失敗・中断・完了状態を記録する。
- [ ] 試料macro、試料間SD、3反復間SD、共通対象でのpaired contrastを実装する。
  ARIの3対を独立反復扱いせず、欠測を0で補完しない。
- [ ] 手計算可能な小ラベルマップ・小スペクトルで指標を検証する。
  背景、孤立画素、単一クラスタ、ラベル番号置換、摂動前後の完全一致、未定義値の集約を含める。

完了条件: 小規模の一連の処理から、期待する指標と集約表が得られ、同一画素対応と評価の固定性を確認できる。

## 5. 本学習前の動作確認

- [ ] 対象実装の最小テストと関連チェックのコマンドを用意し、結果を確認する。
- [ ] 少数batchの学習から全可視抽出・クラスタリング・評価・保存まで通す。
  動作確認runを本実験の結果と分け、短縮学習の値で条件選択をしない。
- [ ] batch size=1024で、全可視A0を含めGPUメモリ・入出力負荷を確認する。
  収まらない場合は状況を報告し、batch size・accumulationを暗黙に変更しない。
- [ ] lr列、weight decay対象、FP16学習とFP32評価の切替、更新回数、再開動作を確認する。
- [ ] GPU・library version・演算設定、実行時間・保存容量の見積りを記録する。
  同一モデルのclean/摂動後表現をK間で再利用できることも確認する。
- [ ] 本文用の代表試料の選択基準・IDを、結果を見る前にユーザーと固定する。
  任意の形状診断を採用する場合は、この時点までに定義を決める。未定義のまま実装・評価しない。

完了条件: 固定configとmanifestで本実験を開始でき、実行コマンド・出力先・所要時間の見通しがある。
本実験開始時には実行対象と負荷をユーザーへ提示する。

## 6. 主実験・補助実験の実行

| 工程 | 条件 | ニューラルネット学習数 |
| --- | --- | ---: |
| 主実験 | A0、M00、M10、M01、M11 × 5-fold × 3反復 | 75 |
| baseline | B0、B1の5-fold・3反復のクラスタリングと評価 | 0 |
| mask率補助実験 | M11-25、M11-75 × 5-fold × 3反復 | 30 |
| 全体解釈用学習 | M00、M11を全試料で各1回 | 2 |

- [ ] B0・B1と主実験5学習条件を、全fold・3反復・全7個のKで完了する。
- [ ] M11-25・M11-75を同じsplit・画素・反復計画で完了する。
  M11-50は主実験のM11を再利用する。
- [ ] 各runについて800 epoch完了、checkpoint、予定・実更新回数、AMP skip回数、実行時間を記録する。
  現行49試料なら312または320 batch/epoch、249,600または256,000予定更新/runとなる。
- [ ] OOF結果の試料・条件・K・反復の欠落や重複、未定義・失敗理由を確認する。
  失敗を隠したり別seedの良好なrunで置き換えたりせず、再開・修正履歴を残す。
- [ ] 3反復完了後にARIと集約結果を生成する。

完了条件: CVの105学習とB0/B1の評価が揃い、全Kの結果をrun台帳から追跡できる。
PCA・KMeans fitや評価計算は上表のニューラルネット学習数に含めない。

## 7. 論文用の定量結果の整理

- [ ] $K_0=8$の主評価表と補助診断表を作る。有効対象数・未定義理由を併記する。
- [ ] M11 vs B0、M11 vs B1、M11 vs M00のpaired plotと条件差を作る。
- [ ] 残る計画比較と2×2交互作用 $(M11-M10)-(M01-M00)$ をablation表へまとめる。
- [ ] 全7個のKの曲線、反復別曲線、mask率25/50/75%の感度解析を作る。
- [ ] LLAとLFRのtrade-off、occupancyによる退化、試料差と反復差を確認する。
  silhouetteで手法を総合順位付けせず、foldを独立反復とした検定・信頼区間を追加しない。

完了条件: 評価文書第8節の必須表・図が揃い、主張が全K・試料差・反復差と対応している。

## 8. 全体学習・探索的解釈

- [ ] 全体学習用に各試料8,192画素の共通抽出集合とmanifestを用意する。
- [ ] B0、B1、M00、M11を全試料でfitする。M00/M11は同じ800 epochで各1回学習する。
  事前固定の反復ID 1相当のseedと $K_0=8$ を使い、CVで良かった条件・seedへ変更しない。
- [ ] 全有効画素に推論し、全試料のマップを保存する。
- [ ] 全共通有効画素からB0を基準にHungarian matchingを行い、対応表・overlapを保存する。
  試料ごとのmatchingやCV指標への転用をしない。
- [ ] 実測反射率・SNVについて、試料内クラスタのband別中央値から試料間中央値・四分位範囲を作る。
  寄与試料ID・数・画素数を記録し、差スペクトルの引き算の向きを明示する。
- [ ] 事前指定した本文例と全試料の補助図を用意する。figure title・axes titleを付けずcaptionで説明する。
- [ ] 材組織・表面状態・劣化との対応を探索的解釈として記述する。
  CV指標を劣化検出精度に置き換えず、代替説明と未確認事項を明示する。

完了条件: 定量評価と探索的解釈を区別した図表・説明、および全体学習の記録が揃う。

## セッション終了時の引き継ぎ

- [ ] 完了したチェック項目と、変更ファイル・実行したコマンド・結果を記録する。
- [ ] 実行していない検証、失敗・中断run、再開対象checkpointを明示する。
- [ ] config・manifest・結果の保存先と、次回最初に行う作業を記録する。
- [ ] 設定変更が必要になった場合はユーザーの決定を設計文書へ反映し、旧条件のrunと混合しない。

次回の開始位置: **3. loader・B0/PCAの新規テストとfold 1のfit/再読込結果確認**。
その確認後、**3. NNの学習・全可視抽出・KMeansの実装**へ進む。
第1段階の品質表・空間図確認は残っており、本学習前までに完了する。
49試料の採用とKYOw単位の分割は確認済み。原材関係の再確認は新情報がある場合に限る。
