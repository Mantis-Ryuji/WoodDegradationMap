# 実験実施 ToDo

更新日: 2026-09-06

第1段階のfixtureテストと入力照合はユーザー実行ログで成功を確認済み。
第2段階は20テストと本番manifestの生成・再読込成功をユーザー実行ログで確認済み。
第3段階のloader・B0/PCAはユーザー実行で検証済み（本番PCAはfold 1）。
ChemoMAEの共通部品・全可視抽出は、ユーザー共有ログでCPUテスト29件の成功を確認済み。
Trainer継承部分はCPUテスト16件とA0/M11のGPU動作確認で成功を確認済み。
Cosine-KMeans・元座標へのラベル復元はCPUテスト15件の成功を確認済み。
重み読込・全Kのクラスタリング・clean testマップ保存の接続はCPUテスト20件で確認済み。
B0/B1/A0/M11のGPU上のtrain 64画素probeも成功。
第4段階のLLA・補正LLAはCPUテスト33件で確認済み。
LFR用の共通摂動生成・反転率計算はCPUテスト27件で確認済み。
cosine-silhouetteと3反復間ARIはCPUテスト33件で確認済み。
試料macro・SD・paired差はCPUテスト34件、評価接続と既存LFRは計52件で確認済み。
OOF・ARI・計画比較の接続もユーザーから31テスト全件成功の報告を受けた。
品質確認レポートはCPUテスト10件と生成成功をユーザー共有ログで確認済み。
保存済み品質表と全49試料の空間図を確認した。
ユーザー決定により負の補間反射率画素をtrain・test共通で背景化する実装を追加した。
新版前処理のテスト全件成功・再生成はユーザー報告と保存記録で確認済み。
実験の入力IDを `production_v1`、動作確認用rootを `preflight_v1` に更新した。変更後の実験テスト・再生成待ち。
本番用rootは `production_v1` とし、動作確認とは分離する。
ユーザー指定によりデータ階層を `data/processed/production_v1/` へ簡略化した。
旧階層での生成確認と区別し、短いパスでの前処理再生成・preflight再構築を次に実行する。
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
- [x] 既存の前処理診断図と品質記録を確認する。保存済み本番データの再生成を前提にしない。
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
  順位図から発生率や採用可否を判断しない。全試料の空間図・品質表の後続確認は下記に記録する。
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

### 本学習前の品質確認資料（2026-09-06確認済み）

- `input_review.py` と `scripts/experiments/prepare_input_review.py` を追加。
  既存のmanifest、sample_quality、mask_quality、reference_band_quality、output_band_summaryを読み、
  全列CSVとHTML表にする。config・cutoff・preprocessing_summary・report_configも表示する。
  49試料分の反射率L2 norm画像と4種類の全体診断図は、元画像へのリンクで一覧表示する。
  図の再生成・色scale変更・HDF5読込・スペクトル再計算・raw data変更は行わない。
- 小さなSNV数値誤差を表示上ゼロへ丸めない。欠損値は欠損として表示し、0で補完しない。
  表の試料・画素数、summaryの件数・cutoff記録、図のreport_config出典を照合する。
  CLIは採用済み49 IDとの一致を要求する。画像欠落は一覧と記録に残し、試料を自動除外しない。
- 保存先は `results/input_review/<UTC時刻>/` の `index.html`、5つのCSV、`review.json`。
  `--output-dir` で新規保存先を明示できる。既存の保存先は上書きしない。
  `review.json` に小規模な出典表・設定のhashと画像欠落一覧を記録する。
  元画像はコピー・hash・再解析しないため、図とデータの対応を独立に再検証したとは扱わない。
  statusは `input_review_prepared`、`manual_review_required=true` であり、品質承認を意味しない。
- テスト10件を追加。全列/数値/リンクの保持、HTML escape、欠落画像の明示、出典不一致、上書き拒否を確認する。
  既存の品質基準・49試料の採用決定・splitは変更していない。

ユーザー実行ログで成功を確認したコマンド（Codexでは再実行していない）:

```powershell
uv run pytest tests/experiments/test_input_review.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

テストは `10 passed in 1.63s`。続く資料生成も成功した。
既存の小規模表を読むCPU処理で、GPU・HDF5全件走査・前処理再生成は行わない。

```powershell
uv run python scripts/experiments/prepare_input_review.py
```

保存先は `results/input_review/20260905T184525_158100Z/`。
ユーザーから49試料・3,902,746画素・画像欠落0件で問題ないとの確認を受けた。
Codexは生成済みCSVと既存PNGを読み取り、次を確認した（HDF5全件再検証ではない）。

- 保持帯域反射率の非有限値、SNV入力標準偏差不正、最終除外画素はいずれも0。
  全49行でmask画素数と保存画素数・除外画素数の収支が一致する。
- 全元帯域でのSNV不正は1,390画素（KYOw02708: 519、KYOw02771: 50、KYOw02784: 821）。
  cutoff前の全帯域診断であり、保持帯域・補間後の最終除外画素数とは区別する。
- 負の補間反射率を1帯域以上含む画素は21試料の計496画素（全保存画素の約0.0127%）。
  1超の反射率を含む画素は0。現行実装は非クリッピング・負値だけでは除外しない仕様。
  後続のユーザー決定により、補間後・SNV前の負値画素をtrain・test共通で背景化する新版を用意した。
- reference表は256帯域中222帯域を保持し、保持帯域のlow_snrは0件。
  output表は反射率・SNV各256帯域、各行49試料、最大非有限値率0。
- SNV二次差分最大値の上位にはKYOw02769、KYOw02771、KYOw02715などがある。
  事前の除外閾値はなく、上位候補という理由で試料や画素を除外しない。
- 全49試料の反射率L2 norm画像を目視した。分離した材片（例: KYOw02702、KYOw16702）、
  穴や亀裂状の空白、局所的に明るい領域（例: KYOw02775中央の文字状領域）が見られる。
  原画像とのoverlay照合は行っておらず、空白の原因・maskの正確さ・明るい領域の材質や劣化との対応は確定しない。
  画像は既存の全試料共通scaleで確認し、色scaleや画素集合を変更していない。

品質記録・空間図の確認工程を完了とする。全画素の科学的妥当性の保証や劣化ラベルの承認とは扱わない。
49試料の採用決定は維持する。旧データ・画素manifestは保存し、新版で有効画素数を再確認する。
現行CLIは動作確認と本番で同じexperiment rootを既定値にし、配下の枝だけを分けている。
本番は動作確認済みrootとは別の新規rootを使う方向で、入力方針の確定後に設定・コマンドを整備する。
本文代表試料の基準・IDはユーザーと結果を見る前に固定する工程として残っている。

### 負の反射率画素の背景化（2026-09-06、実装済み・ユーザー検証待ち）

- ユーザー決定: 補間後256帯域のSNV前反射率に1つでも負値がある画素を、train・test共通で背景にする。
  `valid_spectrum_mask=0`、後段のラベル0とし、学習抽出候補・評価対象から外す。
  元の形態学的maskは保持し、除外座標と理由code 3を保存する。SNV後の負値は除外理由にしない。
- 既存理由1・2を優先し、理由3の除外数をsample_qualityとsummaryに追加する。
  保存反射率の負値画素数は0になる。値のclip、閾値緩和、試料除外は行わない。
- 新版の前処理IDは `production_v1`。条件はconfigへ記録し、ディレクトリ名に列挙しない。
  新規の前処理データ・図を生成後、ユーザーが旧前処理データ・図のディレクトリを削除した。
  旧実験rootの成果物は旧入力に依存する記録であり、新版で再利用しない。
- 前処理テストに、厳密な負値判定、0のband・1超の値の保持、既存理由の優先順位と、
  小規模ENVI fixtureからHDF5のmask・座標・SNV・品質集計までの接続を追加した。
  接続fixtureはmaskを固定し描画を省略する。実データ・GPU・全49試料の代わりにはしない。
  Codexではプロジェクトコード・テストを実行していない。

ユーザーが実行済みのコマンド（再実行不要）:

```powershell
uv run pytest tests/preprocessing/test_production_preprocessing.py tests/experiments/test_input_validation.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
uv run python scripts/preprocess/run_production_preprocessing.py
```

前処理は49試料のrawを再走査するCPU処理。新しいHDF5一式と診断図を保存するため、
旧HDF5計約5.2 GiBと同程度の追加容量に加え図・表の余裕が必要。所要時間は未計測。
再生成後のsummaryを確認した: 49試料、mask画素3,902,746、保存画素3,902,250、除外496（全件理由3）。
HDF5計5,574,878,801 bytes、各試料HDF5と空間図は各49件。保持222帯域・除外34帯域のcutoffは同じ。
configには補間後・SNV前の負値をtrain/test共通で背景化する規則が記録されている。
旧前処理データ・図のディレクトリ不存在を確認した。旧レビューの画像リンクは参照先削除により使えない。
この確認はJSON・ファイル一覧の読み取りであり、新版HDF5の全画素再検証や新版49図の目視とは区別する。

実験configの入力IDを `production_v1` に更新した。各CLIの既定入力も新版へ揃えた。
動作確認rootは `outputs/experiments/preflight_v1/`、本番rootは `outputs/experiments/production_v1/`。
CLIの既定実験rootはpreflightとし、本番は `--experiment-dir`（manifest生成は `--experiment-id`）で明示する。
同じ49 KYOw・split seedでfold割当を維持し、背景を除いた新しい行番号で共有train画素manifestを新規生成・照合する。
旧入力に依存したPCA・学習重み・クラスタ中心・マップ・評価値は新版へ流用しない。

### production_v1入力でのpreflight再構築（今回の実行手順）

データ保存先を `data/processed/production_v1/` に簡略化し、前処理の保存先検証、
各CLIの既定入力、テストfixture、前処理文書を更新した。診断図は `outputs/preprocessing/production_v1/`、
動作確認は `outputs/experiments/preflight_v1/`、本番実験は `outputs/experiments/production_v1/`。
今回の短いパスでの再生成はまだ実行していない。過去の生成結果を今回の実行済み結果とは扱わない。

入力検証にproduction_v1の背景化config照合と、HDF5 probe行の反射率非負チェックを追加した。
対応する5テストを追加し、manifest fixtureの入力IDも更新した。以下の変更後テストはCodexでは未実行。
各コマンドを1つずつ実行し、失敗したら後続へ進まない。

```powershell
uv run pytest tests/preprocessing/test_production_preprocessing.py tests/experiments/test_input_validation.py tests/experiments/test_manifests.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
uv run python scripts/preprocess/run_production_preprocessing.py
uv run python scripts/preprocess/check_sampling_pixels.py --q 8192
uv run python scripts/experiments/check_inputs.py
uv run python scripts/experiments/prepare_manifests.py create
uv run python scripts/experiments/prepare_manifests.py check
uv run python scripts/experiments/prepare_input_review.py
```

確認点: 前処理49試料・保存3,902,250画素・理由3の除外496画素、全49試料でq=8192可能、
入力probe成功、manifest生成・再読込成功。
qと試料数が同じならtrain画素はfold 1–4で319,488、fold 5で327,680のまま、採用行・元座標は新版へ更新される。
`prepare_input_review.py`はmanifest作成後に実行する。先に実行すると実験rootが作られ、
既存ディレクトリを保護するmanifestのcreateが停止するため、この順序を守る。
新レビューの品質表で保存反射率の負値画素数0・理由3の合計496を確認する。
旧実験ディレクトリを移動・改名・コピーする必要はない。

続いてfold 1・repeat 1のPCAと、A0/M11の短縮学習・再開確認を新規実行する。
PCAは約32万train画素を使うCPU fit、ニューラルsmokeはGPU上で各6batch（再開probeを含む）。
どちらも新版入力を読み直す。これは本番800 epochやCV指標計算ではない。

```powershell
uv run python scripts/experiments/fit_baselines.py fit --fold 1
uv run python scripts/experiments/fit_baselines.py check --fold 1
uv run python scripts/experiments/train_neural.py smoke --condition A0 --fold 1
uv run python scripts/experiments/train_neural.py smoke --condition M11 --fold 1
uv run python scripts/experiments/cluster_representations.py smoke --condition B0 --fold 1
uv run python scripts/experiments/cluster_representations.py smoke --condition B1 --fold 1
```

PCAはfit・check成功、各smokeは `checks_passed=true` を確認する。
最後に、今回のニューラルsmoke出力にある `smoke_id` を文字列として代入する。旧IDを使用しない。
以下の値は今回の実行で生成されたID。再実行する場合はその出力のIDへ置き換える。

```powershell
$a0SmokeId = '20260905T194725_418246Z'
$m11SmokeId = '20260905T194855_710691Z'
uv run python scripts/experiments/cluster_representations.py smoke --condition A0 --fold 1 --neural-smoke-id $a0SmokeId
uv run python scripts/experiments/cluster_representations.py smoke --condition M11 --fold 1 --neural-smoke-id $m11SmokeId
```

ユーザーから全コマンド実行・`checks_passed=true`の報告を受けた。
新版入力でのB0/B1/A0/M11クラスタリングsmoke完了記録も確認した。
クラスタリングsmoke IDは順に `20260905T194937_420573Z`、`20260905T194946_230032Z`、
`20260905T195218_899327Z`、`20260905T195232_653899Z`。
GPU評価を含む動作確認・負荷確認と本文代表試料の事前指定は残る。
本番開始時はproduction_v1にmanifestを新規作成し、同じ入力・split・抽出集合を照合して本番fitを実行する。
preflightの成果物を本番結果としてコピーしない。本番開始のコマンドは残る確認が完了した時点で案内する。

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

- [x] 必要な画素をchunkで読むloaderを実装し、SNVを入力・clean targetとして扱う。
  ニューラルネット学習・PCA fit・KMeans fitに同じtrain抽出集合を渡す。
  `data.py` の共通loaderとPCA・NNへの接続を検証済み。
  KMeans用train表現の収集とNN重み読込からの接続もCPUテストで検証済み。
- [x] B0とPCAのfit/transform・正規化・保存/再読込を実装する。
  PCAのfit対象をtrainに限定し、実solverと乱数管理を記録する。
  ユーザーが新規テスト・fold 1の本番fit・checkの成功を報告。保存記録も確認した。
- [x] ChemoMAEの共通構成と条件別Aug・mask・lossを実装する。
  A0は `n_mask=0` と `loss_region="all"`、MAEは `loss_region="masked"`。
  mask率25/50/75%の `n_mask` は4/8/12とする。
  `neural.py` の共通モデル・Aug・可視mask・clean targetのMSEをCPUテストで確認済み。
- [x] 実験プロトコル第4.2節の学習recipeを実装する。
  ChemoMAEの既定Trainer・optimizer・schedulerとの次の差を解消する。
  - `amp_dtype="fp16"`、`grad_clip=None`、`use_ema=False`を明示する。
  - weight decayは0.05、bias・正規化層のみ0とし、既定helperによるCLS・位置埋め込みの除外をそのまま使わない。
  - 初回lr=0から、batch処理前に公式recipeのlr列を設定する。
  - 独立runは `resume_from=None`。再開は同じrunのcheckpointを明示する。
  CPUテスト16件と、A0/M11の短いGPU runで学習・保存・再開を確認済み。800 epochは未実行。
- [ ] 800 epochの最終raw weightsと、再開に必要な学習状態を保存する。
  test指標によるearly stopping・checkpoint選択を入れない。
  epoch境界のcheckpointと最終raw weights保存を追加済み。800 epochの実行は未実施。
- [x] 全可視16次元抽出を実装する。`ExtractorConfig(amp=False)` または全可視maskを明示したencoderを使う。
  `eval()`だけを指定した通常の `forward()` でランダムmaskを残さない。
  CPUテストとA0/M11のGPU上のtrain 8画素probeを確認済み。全test抽出は未実施。
- [x] trainだけでCosine-KMeansをfitし、中心を固定してtestをpredictする。
  同じ学習済み表現を全Kで再利用し、中心・seed・停止状況を保存する。
  `clustering.py` のtrain表現収集・参照KMeans fit/predict・中心保存/再読込をCPUテストで確認済み。
  接続のCPUテスト20件、B0/B1/A0/M11のGPU train 64画素probeも成功。本番全test推論は未実行。
- [x] 元座標へ予測を戻し、背景0・クラスタ1からKのラベル規約を検証する。
  非有限値・ゼロnorm・単位norm誤差の記録を実装し、条件別に画素を無言で落とさない。
  `LabelMap` のchunk追加・重複/背景への誤代入検出・有効画素の全件coverage確認はCPUテストで成功。
  本番test全画素の保存・再読込は未実行。

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

### 第3段階のChemoMAE共通部品の実装・確認記録

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
  800 epochの実行loop、CUDA FP16 AMP + GradScaler、checkpoint保存・再開との接続は、
  下記のTrainer継承部分へ追加し、A0/M11の短いGPU runで動作確認済み。
- 全可視抽出は `model.eval()` と全要素Trueのmaskを指定してencoderだけを呼ぶ。
  FP32・autocast無効・TF32無効とし、終了/例外時にmode・精度設定を復元する。
  `to_latent` の正規化前出力を一時hookで検査し、参照版のeps=1e-12にclampされる
  微小norm・ゼロnorm・非有限値を行番号付き診断で停止する。抽出後の単位norm誤差も返す。
- `tests/experiments/test_neural.py`:
  固定モデルの構成・初期値共有・parameter数、条件別mask/loss、clean target保持、
  乱数分離・状態再生・例外時復元、weight decay区分、lr境界、全可視FP32抽出を対象とする。
  forward/backwardと抽出のfixtureだけTransformer幅・層数を縮小し、CPUで検証する。
  本番モデル構築は固定サイズで検査する。実データ・GPU・800 epoch学習・ファイル出力は使わない。

ユーザー実行済みの確認コマンド（2026-09-06共有ログ。Codexでの再実行はしていない）:

```powershell
uv run pytest tests/experiments/test_neural.py
```

`29 passed, 8 warnings in 10.56s` を確認した。
警告は参照版の `norm_first=True` によりNested Tensorの最適化が無効になる通知。
固定モデル設定の変更や警告の抑制は行わない。
train loaderとの接続・実学習loop・checkpointは下記へ追加した。
本番batch size=1024のGPUメモリ適合性とCUDA乱数の再開はA0/M11の短いrunで確認した。
保存済みconfig・manifest・PCA係数は変更していない。

### 第3段階のTrainer継承・保存・再開の実装・確認記録

- `src/wood_degradation_map/experiments/training.py`:
  ユーザーと合意したとおり、`ExperimentTrainer` はChemoMAE 0.2.1の `Trainer` を継承する。
  `Trainer.fit()` のepoch管理、AMP/GradScalerの初期化、loss計算、atomicなtorch保存を再利用する。
  `train_one_epoch()` では、検証済みのAdamW group・batch処理前のlr・用途別乱数を接続する。
  FP16 AMP、FP32 weights、clippingなし、EMAなし、追加schedulerなしとする。
  `TrainingData.from_fold()` は共通loaderのtrain抽出行列だけをCPUに保持する。
  GPUへ転送するのは各batchの1024行であり、test spectraは学習時に読まない。
- epoch境界でmodel・optimizer・GradScaler・pixel order/mask/AugとCPU/CUDAのtorch RNGを保存する。
  最初にepoch 0のcheckpointも保存する。中断したepochの途中状態は保存せず、最後に保存できた
  epoch境界からやり直す。checkpoint内の履歴を正とし、先行した外部履歴は再開後の保存で置き換える。
  独立runの出力先が既にある場合は停止し、自動resumeしない。
  再開には同じrunのcheckpointを明示し、条件・fold・反復・config・manifest・実装hash・
  library/GPU/演算設定を照合する。GradScalerの復元エラーは握りつぶさない。
  実装や実行環境が変わったcheckpointの移行は、この再開処理の対象外。
- optimizerのstep hookで実更新回数を数え、予定回数・実更新・AMP skip・非ゼロlrでの更新を区別する。
  epoch別loss/lr/scale/時間、入力・seed・環境・checkpoint/最終重みhashを数値記録へ保存する。
  初回lr=0のoptimizer呼び出しも実stepに数え、非ゼロlrでの更新数は別に記録する。
- `scripts/experiments/train_neural.py`:
  `train` は単一CUDA GPUで800 epochに固定する。epoch数・batch size・accumulationの変更引数は設けない。
  `smoke` は別の出力先で短縮epochを2回実行し、epoch 1のcheckpointからepoch 2を再実行する。
  既定では1短縮epochあたり2batchなので、再実行を含め合計6batch。
  lrの分母には本番の312/320 batchを使うが、短縮epochの結果は本学習結果として扱わない。
  画素順序・Aug入力・mask・lr・scaler・更新判断の一致、raw weightsの保存/再読込、
  再開後の重みと全可視表現の誤差、GPUメモリを記録する。probeはtrainの8画素。
  重み/潜在の再開比較には最大絶対誤差1e-6を動作確認の許容値として使い、画素除外には使わない。
  `checks_passed` には非ゼロlrでの実更新が1回以上あることも必要。全AMP skipを成功扱いしない。
- `tests/experiments/test_training.py`:
  A0/M11の継承fit、連続実行と再開の一致、clean target保持、更新/skip計数、
  既存出力の保護、途中epochの保存拒否、条件/反復/manifest/code不一致、
  不正なcheckpoint・GradScaler復元失敗をCPUの小規模fixtureで検証する。
  このfixtureだけbatch sizeとTransformer幅・層数を縮小する。実データやGPUは使わない。

ユーザー実行済みのCPUテスト（Codexでの再実行はしていない）:

```powershell
uv run pytest tests/experiments/test_training.py
```

ユーザー実行済みのA0とM11のGPU動作確認:
本番サイズのモデルとbatch size=1024を使用し、それぞれ既定6batchを処理する。
train入力行列はfold 1で312 MiBのCPUメモリを使い、HDF5読み取り・モデル/optimizer・
checkpoint再読込・再開比較用の領域が別途必要。今回の実測値は下表のとおり。
OOMが発生してもbatch sizeなどを自動変更しない。エラーと保存できた実行記録を確認する。

```powershell
uv run python scripts/experiments/train_neural.py smoke --condition A0 --fold 1
uv run python scripts/experiments/train_neural.py smoke --condition M11 --fold 1
```

出力rootは既存の `outputs/experiments/cv_200hz_snr10_linear256_v1/`。

| 用途 | root以下の配置 |
| --- | --- |
| 本学習の数値記録 | `results/neural/<condition>/fold_<f>/repeat_<r>/` |
| 本学習の重み | `checkpoints/neural/<condition>/fold_<f>/repeat_<r>/` |
| 動作確認の数値記録 | `results/neural_smoke/<UTC実行ID>/<condition>/fold_<f>/repeat_<r>/` |
| 動作確認の重み | `checkpoints/neural_smoke/<UTC実行ID>/<condition>/fold_<f>/repeat_<r>/` |

数値記録は `run.json`、`training_history.json`、`checkpoint.json`、`completion.json`、
`attempt_<UTC>.json`。動作確認は追加で `smoke.json` を保存する。
重みディレクトリ内は参照Trainerの構造に合わせ、再開状態が `checkpoints/last.pt`、
最終raw weightsが `last_model.pt`。動作確認は `smoke_model.pt` と再開確認用 `checkpoints/epoch_1.pt`。
数値記録はGit対象、重みディレクトリ以下は既存ignoreで除外される。
動作確認は実行ごとに新しいUTC実行IDを付け、前の結果を上書きしない。

確認点は終了code 0、`checks_passed=true`、再開入力/lr/scaler/更新判断の一致、
raw weightsの保存/再読込一致、重み/潜在の最大絶対誤差、実更新・AMP skip数、GPUメモリ。
この段階ではKMeans・全test推論・評価指標は未接続。

2026-09-06ユーザー共有ログで、CPUテスト `16 passed, 16 warnings in 5.56s` を確認した。
警告は既知の `norm_first=True` によるNested Tensor最適化の無効化通知。
保存済み `run.json` のGPUはNVIDIA GeForce RTX 4070 Ti SUPER、PyTorchは2.13.0+cu130。

| 条件 | smoke ID（UTC） | 実行全体秒 | 最大GPU allocated | 最大GPU reserved |
| --- | --- | ---: | ---: | ---: |
| A0 | `20260905T154032_240426Z` | 16.20 | 約1.49 GiB | 約1.78 GiB |
| M11 | `20260905T154054_126680Z` | 15.67 | 約0.90 GiB | 約1.09 GiB |

両条件で `checks_passed=true`、本体4回のoptimizer step・うち非ゼロlrで3回・AMP skip 0。
再開確認を含めた実処理は各6batch。画素順序・Aug入力・mask・lr・scaler・更新判断は一致した。
raw weightsの保存/再読込は一致し、再開後の重み・潜在の最大絶対誤差は両条件とも0。
全可視8画素probeの数値異常は0、単位norm最大誤差はA0が `1.1920928955078125e-7`、
M11が `5.960464477539063e-8`。同じ検証を再実行する必要はない。
短縮epochの結果から800 epochの所要時間・安定性・モデル性能を断定しない。

本学習用CLIも実装したが、第1段階の残作業と第5段階の確認を終えるまで本学習へ進まない。
以下は将来の実行・再開方法の記録であり、本学習の実行結果ではない。

```powershell
uv run python scripts/experiments/train_neural.py train --condition M11 --fold 1 --repeat 1
uv run python scripts/experiments/train_neural.py train --condition M11 --fold 1 --repeat 1 --resume outputs/experiments/cv_200hz_snr10_linear256_v1/checkpoints/neural/M11/fold_1/repeat_1/checkpoints/last.pt
```

### 第3段階のCosine-KMeans・ラベル復元の実装・確認記録

- `src/wood_degradation_map/experiments/clustering.py`:
  `collect_train_features()` は共通loaderのtrain抽出行だけを読み、各行の表現を一度収集する。
  NN学習のような末尾dropは行わない。fit時には全Kで同じ `TrainFeatures` を再利用できる。
  PCAのfold・train試料集合・画素数・反復再利用条件を照合し、transform中にfitは行わない。
  非有限値・ゼロnorm・参照epsilonにclampされる微小normは診断付きで停止し、画素を除外しない。
- K・seed・max_iter=500・tol=1e-4を固定し、ChemoMAEの `CosineKMeans.fit()` を1回呼ぶ。
  参照版既定の全体をdeviceへ転送するfitを使い、streaming方式やrestartへ変更しない。
  fit/predictはFP32・autocast/TF32無効。test predictでは固定中心を使う。
  train占有数、表現/中心の単位norm誤差、実行時間、version、deviceを記録する。
- 参照版は実反復数・停止理由を公開しないため、`iterations=None`、
  `stop_reason="not_exposed_by_ChemoMAE_0.2.1"` として保存する。収束したとは推測しない。
  参照版の `inertia_` は最後の中心更新前のE-stepに対応するため、最終保存中心に対する
  目的値を別途計算し、`reference_inertia` と `final_center_inertia` を区別する。
- 中心の保存/再読込は数値NPZ（pickleなし、既存ファイルへの上書きなし）。
  参照版のsave/load helperは中心を再正規化するため、値を変えずに保存し、
  読込時は参照moduleのbufferへ直接復元する。条件・fold・反復・K・seed・versionを照合する。
  この中心ファイルは、後続CLIで既存の `checkpoints/` 以下へ配置する。
- `LabelMap` は0..K-1の予測を元座標へ1..Kとして追加し、背景を0に保つ。
  chunk内/間の重複座標、範囲外座標、背景への誤代入、無効ラベルを拒否する。
  完了時にvalid maskの全有効画素が予測済みか確認し、未予測画素を背景扱いで隠さない。
- `tests/experiments/test_clustering.py`:
  B0/PCAのtrain限定収集と末尾保持、参照fit/predictとの一致、中心の保存/再読込、
  test入力で中心を変更しないこと、無効表現、全test行の座標復元とcoverageをCPU fixtureで検証する。

ユーザー実行で確認済みのコマンド（Codexでは再実行していない）:

```powershell
uv run pytest tests/experiments/test_clustering.py -vv -s --durations=0 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

2026-09-06共有ログで `15 passed in 3.28s`、最長のtest本体0.15秒を確認した。
最初の通常実行は22分42秒で中断（14件成功）したが、`-s`付きの再実行では全件成功。
出力取得・端末・一時ファイル周辺が候補であり、長時間化の原因は未確定。
今後の確認コマンドは当面 `-s` を付ける。KMeansの実験条件は変更していない。
GPU上のKMeansは下記の64画素probeで確認済み。本番test全画素のマップ保存、評価指標は未確認。
実反復数と停止理由の取得は、参照版のAPI制約として残っている。

### 第3段階の重み読込・クラスタリングCLIの接続・確認記録

- `src/wood_degradation_map/experiments/cluster_pipeline.py`:
  B0、保存済みPCA、ChemoMAE最終raw weightsを共通のtransformへ接続した。
  PCAの出典repeatは明示指定できるが、確率的solverの他repeat利用は禁止したまま。
  NNは条件・fold・repeat・train集合・config・manifest・学習実装hash・library version・
  完了epoch/更新数・重みhashを照合し、FP32のraw state_dictをstrictに読み込む。
  本番は800 epoch完了重みだけを使う。smoke重みは明示したIDでsmoke処理にだけ使える。
  学習済みrunの保存形式や、既存の学習実装hash対象ファイルは変更していない。
- train表現を一度収集して全7個のKで共有し、各Kの中心を保存・再読込して値とprobe予測を照合する。
  testはchunkごとに表現を一度抽出し、固定した全Kの中心で予測する。
  1試料ずつ全有効画素coverageを確認し、背景0・クラスタ1..Kのuint8マップを圧縮NPZへ保存する。
  数値結果の保存前後にクラスタ番号を整列しない。
- 出力先（既存experiment root以下）:
  `results/clustering/<condition>/fold_<f>/repeat_<r>/` に
  `run.json`、`fits.json`、`maps/<sample_id>.npz`、最後に `completion.json` を保存する。
  NPZは `labels_k2`、`labels_k4`、…、`labels_k14` の各画像配列を持つ。
  試料ID・fold・条件・反復・背景規約はrunとcompletionで対応付ける。
  中心は `checkpoints/clustering/<condition>/fold_<f>/repeat_<r>/centers_k<k>.npz` に保存する。
  数値結果・マップはGit対象、中心は既存ignoreで対象外。
  既存出力の上書き・暗黙の再fit・自動resumeはしない。
  途中失敗・中断は `failure.json` に記録し、`completion.json` がないrunは完了と扱わない。
- `scripts/experiments/cluster_representations.py`:
  `run` は単一GPUで本番train全抽出行とtest全有効画素を処理する。
  `check` はCPUで保存中心・マップ・hash・出典・coverageを検証し、fit・表現抽出・スペクトル読込をしない。
  NNのcheckでも重みは読込・検証する。元HDF5の座標/mask照合はmanifest loaderで行う。
  `smoke` は共通trainの先頭最大64行だけで抽出・全K fit・中心保存/再読込を確認する。
  testスペクトルは読まず、全testマップやCV指標を生成しない。
  `results/clustering_smoke/<UTC実行ID>/...` と対応する `checkpoints/` に分離して保存する。
- `tests/experiments/test_cluster_pipeline.py`:
  CPU人工データでB0/PCA/NNの接続、全Kでの表現共有、test末尾行・マップ保存/再読込、
  重みの出典不一致・不完全な学習・hash不一致、smokeと本番の分離、失敗時の記録を検証する。
  NN fixtureだけモデル幅・層数・batch設定を縮小し、実際の800 epoch学習は行わない。

ユーザー実行済みのCPUテスト（Codexでは再実行していない）:

```powershell
uv run pytest tests/experiments/test_cluster_pipeline.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

ユーザー実行済みのGPU確認（既存smoke重みを再利用し、再学習していない）:

```powershell
uv run python scripts/experiments/cluster_representations.py smoke --condition B0 --fold 1
uv run python scripts/experiments/cluster_representations.py smoke --condition B1 --fold 1
uv run python scripts/experiments/cluster_representations.py smoke --condition A0 --fold 1 --neural-smoke-id 20260905T154032_240426Z
uv run python scripts/experiments/cluster_representations.py smoke --condition M11 --fold 1 --neural-smoke-id 20260905T154054_126680Z
```

確認点: `clustering_smoke_completed`、`checks_passed=true`、
`centers_and_probe_labels_save_load_exact=true`、train probe行、GPU最大メモリと所要時間。
2026-09-06共有ログでCPUテスト `20 passed, 14 warnings in 8.78s` を確認した。
警告は既知の `norm_first=True` によるNested Tensor最適化の無効化通知。
GPUは4条件とも上記status・2つの真偽値が成功し、K=2/4/6/8/10/12/14を処理した。
全条件のprobeはKYOw02702の同じ64行。中心全値と先頭最大8行の保存/再読込予測は一致した。

| 条件 | clustering smoke ID（UTC） | 記録秒 | 最大GPU allocated（MiB） | 最大GPU reserved（MiB） |
| --- | --- | ---: | ---: | ---: |
| B0 | `20260905T171557_535419Z` | 0.80 | 8.32 | 22 |
| B1 | `20260905T171604_648514Z` | 0.74 | 8.14 | 22 |
| A0 | `20260905T171611_746388Z` | 1.00 | 34.65 | 48 |
| M11 | `20260905T171622_056599Z` | 1.00 | 34.65 | 48 |

記録秒は出典読込から出力保存までで、最初のmanifest検証時間を含まない。
同じ確認を再実行する必要はない。testスペクトル・testマップ・CV指標はこのGPU probeの対象外。
このsmokeでは本番train全行のKMeansメモリ・全test抽出の負荷まで確認したとは扱わない。
本番runではB0のtrain表現だけで312–320 MiB、16次元表現は19.5–20 MiBのCPU領域が必要。
GPUにはKMeansの正規化入力・一時配列が追加される。NN抽出chunkは既定1024行、FP32。
testマップは1試料分を全Kについて保持する。全test表現の一括保持はしない。

以下は将来の本番マップ生成・保存確認のコマンドであり、今回の即時実行対象ではない。
評価CLIの接続は後述の検証待ち。品質表・空間図確認と本学習前確認を先に完了する。

```powershell
uv run python scripts/experiments/cluster_representations.py run --condition B0 --fold 1 --repeat 1
uv run python scripts/experiments/cluster_representations.py check --condition B0 --fold 1 --repeat 1
```

## 4. 評価・集計の実装

- [x] 評価文書の式に従いLLA-3/5/9と補正LLAを実装する。
  有効近傍対の計数、背景・中心画素の除外、coverage、単一クラスタ時の未定義を確認する。
  `spatial_metrics.py` の計算部はユーザー共有ログで33テスト成功（0.17秒）を確認済み。
- [ ] LFR用に、SNV入力へnoise・shift・両方を各5回適用する。
  同一画素の15入力を全条件・全K・全学習反復で共有できる生成・再利用処理を実装する。
  評価Augは明示的にtraining mode、encoderは推論mode・全可視とする。
  摂動後にPCAや中心をfitし直さず、各5回のLFRと平均を保存する。
  `perturbations.py` と `lfr.py` の生成・共有・固定中心予測・試料内集計はCPUテスト27件で成功。
  本番評価CLIと数値結果保存を `evaluation_pipeline.py` に接続し、CPUテスト成功を確認済み。
- [ ] `silhouette_samples_cosine_gpu` でfoldの全test画素のscoreを計算し、試料ごとに平均する。
  FP32、singleton、使用クラスタ数1、ゼロ距離などの規約を確認する。
  `diagnostic_metrics.py` の参照関数呼出・試料境界による集計はCPUテストで成功。本番/GPUは未確認。
- [ ] train fit・clean test・摂動後のoccupancy、使用クラスタ数、最大占有率を保存する。
  試料別とfold全体を区別し、未整列クラスタ番号をfold間で平均しない。
- [ ] 同一試料の3反復間ARIを3対計算し、退化flagとともに保存する。Hungarian matchingは使用しない。
  `diagnostic_metrics.py` の画素照合・整数contingency・3対と試料内平均はCPUテストで成功。
  本番3反復のラベル読込・結果保存を `oof_pipeline.py` に接続し、CPUテスト成功を確認済み。
- [ ] 試料・fold・条件・K・反復・摂動種別を追跡できる数値出力とrun台帳を実装する。
  未定義理由、利用可能数、失敗・中断・完了状態を記録する。
- [x] 試料macro、試料間SD、3反復間SD、共通対象でのpaired contrastを実装する。
  ARIの3対を独立反復扱いせず、欠測を0で補完しない。
  `aggregation.py` の共通対象・未定義/失敗の記録と手計算例はCPUテスト34件で成功。
  OOFファイル読込・集計結果保存の接続もCPUテスト成功を確認済み。本実験は未実行。
- [ ] 手計算可能な小ラベルマップ・小スペクトルで指標を検証する。
  背景、孤立画素、単一クラスタ、ラベル番号置換、摂動前後の完全一致、未定義値の集約を含める。

完了条件: 小規模の一連の処理から、期待する指標と集約表が得られ、同一画素対応と評価の固定性を確認できる。

### 第4段階のLLA・補正LLAの実装・確認記録

- `src/wood_degradation_map/experiments/spatial_metrics.py`:
  `local_label_agreement(labels, valid_mask, k=...)` は1試料のラベルマップと本番有効maskを受け取る。
  maskを予測ラベルから推測せず、背景への割当や有効画素の予測欠落は入力エラーとして停止する。
  幅3/5/9の正方近傍で中心自身と背景を除き、画像境界のwraparoundやpaddingによる近傍を作らない。
  近傍対は式どおり有向で数え、分子と分母を整数で合計してからFP32で一致率を計算する。
  画素ごとの局所一致率を単純平均しない。
- `LLAResult` は各クラスタの画素数・占有率、使用クラスタ数、最大占有率、帰無一致確率Pと、
  各幅の `LLAWindowResult` を返す。孤立画素も占有率の分母から落とさない。
  `pixels_with_neighbors` と `neighbor_pixel_fraction` は、少なくとも1つ有効近傍を持つ中心の
  件数と全有効画素に対する割合。coverageの診断であり、LLAの分母を置き換えない。
  未定義scoreはNone（JSON null）と理由を返す。空の有効画素集合は入力失敗として例外にする。
- 補正値は占有数による非復元の偶然一致確率を使い、負値をclipしない。
  N<2・近傍対なし・単一クラスタについて、該当する未定義理由をすべて記録する。
  近傍対がある単一クラスタの未補正LLAは1、補正LLAは未定義。
  補正式を整数件数の差と最後のFP32除算へ等価変形し、丸めたPが1となる場合にも
  多クラスタを単一クラスタと誤判定しない。epsilonや新しい除外閾値は追加しない。
- `tests/experiments/test_spatial_metrics.py`:
  手計算可能な境界・交互ラベル・対角・孤立画素・単一画素、ラベル置換・転置・背景余白への不変性、
  不正入力、JSONの未定義値、極端な占有率を検証する。
  小マップの全有効画素対を独立に列挙し、全3幅の整数件数と有理数による期待値を照合する。
  極端な占有率の確認は件数だけを使い、巨大配列は生成しない。
- 計算はNumPy/CPUで、画像面積×近傍offset数に比例する処理と画像面積に比例する一時領域を使う。
  入力・乱数状態は変更せず、GPU・スペクトル読込・fit・ファイル保存はしない。
  この段階では保存マップとの一括接続、LFR、silhouette、ARI、試料間集計を追加していない。

ユーザー実行済みの最小テスト（Codexでは再実行していない）:

```powershell
uv run pytest tests/experiments/test_spatial_metrics.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

確認点は全件成功と、未定義が0/1で補完されず理由付きで保持されること。
実データ・GPU・学習を使わない。ユーザー共有ログで `33 passed in 0.17s` を確認済み。
同じテストの再実行は不要。

### 第4段階のLFR共通摂動・反転率の実装・確認記録

- `src/wood_degradation_map/experiments/perturbations.py`:
  `SharedPerturbations` は1試料の全保存行をHDF5行順で受け取り、noise・shift・bothを各5回生成する。
  seedは既存の試料ID・種類・drawの計画をそのまま使用し、条件・K・学習反復を含めない。
  `SpectraAugmenter` を明示的にtraining modeにし、対象操作の確率1・それ以外0で呼ぶ。
  強度・batch内の操作順ランダム化・各操作後の再中心化/再正規化は固定済み設定を維持する。
- 参照実装ではbatch境界が乱数割当に影響するため、生成幅を1024行に固定し、試料先頭から区切る。
  これは学習batch sizeとは別の生成処理の規約。末尾の部分batchも処理する。
  loaderの読み取りchunkをbufferで組み直し、変えても同じ生成batchにする。
  種類/drawごとに独立した連続RNG streamを持ち、全条件の共通入力としてchunkごとに再利用する。
  途中欠落・重複・逆順・試料混入は拒否し、非有限/ゼロ/epsilon-clamp対象は元HDF5行付きで停止する。
- 生成結果はcleanと15摂動のFP32 CPU配列を持つread-only block。全試料分をlistや永続cacheにしない。
  1 blockのスペクトル配列は最大16 MiB程度で、生成時のGPU・copy・計算一時領域は別途必要。
  同じblockを全条件・全K・全学習反復へ渡してから解放する。
  recordは生成幅・試料・行順・設定・15seed・device/library versionを持つ。
  同じ入力・生成幅・device・softwareでの再生を対象とし、CPUとCUDAのビット一致は保証しない。
  RNG/FP32 scopeを抜けてからyieldし、呼出側の処理へ乱数状態や演算設定を持ち越さない。
- `src/wood_degradation_map/experiments/lfr.py`:
  `LFRAccumulator` は1試料・条件・K・学習反復に対応する。全15drawのラベルとcleanを
  試料ID・HDF5行・元座標で照合し、画素別の反転件数を整数で合計してからFP32で割る。
  chunkごとの率を平均しない。種類ごとに各5回の値とFP32平均、clean/各drawのoccupancyを返す。
  未完了試料、draw欠落/重複、座標不一致・重複、背景0の混入を成功結果として返さない。
  単一画素・単一クラスタのLFRは定義されるため、占有率を併記してそのまま計算する。
  番号matchingをせず、固定中心からのラベル変化を測る。
- `accumulate_lfr_block()` はcleanと15摂動をそれぞれ一度だけtransformし、全Kで共有する。
  `NeuralRepresentation` のeval・全可視・FP32抽出を再利用し、PCA/encoder/中心をfitし直さない。
  各consumerにはスペクトルのprivate copyを渡し、他条件と共有する入力の上書きを防ぐ。
- `tests/experiments/test_lfr.py`:
  小規模CPU人工データで参照augmenterとの一致、読み取りchunkを変えた再生、RNG復元、
  元入力保持、全15drawの対応、手計算のLFRと5回平均、欠落/重複の拒否、K間の表現再利用、
  NNの全可視・FP32・decoder非使用を検証する。
  複数生成batchのfixtureだけ生成幅を7へ縮小する。本番の1024という定数も別テストで照合する。
  本番規模の性能・GPUでの生成・評価CLI・結果ファイル保存はまだ未確認/未接続。

ユーザー実行済みの最小テスト（Codexでは再実行していない）:

```powershell
uv run pytest tests/experiments/test_lfr.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

ユーザー共有ログで `27 passed, 1 warning in 2.93s` を確認済み。
警告は既知の `norm_first=True` によるNested Tensor最適化の無効化通知。
同じテストの再実行は不要。GPUでの摂動生成と本番評価への接続は未確認。

### 第4段階のcosine-silhouette・3反復間ARIの実装・確認記録

- `src/wood_degradation_map/experiments/diagnostic_metrics.py`:
  `fold_silhouette()` は、検証済みfold/inventoryから得るtest試料ID・画素数を別途受け取り、
  全試料・全保存行が揃っていることを確認する。expectedの件数を入力表現から推測しない。
  表現と予測ラベルは同じSpectrumBatchの行・元座標に基づいて呼出側で対応付ける。
  ソートした試料順で全test表現をpoolし、ChemoMAE 0.2.1の参照関数を1回だけ呼ぶ。
  FP32・autocast/TF32無効・eps=1e-12を明示し、別の正規化や距離定義へ置き換えない。
- 戻り値はpoolした画素scoreと各試料のoffset・画素数・平均・占有数・singleton画素数、
  試料macro平均と利用可能試料数。singletonは試料内ではなくfold全体のクラスタ占有数で判断する。
  fold全体の使用クラスタ数1または全画素singletonは全試料のscoreを理由付きNoneとする。
  不正な入力表現は、指標の定義域外とは分けて元HDF5行付きの入力失敗にする。
  参照関数の出力が非有限または[-1,1]外の場合は元画素を示して停止し、clipや無言の除外をしない。
- 参照関数はN×N距離行列を作らないが、全N×D表現と複数のN×D作業配列をdeviceへ置く。
  `chunk_pixels`（既定1,000,000）は主にN×Kの比較tileを制限し、全体のGPUメモリを上限化しない。
  CPU fixture以外のGPU負荷は未計測で、暗黙のdevice fallbackはしない。
- `repeat_ari()` は1試料・条件・fold・Kについて、反復1/2/3の全画素ラベルを受け取る。
  呼出側のrun読込で条件・fold・Kを検証し、ここでは試料ID・全HDF5行・元座標の一致を照合する。
  3対のcontingencyと組合せ件数を整数で保持し、積のint64 overflowを避けて最後の除算をFP32で行う。
  ラベル番号の整列やHungarian matchingは行わない。
  N<2は未定義、両分割が単一クラスタまたは全画素singletonの完全一致は1とし、
  各反復の単一クラスタ/all-singleton flagと使用クラスタ数を併記する。
  3対の値と試料内平均を返し、反復対のSDを独立反復の不確実性として追加しない。
- `tests/experiments/test_diagnostic_metrics.py`:
  pooled計算1回と試料macro、sklearnの小規模cosine-silhouette/ARIとの一致、
  chunk変更・singleton・厳密なゼロ距離・未定義・非有限/範囲外値・画素欠落/不一致、
  ARIの手計算の負値・番号置換・退化・整数積を検証する。
  整数積のテストだけ各100,000要素の整数ラベルを使い、スペクトル大配列やGPUは使わない。
  本番データの抽出・評価CLI・結果ファイル保存・OOF集計はこの段階では未接続。

ユーザー実行済みの最小テスト（Codexでは再実行していない）:

```powershell
uv run pytest tests/experiments/test_diagnostic_metrics.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

ユーザー共有ログで `33 passed in 2.64s` を確認済み。同じテストの再実行は不要。
実データ・GPU・学習は使っていない。本番CLIへの接続とGPU検証は残っている。

### 第4段階の試料macro・SD・paired差（CPU検証済み）

- `src/wood_degradation_map/experiments/aggregation.py`:
  `ScoreRecord` は試料・test fold・条件・K・metric・反復ID・状態・値・理由を持つ。
  通常の指標は `lla_3/5/9`、`adjusted_lla_3/5/9`、`lfr_noise/shift/both`、`silhouette` を識別する。
  LFRの入力値は各摂動種の5回平均であり、drawを学習反復として渡さない。
  expectedの試料→test fold対応は保存済みmanifestから呼出側が渡し、入力レコードから推測しない。
  全試料×3反復のレコードを要求する。欠落・重複・fold/条件/K/metric不一致はエラー。
  `defined` は有限値と理由なし、`undefined/failed/interrupted` はNoneと明示的理由を必須にする。
  NaNを未定義値として取り込まず、実行失敗を指標の定義域外と同一視しない。
- `aggregate_scores()` は3反復すべてで値が定義された共通試料を使う。
  試料内の反復平均と、その試料間SD（ddof=1）、各反復の試料macroとその反復間SD（ddof=1）、
  3つのmacroの平均を返す。計算はFP32。fold平均の平均やpixel-weighted平均にはしない。
  反復別の元の利用可能数・除外理由、共通対象数・ID・除外ID、全試料/反復の入力値を保持する。
  共通0試料は集約値を理由付きNone、共通1試料は試料間SDだけ未定義（反復間SDは計算可能）。
  失敗/中断の出典が含まれる場合はflagを残し、残りの値で全run完了を装わない。
- `paired_difference()` は条件−参照条件を同じ試料・fold・K・反復で引いてから集計する。
  両条件・3反復で共通の試料に限定し、条件ごとに対象が違うmacro平均の差を代用しない。
  共通対象から外れた試料でも、定義できる反復別の差と、計算できない差の出典理由は保持する。
  LFRの改善方向が負であることを理由に符号を反転しない。
- `aggregate_ari()` は各試料の3対平均をmacro平均し、試料間SDを返す。
  反復対の値・未定義理由・退化flagを保持し、3対のSDや通常の反復別paired contrastは作らない。
  試料レコードや反復対が欠けた状態で平均しない。
- `tests/experiments/test_aggregation.py`:
  手計算の異なる2種類のSD、foldサイズの違い、共通0/1試料、条件ごとの利用可能集合の違い、
  paired差の符号、負の補正LLA、失敗/中断、レコード欠落/重複/不整合、ARIの退化・未定義を検証する。
  本番のファイル読込・数値結果保存・全条件/全Kの集計CLIはまだ接続していない。
  有意差検定・信頼区間・指標をまとめた順位scoreは追加していない。

ユーザーが実行する最小テスト（Codexでは未実行）:

```powershell
uv run pytest tests/experiments/test_aggregation.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

2026-09-06のユーザー共有ログで `34 passed in 2.47s` を確認した。
小規模な数値/ラベルfixtureのみで、実データ・GPU・学習は使っていない。同じ確認の再実行は不要。

### 第4段階の評価CLI・保存結果への接続（CPU検証済み）

- `evaluation_pipeline.py` と `scripts/experiments/evaluate_representations.py` を追加。
  1つのfoldで指定した条件×学習反復を対象に、本番の保存済み表現・全7Kの中心・clean testマップを読む。
  全入力runの完了・manifest・出典・hashを検証してから出力を作る。PCA/NN/KMeansの再fitはしない。
  PCAの出典反復はcleanマップのrun記録から引き継ぐ。smoke重みやtrain probeは本番評価に入れない。
- 全test画素を1回読み、試料ごとに15摂動を生成。同じブロックを指定条件・反復へ渡す。
  cleanと15摂動の各表現を全Kで再利用し、元のclean予測が保存済みマップと完全一致することを確認する。
  `lfr.py` の `accumulate_lfr_block()` は既存の加算に加えclean表現と予測を返し、
  silhouette用のclean再抽出を避ける。既存の呼出側は戻り値を使わなくてもよい。
- 試料ごとのLLA/補正LLA、近傍coverage、clean/各摂動のoccupancy、15回のLFRと3種類の平均を保存する。
  silhouetteは各run・Kで全test試料をまとめて計算し、試料ごとの平均とoffset、pixel-wise FP32値を保存する。
  未定義は理由付きJSON null。画素欠落・数値異常・clean予測の不一致は失敗とし、条件別に画素を落とさない。
- 保存先は `results/evaluation/<condition>/fold_<f>/repeat_<r>/`。
  `run.json` に出典、固定config、code hash、runtime、対象条件/反復、chunk幅、CPU表現容量を記録。
  `samples/<KYOw>.json` にLLA/LFR詳細、`silhouette/k<K>.json` と `pixels_k<K>.npy` にsilhouetteを保存。
  `scores.json` は集計部の `ScoreRecord` と同じ形式で、全test試料×全K×10指標を保持する。
  `shared_inputs.json` は生成条件・seed・clean/座標/15入力のストリームhashを保持し、摂動スペクトルは保存しない。
  `completion.json` を最後に作る。途中失敗・中断は `failure.json` に残し、既存出力の上書き・自動再開はしない。
- `check` は学習・表現抽出・摂動生成・指標再計算をせず、出典・保存hash・試料/K/指標coverageを確認する。
  複数の条件/反復を指定したcheckでは、別々の呼出で評価した場合も入力hashと生成条件の一致を要求する。
  後続の条件比較・OOF集計でも `check_evaluations()` によるこの照合を通す。seedだけで一致を仮定しない。
- メモリは、指定runごとに全test clean表現をCPUに保持する（画素数×次元×4 bytes）。
  モデル・中心はGPUに保持し、silhouetteはrun/Kを順に処理する。追加の全fold配列も必要で、
  silhouetteのchunk幅が制限するのはN×K作業領域のみ。CLIは容量の一部の見積りと進捗を表示する。
  自動的な画素間引き・CPU fallback・設定変更はしない。所要時間はjoint処理の経過時間で、run間で足し合わせない。
- 新規CPUテストは2つのtest試料を含む小規模fixtureを使用する。
  B0/B1の複数反復での実入力共有・全K再利用・fold pooled silhouette、合成NN重みからの全可視推論、
  保存/照合、未定義、改変、画素/試料欠落、中断、別呼出での入力一致を確認する。
  計算部の意味は変更しない。`lfr.py` の戻り値変更に対応して既存LFRテストも対象とする。

今回ユーザーが実行するコマンド（Codexでは未実行）:

```powershell
uv run pytest tests/experiments/test_evaluation_pipeline.py tests/experiments/test_lfr.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

新規25件と既存27件の計52件について、ユーザーから全件成功の報告を受けた。
実データ・GPU・本学習は使っていない。所要時間の共有はなく、時間値は記録しない。
CPUテスト成功後も、本番評価のGPU負荷・所要時間・保存容量は別途確認が必要。
本番3反復のARI保存、全foldのOOF集計CLI、計画比較・2×2交互作用は下記のCPU接続テストで確認済み。

以下は全指定runの本番cleanマップが揃った後の使用例であり、今回の即時実行対象ではない。
GPU実行前の品質表・空間図確認と負荷確認を先に行う。

```powershell
uv run python scripts/experiments/evaluate_representations.py run --conditions B0 B1 --fold 1 --repeats 1 2 3
uv run python scripts/experiments/evaluate_representations.py check --conditions B0 B1 --fold 1 --repeats 1 2 3
```

### 第4段階のOOF・ARI・計画比較の保存（CPU検証済み）

- `oof_pipeline.py` と `scripts/experiments/aggregate_oof.py` を追加。
  指定条件について全5fold・3反復の評価成果物を `check_evaluations()` で検証する。
  保存済みmanifestから試料のtest foldを取り出し、全試料が1回ずつOOF対象になることを確認する。
  同じfoldの全指定条件・反復で摂動の実現値が一致することを要求し、欠けたrunや失敗・中断を無言で除外しない。
- 保存済みcleanマップを元HDF5座標で読み、各試料・条件・全7Kについて3反復間のARIを計算する。
  3対の値、contingency、退化flag、試料内平均を残し、試料macroと試料間SDを保存する。
  ARIの反復対SDは作らない。spectra読込・学習・再fit・推論・Hungarian matchingは行わない。
- 通常の10指標は既存 `ScoreRecord` を読み、全OOF試料を対象に既存の集計部へ渡す。
  試料macro、試料間SD、3反復間SD、共通対象、反復別利用可能数と未定義理由を保存する。
  fold平均の平均にはしない。入力の欠落/重複/試料・fold・反復不一致はエラー。
- 実験プロトコル第4.3節の10組の計画比較は、必要条件が指定されている組について全10指標・全Kを保存する。
  2×2交互作用は主評価6指標（LLA-3/5/9とnoise/shift/bothのLFR）について保存する。
  同一試料・反復で `(M11-M10)-(M01-M00)` をFP32計算してから、4条件×3反復の共通対象で集約する。
  条件ごとに異なる対象のmacro差を代用せず、LFRの符号を反転しない。計画外の比較・検定・CIは追加しない。
- 保存先は `results/oof/<snapshot>/`。snapshotの既定名はUTC時刻で、既存名は上書きしない。
  `run.json` に固定config・manifest・code hash・全出典runの完了記録hash・比較計画を保持する。
  `scores.json`、`summaries/<condition>/k<K>.json`、`ari/<condition>/k<K>.json`、
  `comparisons/<condition>_minus_<reference>/k<K>.json`、`interaction/k<K>.json` を保存する。
  `completion.json` を最後に作り、中断・計算失敗は `failure.json` に残す。
  少数の条件のみを集計した場合は、その指定範囲と未作成比較を明示し、全条件完了とは扱わない。
- `check` は出典を再検証し、成果物hash・保存ファイル集合・対象条件/試料/run件数を照合する。
  ARIや指標・集約値の再計算はしない。重みは既存の検証経路でCPUへ読み、SNV・GPUは使わない。
  既存の評価計算部・保存形式・code hash対象ファイルは変更していない。
- 新規31テストを追加。交互作用の手計算・共通対象・元の符号、小規模保存fixtureでの全fold集計、
  欠落/重複/変更出典/無効画素/中断を検証する。
  接続テスト1件は小規模B0の全5fold・3反復でクラスタリング・評価成果物を実際に作り、OOF保存/checkへ接続する。
  その他の保存テストは合成scoreと出典検証adapterで集計部分を分離しており、本番計算の証拠とは扱わない。

今回ユーザーが実行するコマンド（Codexでは未実行）:

```powershell
uv run pytest tests/experiments/test_oof_pipeline.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true
```

ユーザーから31テスト全件成功の報告を受けた。所要時間の共有はなく、時間値は記録しない。
小規模CPUクラスタリングと評価も含む接続テストであり、実データ・GPU・本学習は使っていない。
本番GPU評価、代表試料の事前固定、本学習前の負荷確認は引き続き未完了。
品質表・空間図の後続確認は第1段階の記録を参照する。

以下は指定条件の本番評価が全5fold・3反復で揃った後の使用例であり、今回の即時実行対象ではない。

```powershell
uv run python scripts/experiments/aggregate_oof.py run --conditions B0 B1 A0 M00 M10 M01 M11 --snapshot main_oof_v1
uv run python scripts/experiments/aggregate_oof.py check --snapshot main_oof_v1
```

## 5. 本学習前の動作確認

- [ ] 対象実装の最小テストと関連チェックのコマンドを用意し、結果を確認する。
- [ ] 少数batchの学習から全可視抽出・クラスタリング・評価・保存まで通す。
  動作確認runを本実験の結果と分け、短縮学習の値で条件選択をしない。
- [x] batch size=1024で、全可視A0を含めGPUメモリ・入出力負荷を確認する。
  収まらない場合は状況を報告し、batch size・accumulationを暗黙に変更しない。
  A0/M11の各6batchとtrain行列読込で確認済み。長時間の本学習は未実行。
- [x] lr列、weight decay対象、FP16学習とFP32評価の切替、更新回数、再開動作を確認する。
  CPUテストとA0/M11のGPU再開確認で検証済み。
- [ ] GPU・library version・演算設定、実行時間・保存容量の見積りを記録する。
  同一モデルのclean/摂動後表現をK間で再利用できることも確認する。
- [ ] 本文用の代表試料の選択基準・IDを、結果を見る前にユーザーと固定する。
  任意の形状診断を採用する場合は、この時点までに定義を決める。未定義のまま実装・評価しない。

完了条件: 固定configとmanifestで本実験を開始でき、実行コマンド・出力先・所要時間の見通しがある。
本実験開始時には実行対象と負荷をユーザーへ提示する。

### 合成入力によるGPU評価smoke（今回の実装・実行手順）

`evaluation_smoke.py`と専用CLIを追加した。保存済みクラスタリングsmokeのrun・manifest・
config・code hash・fit記録・中心hashを照合し、そのrunが使ったPCA/ニューラル重みを読む。
学習・PCA/KMeansの再fitはしない。既存の学習・クラスタリング・本番評価モジュールは変更していない。

固定seedの合成SNV 1,025画素（41×25）で、LFRの15摂動・全7個のK、LLA、silhouetteを計算する。
1,024画素batchと末尾1画素を通し、clean/摂動表現をK間で共有する。
保存した配列・指標の再読込一致、固定中心の不変性、保存clean表現からのラベル再現を確認する。
単一クラスタのsilhouetteは未定義のまま保存し、別の既知4点でsilhouette kernelの正解値を確認する。
合成入力の座標・行番号は実HDF5を指さず、各値はCV指標や条件比較に使用しない。
CLI開始時の通常manifest検証は実HDF5の座標・maskを読むが、評価smokeは実試料のSNVを読まない。

出力先は `outputs/experiments/preflight_v1/results/evaluation_smoke/<ID>/...`。
本番用score schema・completion statusと分離し、既存ディレクトリを上書きしない。
GPU・library・演算設定、時間、最大GPUメモリ、出力容量も保存するが、
この合成probeを本番全foldの負荷見積りや科学的な入力確認の代わりにしない。

新規CPUテストでB0/PCAと小型ニューラルfixtureの保存接続、出典違い、全K共有・末尾、
単一クラスタ、中断、保存破損、上書き拒否を検証する。Codexではテスト・GPUとも未実行。
まず次を実行し、成功後にGPUの4コマンドを順番に実行する。

```powershell
uv run pytest tests/experiments/test_evaluation_smoke.py -vv -s --durations=10 -o faulthandler_timeout=30 -o faulthandler_exit_on_timeout=true

uv run python scripts/experiments/smoke_evaluation.py --condition B0 --fold 1 --clustering-smoke-id 20260905T194937_420573Z
uv run python scripts/experiments/smoke_evaluation.py --condition B1 --fold 1 --clustering-smoke-id 20260905T194946_230032Z
uv run python scripts/experiments/smoke_evaluation.py --condition A0 --fold 1 --clustering-smoke-id 20260905T195218_899327Z
uv run python scripts/experiments/smoke_evaluation.py --condition M11 --fold 1 --clustering-smoke-id 20260905T195232_653899Z
```

確認点は全件passと、各GPU出力の `status=evaluation_smoke_completed`、`checks_passed=true`、
`generation_block_sizes=[1024, 1]`。時間・最大メモリを実行後に確認する。全test評価とOOF集約は開始しない。

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

次回の開始位置: **上記「合成入力によるGPU評価smoke」のテスト・GPU出力を確認する**。
`data/processed/production_v1/`への保存先簡略化後の再生成・preflight再構築はユーザーから全件成功の報告あり。
続いて本文代表例の事前指定と本番全量の負荷確認へ進む。代表例の基準・IDは未決定のまま保持する。
OOF・ARI・計画比較の接続31テストはユーザーから全件成功の報告あり。
評価パイプライン接続と既存LFRの計52テストはユーザーから全件成功の報告あり。
試料macro・SD・paired差のCPUテスト34件は成功済み。
loader・B0/PCA（本番fitはfold 1）と、ChemoMAE共通部品のCPUテストは確認済み。
Trainer継承部分とA0/M11のGPU動作確認、KMeans共通部品のCPUテスト15件は成功。
接続テスト20件とB0/B1/A0/M11のGPU train 64画素probeも成功。本番全test推論・評価は未確認。
第1段階の品質レポートは10テスト成功・生成済み。保存済み品質表と全49試料の空間図を確認済み。
49試料の採用とKYOw単位の分割は確認済み。原材関係の再確認は新情報がある場合に限る。
