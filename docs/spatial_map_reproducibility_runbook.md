# 空間マップ再現性補助実験の実行手順

2026-10-15時点で実装を追加した。**学習・評価・描画・テスト・静的検査は未実行**。
2026-10-16追記：前回案内した4テストファイルはユーザー環境で全通過したとの報告を受けた。
その後、初回マップ基準の表示用色合わせと対応テストを追加した。**今回の変更後のテスト・描画は未実行**。
研究条件と解釈は[実験計画](design/spatial_map_reproducibility_plan.md)を正とする。
以下のコマンドは実行手順であり、実行済み記録ではない。

## 対象と保存先

- M11・K=8、全49試料。既存の401,408 fit画素を共有する。
- encoder 1は既存学習を参照する。encoder 2・3を各800 epoch学習し、それぞれ3 seedのKMeansをfitする。
- encoder内9対とencoder間27対を分ける。初回マップからの8対はこの36対の内数。
- 3×3・5×5・9×9近傍はChebyshev半径1・2・4画素。内部境界の中点を使い、方向別の被覆率を保存する。
- 再現性の合否判定、FT-IRの採取範囲・位置の決定は行わない。

| 用途 | 既定path |
| --- | --- |
| 変更しない元実験 | `outputs/experiments/global_v1/` |
| 変更しない入力 | `data/processed/production_v1/`・`data/metadata/古材メタデータ.csv` |
| 本補助実験の出力 | `outputs/experiments/global_spatial_reproducibility_v1/` |
| 専用CLI | `scripts/experiments/global_spatial_reproducibility.py` |

元実験と補助実験の出力pathは、同一・親子関係とも拒否する。
既存の`global_fit.py`や`global_k8_3seed.py`は変更していない。
既存成果物を検証する際のcode hashも維持する。

## 実行前の小規模確認

既存のPython 3.11.15環境とlockに従う。新しい依存関係は追加していない。
リポジトリrootから実行する。`--no-sync`により、この操作で依存関係を更新しない。

```powershell
uv run --no-sync pytest tests/experiments/test_spatial_reproducibility_metrics.py tests/experiments/test_spatial_reproducibility_training.py tests/experiments/test_spatial_reproducibility_reporting.py tests/experiments/test_spatial_reproducibility_pipeline.py -q
```

小規模配列・一時ディレクトリ・mockを使い、境界中点、距離換算、mask不一致、空境界、分裂側／合流側、
ARI、seed、出典、集計の欠測規則、保存・表示処理を確認する。実データの学習や全件評価は行わない。
通常のpytest cacheと一時ファイルは作成される。テストが通ったことだけでGPU学習・全件描画の成功とは扱わない。

## 準備と元成果物の監査

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py prepare
```

CPUで元manifest・encoder重み・KMeans中心・全試料mapの由来、hash、shape、maskを検証する。
入力検証に伴うファイル読み取りとhash照合がある。新しい抽出座標やmapは生成しない。
全seed一覧と衝突検査、9マップの反復ID、36対、境界定義version、元成果物の参照を`protocol/`に固定する。

初回$(e,k)=(1,1)$には元の`results/clustering/M11/repeat_1/maps/`を使う。
`results/global_k8_3seed_v1/clustering/M11/repeat_1/`が存在する場合、元の初回mapとの
ラベル配列一致・変更画素数・ARI・境界比較を`protocol/initial_map_audit.json`に記録する。
この再生成版は9マップに数えず、差があっても初回mapを差し替えない。
再生成版が存在しない場合は監査表にその旨を残す。正本とKMeans repeat 2・3は必須。

再実行時は次で保存済みprotocolと現在の出典を照合する。

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py prepare --resume
```

## encoderの追加学習

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py train --encoders 2 3 --device 0
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py training-check --encoders 2 3 --device 0
```

GPU 0で2回を逐次学習する。各回800 epoch、FP16 AMP・FP32パラメータ・最終raw重みを使用する。
全fitスペクトルのCPU配列、モデル・optimizer・batchのGPUメモリ、各epochのcheckpoint I/Oが必要。
実測所要時間・ピークメモリは未確認。原データ・既存全体fitの出力は上書きしない。

中断時は同じ引数に`--resume`を追加する。最新の完了epochの`checkpoints/last.pt`から再開し、
中断したepochはやり直す。完了済みencoderはcheckpointと最終重みを照合してskipする。
`--encoders 2`のように一方だけの実行も可能。seedの選び直しやepoch数の変更は受け付けない。

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py train --encoders 2 3 --device 0 --resume
```

`training-check`は同じCUDA環境でcheckpoint・optimizer・RNG状態・履歴と最終raw重みの対応を確認する。
学習・ファイル書き込みは行わない。通常の`check`は後述の全成果物のCPU監査である。

## 各encoderのKMeansと全画素map

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py cluster --encoders 2 3 --device 0 --chunk-pixels 1024
```

各encoderのfit画素表現を一度抽出してCPU上に保持し、その3 seedで再利用する。
全画素の表現はchunk単位で処理し、大量の中間表現をディスク保存しない。
各KMeans fitの中心を保存・再ロードし、中心とprobe予測の一致を確認して全49試料のmapを作る。
追加計算はKMeans fit 6回・追加map 6枚／試料。

中断後は同じコマンドに`--resume`を付ける。完了したencoder×KMeans単位を検証してskipし、
完了していないfitは同じseedで最初から処理する。部分map集合を評価へ流さない。
stageは一時ディレクトリで作り、完成後に公開する。公開済み出力の改変や部分破損を検出したら停止する。

## 評価と集計

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py evaluate
```

CPUで試料単位に9マップを読み、同じ元mask上の36対を評価する。
欠けたmapやmask変更は入力エラーにする。境界が空の正しいmapは保持し、
「境界あり→空」は0、「空→境界あり」は未定義として理由・分母を残す。
境界が空の両者を一致率1へ置き換えない。

主集計は比較群×指標ごとのcomplete-caseとする。境界一致率の小さい側・大きい側は、
同一encoder内9対またはencoder間27対の全方向が定義されることを要求し、近傍幅間の対象試料を揃える。
部分欠測の値は診断表に残すが、残存対だけの平均で主集計を補わない。
試料等重みのmacro平均・試料間分布を報告し、36対を独立標本として検定・CIを計算しない。

分裂・合流の重なり表とVI成分も保存する場合は、最初の評価実行時に次を使う。
計算対象は全36対で、結果を見て都合のよい対だけを選ぶ機能は設けない。

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py evaluate --include-overlap
```

評価済み出力の確認・skipは`--resume`で行う。`--include-overlap`は元の実行と同じ指定が必要。
既存出力を上書きしてオプションを変更しない。

## 図表と完了確認

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py report --dpi 160
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py check
```

CPUで全49試料の3×3一覧、境界重ね描き、3尺度の境界頻度、初回からの方向別比較、
全体のARIと境界一致率の対応・分布を保存する。全試料分のPNGを生成するため、図の数と出力容量が増える。
`--dpi`は描画解像度のみを変え、計算上の画素座標・許容距離は変えない。
3×3一覧は初回`(1,1)`を基準に色を揃える。全49試料の有効画素を合算し、各反復との
重なり画素数が最大になる一対一の対応を求め、同じ対応を全試料に使用する。
背景0・初回の表示番号を維持し、元ラベル配列や指標値は変更しない。境界図・頻度図も元ラベルから作る。
全体の対応表は`results/report/color_matching.json`へ保存する。重なり0の割当も記録し、
分裂したクラスタを同じ色へ統合することはしない。色の一致は化学的意味の一致を保証しない。
頻度は共有encoderを含む9マップの記述量であり、独立試行の確率や化学的確信度ではない。

描画の再実行も`--resume`で検証・skipする。DPIは保存済み設定と一致する必要がある。
`check`は由来・hash・対象数・集計整合性を確認する技術的な完了確認であり、再現性の合否判定ではない。

## 成果物

補助実験rootからの相対pathを示す。

| path | 内容 |
| --- | --- |
| `protocol/run.json` | seed・境界定義・組合せ・元manifest／モデル／mapの出典 |
| `protocol/initial_map_audit.json` | 初回mapと同seed再生成版の監査 |
| `results/neural/M11/encoder_{2,3}/` | 学習run・履歴・checkpoint記録・完了記録 |
| `checkpoints/neural/M11/encoder_{2,3}/` | 最終raw重みと再開用checkpoint |
| `results/clustering/M11/encoder_{e}/kmeans_{k}/` | 追加map・run・完了hash |
| `checkpoints/clustering/M11/encoder_{e}/kmeans_{k}/centers_k8.npz` | KMeans中心。encoderの由来は対応するrunに結び付く |
| `results/evaluation/map_sources.json` | 49×9 mapの元path・hash・shapeと二つの反復ID |
| `results/evaluation/map_diagnostics.csv` | 画素数・有効隣接辺数・境界数／密度・占有率 |
| `results/evaluation/pair_metrics.csv` | 49×36対×3尺度×2方向、分子／分母・未定義理由・ARI |
| `results/evaluation/initial_reference_metrics.csv` | 初回からの8対の抜粋 |
| `results/evaluation/samples/{sample_id}.json` | 対別の構造化値・退化flag・任意の重なり表／VI |
| `results/evaluation/summary.json` | 試料別要約、完全対数、除外理由、macro分布、共通試料による群差 |
| `results/report/` | 全試料の図、境界頻度配列、全体図、描画条件・hash |
| `results/report/color_matching.json` | 初回基準の表示用対応。全49試料を合算した重なり表、元番号→表示番号、重なり0の割当 |
| `attempts/` | 実行段階・完了／失敗／中断・例外型・所要時間・再開設定 |

JSONの未定義値は`null`、CSVでは空欄＋理由。頻度配列の無効画素はNaNとし、有効画素の0とは区別する。
重みは既存の`checkpoints/`のignore規則に従う。新しい`.pt`等をソース管理へ加えない。

## 失敗時の扱いと確認点

- 再実行時にrun・config・code・runtime・出典が違えば停止する。完了記録や元成果物を手編集して通さない。
- 学習はepoch境界から、clusteringは未完了fit単位から、評価・描画は未完了stageから再試行する。
- 初回checkpoint生成前の失敗、部分破損、OS停止時の残存stageは自動削除・上書きしない。状況を確認して復旧する。
- `attempts/`は通常の例外・KeyboardInterruptを記録する。プロセス強制終了・OS停止時の記録を保証しない。
- 境界密度が高いと近接一致も高くなり得る。指標だけで順位付けせず、密度・占有率・元map・方向別図を併読する。
- 初回mapへの位置対応を保ち、境界の安定とクラスタの化学的意味の安定を区別する。

本手順書の作成時点では、実データでの完了・所要時間・図の見た目は未確認。

表示用色合わせの変更後は、次の2ファイルで割当・入力不変性・描画・対応表保存を確認する。
小規模の一時ファイルと図を生成するテストであり、実データの学習は行わない。

```powershell
uv run --no-sync pytest tests/experiments/test_spatial_reproducibility_reporting.py tests/experiments/test_spatial_reproducibility_pipeline.py -q
```
