# 空間マップ再現性補助実験の実行手順

M11・K=8の学習済みencoder 3組と、各encoderで共通のCosine-KMeans 3 seedを使い、
全49試料の9マップを比較する。研究条件・指標・解釈は[実験計画](design/spatial_map_reproducibility_plan.md)を参照する。

## 入力と出力

| 用途 | 既定path |
| --- | --- |
| 全体fit・encoder 1・初回マップ | `outputs/experiments/global_v1/` |
| 学習済みencoder 2・3と再利用マップ | `outputs/experiments/global_spatial_reproducibility_v1/` |
| 前処理済み入力 | `data/processed/production_v1/` |
| 試料メタデータ | `data/metadata/古材メタデータ.csv` |
| 出力 | `outputs/experiments/global_spatial_reproducibility_v2/` |
| CLI | `scripts/experiments/global_spatial_reproducibility.py` |

全体fitの入力先は`--source-experiment-dir`、学習済み成果物の入力先は`--artifacts-dir`、
出力先は`--experiment-dir`で指定できる。入力先の成果物は読み取り専用とし、出力先とは分離する。
学習済みencoderと再利用マップを参照するため、入力フォルダは残しておく。

| 反復 | 使用するマップ |
| --- | --- |
| `(1,1)` | 全体fitの`results/clustering/M11/repeat_1/maps/` |
| `(1,3)` | 全体fitの`results/global_k8_3seed_v1/clustering/M11/repeat_3/maps/` |
| `(2,1)`・`(2,3)`・`(3,1)`・`(3,3)` | 学習済み成果物の`results/clustering/M11/encoder_{e}/kmeans_{k}/maps/` |
| `(1,2)`・`(2,2)`・`(3,2)` | 出力先の`results/clustering/M11/encoder_{e}/kmeans_2/maps/`に生成 |

第1seedは1960870825、第3seedは2051198256。
第2seedは`_seed("kmeans", "global_spatial_reproducibility", "seed2_v2", 2, 8)`から導出し、
3encoderで共有する。整数値と衝突検査は`prepare`の出力と`protocol/run.json`で確認できる。
encoderの学習は不要。既存の全体fitと`global_k8_3seed_v1`の集計はそのまま使用できる。

## テスト

リポジトリrootから実行する。既存のPython環境とlockを使い、`--no-sync`により依存関係を更新しない。

```powershell
uv run --no-sync pytest tests/experiments/test_spatial_reproducibility_metrics.py tests/experiments/test_spatial_reproducibility_training.py tests/experiments/test_spatial_reproducibility_reporting.py tests/experiments/test_spatial_reproducibility_pipeline.py -q
```

小規模配列・合成マップ・一時ファイルを使い、境界中点、距離換算、空境界、ARI、seed、入力検証、
3組のfit、再開、集計、保存・描画を確認する。実データの学習・GPU処理・全件評価は行わない。
pytest cacheと一時ファイル・図が作成される。

## 準備から図表出力まで

既定pathを使用する場合は、次の順に実行する。

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py prepare
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py cluster --encoders 1 2 3 --device 0 --chunk-pixels 1024
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py evaluate --include-overlap
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py report --dpi 160
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py check
```

| 段階 | 処理と負荷 |
| --- | --- |
| `prepare` | CPUで入力の完了記録、重み・中心・mapの出典、hash、shape、maskを確認し、seed・反復対・評価条件を`protocol/`へ保存する。モデル・多数のmapの読み取りを伴う |
| `cluster` | GPU 0で各encoderのfit画素表現を抽出し、第2seedを計3回fitする。全49試料を推論し、計147マップと3組の中心を保存する |
| `evaluate` | CPUで各試料の9マップ・36対を評価し、49試料を等重みで集計する。`--include-overlap`は全36対の重なり表とVI成分も保存する |
| `report` | CPUで3種類・99枚のPNG、49試料の境界頻度配列、captionと配色対応表を保存する |
| `check` | 出典・hash・対象数・評価と図表の対応を検証する。再現性の合否判定ではない |

fit画素は既存の401,408座標を共有する。Cosine-KMeansはK=8、`max_iter=500`、`tol=1e-4`、FP32を使用する。
全画素の表現は`--chunk-pixels`単位で処理し、中間表現をディスクへ保存しない。
各fitの中心は保存・再ロードして中心とprobe予測の一致を確認する。

## 中断後の再開

生成段階は同じ引数に`--resume`を付ける。完了済みの段階・反復は検証してskipする。

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py prepare --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py cluster --encoders 1 2 3 --device 0 --chunk-pixels 1024 --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py evaluate --include-overlap --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py report --dpi 160 --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py check
```

clusteringは未完了fit単位、評価・描画は未完了stage単位で再試行する。
`--encoders 2`のように対象encoderを限定できる。`--include-overlap`・DPI・入力先などは保存済み設定と揃える。
`check`に`--resume`は付けない。

入力・設定・code・runtimeの不整合や、成果物の破損を検出した場合は停止する。
完了記録を手編集して通さず、エラーに示された入力や出力を確認する。
部分生成物やOS停止時の残存stageは自動削除・上書きしない。

## 評価と集計の確認点

- 同一encoder内9対とencoder間27対を分ける。初回`(1,1)`からの8対は36対の内数。
- ARIは元ラベルで計算する。境界一致率は内部境界の中点に対して双方向に計算し、
  3×3・5×5・9×9近傍をChebyshev半径1・2・4画素として扱う。
- 9マップは同じ有効画素集合を使う。欠けたmapやmask変更は入力エラーとする。
- 境界がある側から空の側への一致率は0、対応元が空の場合は未定義とする。
  両方が空でも一致率1にはしない。
- 主集計は比較群×指標ごとのcomplete-caseとする。境界一致率は予定した全対・全方向が定義された試料を使い、
  近傍幅間の対象試料を揃える。部分欠測は診断表に残し、残存対の平均で補わない。
- 試料等重みのmacro平均・試料間分布を報告する。36対を独立標本として検定・信頼区間を計算しない。

## 図の読み方と論文での使い方

| 図 | 読み取る内容 | 用途 |
| --- | --- | --- |
| `summary/01_ari_vs_boundary_coverage.png` | 横軸は試料ごとの対平均ARI、縦軸は対平均の境界一致率。上段はencoder内、下段はencoder間、左から3・5・9近傍 | 本文主図1枚 |
| `samples/{sample_id}/01_nine_maps.png` | 行がencoder、列がCosine-KMeans初期化。同じ行で初期化間、行をまたいで再学習を含む違いを見る | 全49試料。代表7試料を付録候補とする |
| `samples/{sample_id}/03_boundary_frequency.png` | 左から3×3・5×5・9×9近傍に内部境界が存在したマップの割合 | 全49試料。代表7試料を付録候補とする |

本文主図の青は対ごとの双方向の小さい側の平均、橙は大きい側の平均で、縦線は2値を結ぶ。
信頼区間ではない。ARIが低く境界一致率が高い点は、分割の変動に対して境界配置に共通性がある試料の候補となる。
境界密度が高いと近接対応も増えるため、密度・占有率と3×3マップを併読する。
対平均では特定の反復のずれが見えにくいため、個々の差は3×3マップと`pair_metrics.csv`で確認する。

境界頻度図の黄色（1）は9マップすべてで近傍内に境界があり、紫（0）はすべてで近傍内に境界がない場所を示す。
紫は不安定を意味しない。中間色は反復によって境界の有無が変わる場所である。
近傍幅を広げると頻度は下がらないため、9×9で黄色が増えることだけで安定とは判断しない。
頻度は9マップの記述量であり、独立試行の確率や化学的な意味の一致を示す値ではない。

3×3マップは全49試料の有効画素を合算した重なりから、初回`(1,1)`との一対一の対応を求める。
この対応に`global_v1/results/figures/global_k8_v1/M11/matching.csv`の変換を合成し、
同じ全体fit図の`report.json`のpaletteを使用する。全試料で同じ対応を使い、背景0を維持する。
色合わせは表示だけに適用し、元ラベル・指標・境界頻度は変更しない。

図・subplotにタイトルを付けず、マップ類のpixel軸・目盛・枠線は表示しない。
試料IDは画像群の下に配置し、表記はCosine-KMeansに統一する。説明は`captions.json`へ保存する。
`--dpi`は描画解像度だけを変え、画素座標・許容距離には影響しない。

## 成果物

出力先からの相対pathを示す。学習済み重みと再利用6マップは入力先を参照し、コピーしない。

| path | 内容 |
| --- | --- |
| `protocol/run.json` | seed、境界定義、反復対、入力manifest・モデル・mapの参照 |
| `protocol/initial_map_audit.json` | 初回mapと全体fit図が使用する同seed再生成版の照合 |
| `results/clustering/M11/encoder_{e}/kmeans_2/` | 生成したmap・run・完了hash |
| `checkpoints/clustering/M11/encoder_{e}/kmeans_2/centers_k8.npz` | Cosine-KMeans中心 |
| `results/evaluation/map_sources.json` | 49×9 mapのpath・hash・shape・反復ID |
| `results/evaluation/map_diagnostics.csv` | 有効画素数、隣接辺数、境界数・密度、占有率 |
| `results/evaluation/pair_metrics.csv` | 49×36対×3尺度×2方向の境界一致率、分子・分母、未定義理由、ARI |
| `results/evaluation/initial_reference_metrics.csv` | 初回からの8対 |
| `results/evaluation/samples/{sample_id}.json` | 対別の値、退化flag、重なり表・VI |
| `results/evaluation/summary.json` | 試料別要約、完全対数、除外理由、macro分布、共通試料による群差 |
| `results/report/` | 3種類・99枚のPNG、49試料の境界頻度配列、描画条件・hash |
| `results/report/color_matching.json` | 初回との対応・重なり表、全体fit表示IDへの変換、合成対応・palette |
| `results/report/captions.json` | 図の役割、panelの意味、試料数、表示上の説明 |
| `attempts/` | 実行段階、完了・失敗・中断、例外型、所要時間、再開設定 |

JSONの未定義値は`null`、CSVでは空欄と理由を使う。境界頻度配列の無効画素はNaNとし、有効画素の0と区別する。
中心等の学習済みpayloadは既存の`checkpoints/`のignore規則に従う。
