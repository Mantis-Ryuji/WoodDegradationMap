# 検証履歴

## 1. 位置づけ

この文書は、本番開始前に実施した入力確認、テスト、GPU preflightと、完了した本番runの
工学的な実行記録の要約である。
研究条件や評価方法の定義は[design/README.md](design/README.md)以下、現在の実行状態は
[../ToDo.md](../ToDo.md)を参照する。ここに記す成功は実装と保存契約の工学的確認であり、
モデル性能や劣化との対応を示す結果ではない。

特記がない限り、コマンドはユーザーが `uv` で実行し、その出力を共有した。Codexは同じテストを
再実行していない。個々のhash、runtime、GPU memory、completionは、preflightでは
`outputs/experiments/preflight_v1/`、本番runでは `outputs/experiments/production_v1/` の保存記録を
正とする。

## 2. 本番入力

2026-09-06、補間後・SNV前の反射率が1帯域でも負の画素を背景とする前処理を再生成した。

| 項目 | 確認値 |
| --- | ---: |
| 採用試料 | 49 |
| 形態処理後mask画素 | 3,902,746 |
| 保存有効画素 | 3,902,250 |
| 負の補間反射率による除外 | 496 |
| 保持元帯域 | 222 |
| 除外元帯域 | 34 |
| HDF5合計 | 5,574,878,801 bytes（約5.19 GiB） |

除外496画素は全件 `excluded_reason_code=3` として記録され、train・testで同じ背景規則を使う。
反射率のclipやSNV後の負値による除外は行わない。49試料すべてで$q=8192$の非復元抽出が可能で、
KYOw単位の5-fold splitにtrain/testの試料重複はない。異なるKYOw間の同一原材関係は不明であり、
ユーザー決定によりKYOwだけをsplit単位とした。

## 3. 自動テスト

| 対象 | 結果 | 主な確認 |
| --- | ---: | --- |
| manifest | 20 passed | 49試料、5-fold、共通train座標、保存・再読込 |
| neural共通部品 | 29 passed | 初期化、optimizer、augmentation、全可視FP32抽出 |
| 学習・再開 | 16 passed | epoch/update、checkpoint、同一run再開、失敗記録 |
| clustering | 15 passed | train限定fit、固定center、全test行と座標復元 |
| clustering pipeline | 20 passed | 全K map、重み由来、破損・中断・上書き拒否 |
| spatial metrics | 33 passed | LLA、補正、近傍・境界、未定義値、入力契約 |
| LFR | 27 passed | 共通摂動、連続乱数stream、flip集計、全可視推論 |
| diagnostic metrics | 33 passed | cosine-silhouette、ARI、degenerate case、完全性 |
| aggregation | 34 passed | 試料macro、2種類のSD、paired差、欠損の可視化 |
| input review | 10 passed | 表・リンク・出典保持、欠損画像、上書き拒否 |
| OOF pipeline | 31 passed | 5-fold・3反復の完全性、ARI、計画比較、snapshot検証 |
| evaluation pipeline + LFR回帰 | 52 passed | clean mapから評価・保存・再読込までの接続 |
| 前処理 + 入力検証 | 全件passed | 負値背景化、HDF5契約、`production_v1` 入力との照合 |
| evaluation smoke | 全件passed | B0、B1、A0、M11の合成入力GPU経路 |

`chemomae.models.chemo_mae` からTransformerのnested tensorに関するwarningが出るテストがあるが、
共有された実行では失敗や数値契約違反はなかった。

## 4. GPU preflight

### 4.1 学習と再開

| 条件 | smoke ID | epoch 2の16 batch | 単純外挿したfold 1全学習 | peak allocated | 再開誤差 |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | `20260905T201934_444403Z` | 0.6261 s | 約2.71 h | 1,524.36 MiB | weights 0、latent 0 |
| M11 | `20260905T202020_824505Z` | 0.5613 s | 約2.43 h | 919.33 MiB | weights 0、latent 0 |

各smokeは16 batch × 2 epochとepoch 2の再開replay、合計48 batchを実行した。入力順、augmentation、
mask、学習率、GradScaler、optimizer stepが再開前後で一致し、AMP skipは0だった。上表の時間は短い
区間からの単純外挿であり、本番800 epochの実時間ではない。

### 4.2 評価経路

合成入力によるGPU評価smokeは4条件すべて `checks_passed=true` だった。

| 条件 | wall time | peak allocated |
| --- | ---: | ---: |
| B0 | 0.812 s | 20.07 MiB |
| B1 | 0.986 s | 20.02 MiB |
| A0 | 1.250 s | 256.81 MiB |
| M11 | 1.150 s | 256.81 MiB |

B0・fold 1では実データの全量preflightも実施した。

| 工程 | wall time | peak allocated | 保存量 |
| --- | ---: | ---: | ---: |
| 全K clustering | 39.24 s | 633.41 MiB | 約0.94 MiB |
| 全test評価 | 203.12 s | 4,476.24 MiB | 約25.45 MiB |

test 10試料・906,428画素・7種類のKを処理し、700 score rowsは全件definedだった。train/testの
試料重複、保存試料の欠落、確認対象のmanifest・run・fit・score・shared input・code hashの不一致は
なかった。この結果はB0・fold 1のpreflightであり、本番CV結果には含めない。

## 5. 本番CV実行記録

### 5.1 A0・fold 1

2026-09-06、A0・fold 1の3反復について、学習、全7Kのclustering、全test評価と各checkを完了した。
学習completion、training history、重みとcheckpointのhashを照合し、各反復で800 epoch・249,600
attempted updatesが揃っていることを確認した。

| repeat | optimizer updates | AMP skips | epoch時間合計 | 最終train loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 249,505 | 95 | 2:47:33.955 | 0.0002234270 |
| 2 | 249,503 | 97 | 2:41:39.832 | 0.0002196052 |
| 3 | 249,507 | 93 | 2:47:37.736 | 0.0002236418 |

`optimizer_updates + amp_skips == attempted_updates` は3反復すべてで成立した。 `training_seconds` は
各epochの処理時間の合計であり、入力読込、source検証、保存処理などを含むCLI全体のwall timeではない。
本番学習のcompletion schemaはGPU peakを保存しないため、その値は未記録である。第4.1節のpreflight値を
本番runの実測値として代用しない。

repeat 2はepoch 26のcheckpoint保存後、 `checkpoint.json.tmp` から `checkpoint.json` への置換で
Windowsの `PermissionError` が発生した。保存済み `last.pt`、一時記録、training historyのepoch・更新数・
SHA-256が一致することを確認し、同じcheckpointを明示してepoch 27から再開した。失敗attemptと完了attemptは
両方保存され、最終completionと重みのhashも一致した。

### 5.2 Clustering・評価

各反復でfold 1のtest 10試料・906,428画素を処理し、事前固定した
$K\in\{2,4,6,8,10,12,14\}$ が揃った。runは `checks_passed=true`、既存成果物のcheckはclustering・
評価とも成功した。

| repeat | clustering wall | clustering peak allocated / reserved | 評価wall | 評価peak allocated / reserved | run別保存量 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 109.94 s | 255.81 / 388.00 MiB | 540.91 s | 374.90 / 564.00 MiB | 123.91 MiB |
| 2 | 81.81 s | 255.81 / 388.00 MiB | 474.94 s | 374.90 / 564.00 MiB | 124.25 MiB |
| 3 | 71.12 s | 255.81 / 388.00 MiB | 467.34 s | 374.90 / 564.00 MiB | 123.76 MiB |

保存量は各反復のneural・clustering・evaluationのresultsとcheckpointディレクトリを合計した値で、
3反復合計は371.92 MiBである。共有入力hashは3反復で
`d3004aefb441eaa86c70963d9e96537a3f538371cbc279eb85e242b696ac1bcd` と一致した。これは実行・保存契約の
確認であり、A0の性能評価や条件間比較ではない。性能は必要な全fold・反復が揃ったOOF snapshotで評価する。

## 6. 本文代表試料

結果を見る前に、各樹種で保存有効画素数が最大の試料を本文表示例として固定した。

| 樹種 | 試料 | 保存有効画素数 |
| --- | --- | ---: |
| クリ | KYOw02789 | 125,946 |
| ケヤキ | KYOw02777 | 121,687 |
| スギ | KYOw02784 | 117,549 |
| ツガ | KYOw02787 | 106,684 |
| ヒノキ | KYOw02720 | 131,174 |
| マツ | KYOw02769 | 161,734 |
| モミ | KYOw16750 | 58,739 |

選択規則、metadataとmanifestのSHA-256、解釈上の制約は
[design/visualization_and_interpretation.md](design/visualization_and_interpretation.md)第4.2節に記録した。
