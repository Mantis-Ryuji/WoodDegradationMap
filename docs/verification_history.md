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

<a id="production-inputs"></a>

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

以下はそれぞれの実装時点で共有された検証結果であり、単一時点の全suiteの内訳ではない。
2026-09-07の一括検証は[第7.2節](#chemomae-022-validation)に示す。

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

以下の完了runでは、共通して次を確認した。個別の時間・更新数・中断経緯は各節に記す。

| 工程 | 確認済みの状態 |
| --- | --- |
| NN学習 | 800 epoch。attempted updatesはfold 1–4で249,600、fold 5で256,000。optimizer updatesとAMP skipsの和は試行数に一致し、nonzero LR updatesはoptimizer updatesより1少ない。重みの実在とcompletionのSHA-256を照合済み |
| clustering | `clean_test_maps_completed`・`checks_passed=true`。保存物checkは`validated_existing_clustering` |
| 評価 | `full_test_evaluation_completed`・`checks_passed=true`。保存物checkは`validated_existing_evaluation` |

`training_seconds`はepoch処理時間の合計で、CLI全体のwall timeではない。本番学習completionにGPU peakはなく、
preflight値で補わない。評価の`wall_seconds`はsource検証を除く評価開始からconsumer保存までの時間で、
共同評価では組合せ別の処理時間を表さない。性能比較は全fold・反復が揃ったOOF snapshotで行う。

### 5.1 主ニューラル5条件、fold 1

2026-09-06から2026-09-10にかけて、A0・M00・M10・M01・M11のfold 1・repeat 1–3、計15 runの学習と
全7Kのclusteringを完了した。各学習runは800 epoch・249,600 attempted updatesで、
completion、training history、重み・checkpointのSHA-256を照合した。

| condition | repeat | optimizer updates | AMP skips | training seconds | clustering wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 1 | 249,505 | 95 | 10,053.9554 | 109.94 s |
| A0 | 2 | 249,503 | 97 | 9,699.8320 | 81.81 s |
| A0 | 3 | 249,507 | 93 | 10,057.7363 | 71.12 s |
| M00 | 1 | 249,501 | 99 | 9,701.8195 | 83.97 s |
| M00 | 2 | 249,508 | 92 | 9,342.9576 | 71.21 s |
| M00 | 3 | 249,500 | 100 | 9,382.7762 | 77.03 s |
| M10 | 1 | 249,504 | 96 | 9,976.5919 | 112.44 s |
| M10 | 2 | 249,501 | 99 | 9,956.5483 | 117.43 s |
| M10 | 3 | 249,501 | 99 | 9,865.9994 | 116.56 s |
| M01 | 1 | 249,503 | 97 | 9,898.3589 | 109.03 s |
| M01 | 2 | 249,503 | 97 | 9,917.9678 | 114.41 s |
| M01 | 3 | 249,499 | 101 | 9,195.9055 | 99.60 s |
| M11 | 1 | 249,503 | 97 | 9,822.4305 | 102.69 s |
| M11 | 2 | 249,503 | 97 | 9,985.8211 | 87.41 s |
| M11 | 3 | 249,501 | 99 | 9,959.2961 | 100.91 s |

`AMP skip`は有限なlossに対するscaled backward後、動的GradScalerが非有限勾配を検出し、
そのbatchの`optimizer.step()`を呼ばずにloss scaleを下げた回数である。モデル初期値・画素順・mask・
augmentationはrepeatで変わり、mask・loss領域・augmentationは条件でも異なるため、勾配とloss scaleの
軌跡が変わり、run間でskip数が完全には一致しない。15 runのskipは92～101回、249,600 attempted updatesの
約0.037～0.040%であった。skipはepoch 47～59に始まりepoch 785～799まで分散し、該当epochの大半で1回、
最大3回だったため、学習開始時だけの異常ではなく、GradScalerが学習中にscaleを上げ、数値範囲の限界で
下げる動作として記録する。run間の実optimizer update数の最大差は9回（attempted updatesの約0.0036%）である。
固定予算は共通のepoch数とattempted updatesで定義し、skip分をrun別に補填して条件依存の延長は行わない。

A0・repeat 2はepoch 26のcheckpoint保存後、`checkpoint.json.tmp`から`checkpoint.json`への置換で
Windowsの`PermissionError`が発生した。`last.pt`、一時記録、training historyのepoch・更新数・
SHA-256を照合し、同じcheckpointを明示してepoch 27から再開した。失敗attemptと完了attemptは
両方保存され、最終completionと重みのhashも一致した。

M01・repeat 3はepoch 432のcheckpoint保存後、Windowsの予期しない再起動によって中断した。
`last.pt`、`checkpoint.json`、training historyのepoch・更新数・SHA-256が一致することを確認し、
同じcheckpointを明示してepoch 433から再開した。最終的に800 epochを完了し、completion、
training history、重み・checkpointのhashも一致した。

M11・repeat 2はepoch 487の途中で`KeyboardInterrupt`により中断し、その時点では486 epochが完了していた。
同じrunの`last.pt`を明示してepoch 487を先頭から再実行し、800 epochまで完了した。中断前後のattempt記録を
両方保持し、最終completion、training history、重み・checkpointの整合を確認した。

### 5.2 clean test map・全test評価

15組合せすべてでfold 1のtest 10試料・906,428画素と$K\in\{2,4,6,8,10,12,14\}$を処理した。
clusteringのGPU peak allocated / reservedは全runで255.81 / 388.00 MiBだった。

2026-09-08から2026-09-10にかけて、A0・M00の6組合せはまとめて、M10・M01・M11の各3組合せは
個別に、同じ固定configと共通摂動で評価した。

| 対象 | 評価単位 | `wall_seconds` | peak allocated | peak reserved |
| --- | --- | ---: | ---: | ---: |
| A0・M00、repeat 1–3 | 6組合せの共同評価・最後のconsumer | 2,672.56 s | 495.77 MiB | 684.00 MiB |
| M10、repeat 1 | 1組合せ | 575.89 s | 374.90 MiB | 564.00 MiB |
| M10、repeat 2 | 1組合せ | 489.78 s | 374.90 MiB | 564.00 MiB |
| M10、repeat 3 | 1組合せ | 559.64 s | 374.90 MiB | 564.00 MiB |
| M01、repeat 1 | 1組合せ | 572.00 s | 374.90 MiB | 564.00 MiB |
| M01、repeat 2 | 1組合せ | 594.00 s | 374.90 MiB | 564.00 MiB |
| M01、repeat 3 | 1組合せ | 468.02 s | 374.90 MiB | 564.00 MiB |
| M11、repeat 1 | 1組合せ | 505.11 s | 374.90 MiB | 564.00 MiB |
| M11、repeat 2 | 1組合せ | 546.94 s | 374.90 MiB | 564.00 MiB |
| M11、repeat 3 | 1組合せ | 575.82 s | 374.90 MiB | 564.00 MiB |

共有入力SHA-256は15組合せとも
`4d11b22228242221662bbeb0dbe634064963ab98e37deec3f5ef949dedc894e7`で一致した。

### 5.3 主ニューラル、fold 2

#### M11

2026-09-10から2026-09-11にかけて、M11のfold 2・repeat 1–3の学習、全7Kのclustering、
全test評価と各checkを完了した。test対象は10試料・781,665画素である。

| repeat | optimizer updates | AMP skips | training seconds | clustering wall | evaluation wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 249,501 | 99 | 9,663.0157 | 109.95 s | 505.30 s |
| 2 | 249,494 | 106 | 10,193.7247 | 115.95 s | 435.66 s |
| 3 | 249,502 | 98 | 9,926.3471 | 67.60 s | 420.52 s |

repeat 2はepoch 225の途中で`KeyboardInterrupt`により中断し、224 epoch完了時の`last.pt`を
明示して同runを再開した。中断したepochを先頭から再実行し、最終的に800 epochまで完了した。
失敗attemptと完了attemptは両方保存されている。

評価の共有入力SHA-256は3反復とも
`b443abae9970ba54dd39d5fecc03196f82b9181daf0a1e0853ee1aee68702a93`で一致した。

#### A0

2026-09-10から2026-09-11にかけて、A0のfold 2・repeat 1–3の学習、全7Kのclustering、
全test評価と各checkを完了した。test対象はM11と同じ10試料・781,665画素である。

| repeat | optimizer updates | AMP skips | training seconds | clustering wall | recorded evaluation wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 249,506 | 94 | 9,960.2955 | 69.67 s | 1,128.62 s |
| 2 | 249,503 | 97 | 9,908.3479 | 73.10 s | 1,131.90 s |
| 3 | 249,503 | 97 | 9,741.9955 | 72.85 s | 1,135.76 s |

3 runとも800 epoch・249,600 attempted updatesを1回のattemptで完了し、resumeは行っていない。
評価は3 consumerを1回のCLIでまとめて実行した。表の評価時間は各consumer保存までの共同経過時間である。
共有入力SHA-256は3反復とも
`b443abae9970ba54dd39d5fecc03196f82b9181daf0a1e0853ee1aee68702a93`で一致した。

### 5.4 A0、fold 3

2026-09-11から2026-09-12にかけて、A0のfold 3・repeat 1–3の学習、全7Kのclustering、
全test評価と各checkを完了した。test対象は10試料・667,682画素である。

| repeat | optimizer updates | AMP skips | training seconds | clustering wall | recorded evaluation wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 249,505 | 95 | 9,962.6694 | 68.59 s | 1,153.42 s |
| 2 | 249,498 | 102 | 9,864.6774 | 66.79 s | 1,156.94 s |
| 3 | 249,500 | 100 | 9,992.5141 | 120.63 s | 1,160.68 s |

3 runとも800 epoch・249,600 attempted updatesを1回のattemptで完了し、resumeは行っていない。
評価は3 consumerを1回のCLIでまとめて実行した。表の評価時間は各consumer保存までの共同経過時間である。
共有入力SHA-256は3反復とも
`f004c5a1934de561f4fc4f22bf8c006e4871dd741023eea89dfc3b343fbaac0d`で一致した。

### 5.5 A0、fold 4

2026-09-12、A0のfold 4・repeat 1–3の学習、全7Kのclustering、全test評価と各checkを完了した。
test対象は10試料・750,105画素である。

| repeat | optimizer updates | AMP skips | training seconds | clustering wall | recorded evaluation wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 249,503 | 97 | 9,929.3881 | 75.22 s | 1,089.90 s |
| 2 | 249,507 | 93 | 9,927.4507 | 73.99 s | 1,093.48 s |
| 3 | 249,512 | 88 | 9,923.7343 | 72.73 s | 1,097.28 s |

3 runとも800 epoch・249,600 attempted updatesを1回のattemptで完了し、resumeは行っていない。
評価は3 consumerを1回のCLIでまとめて実行した。表の評価時間は各consumer保存までの共同経過時間である。
共有入力SHA-256は3反復とも
`f2031be45cf584a9410770fdfdeef48b1d0298e95f2b463466a999f89581d917`で一致した。

### 5.6 A0、fold 5

2026-09-12、A0のfold 5・repeat 1–3の学習、全7Kのclustering、全test評価と各checkを完了した。
test対象は9試料・796,370画素である。

| repeat | optimizer updates | AMP skips | training seconds | clustering wall | recorded evaluation wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 255,904 | 96 | 10,039.6633 | 73.35 s | 1,231.48 s |
| 2 | 255,900 | 100 | 10,532.9782 | 73.73 s | 1,235.46 s |
| 3 | 255,904 | 96 | 10,204.7269 | 104.51 s | 1,240.07 s |

3 runとも800 epoch・256,000 attempted updatesを1回のattemptで完了し、resumeは行っていない。
評価は3 consumerを1回のCLIでまとめて実行した。表の評価時間は各consumer保存までの共同経過時間である。
共有入力SHA-256は3反復とも
`b237eb9bcba6482f3cf3723b7b24aa604057cb43913713e6436b6dc16cc09df2`で一致した。

### 5.7 B0・B1、全5 folds

2026-09-10、B1の16次元PCAを各foldの共通train画素で1回ずつ、計5回fitし、保存・再読込と由来を
検証した。fold 1～4は39試料・319,488画素、fold 5は40試料・327,680画素を使用した。
実solverは全foldで`covariance_eigh`、`pca_reusable_across_repeats=true`であり、保存・再読込後の
probe最大絶対誤差は全foldで$2.50\times10^{-7}$以下だった。同じfoldのB1 repeat 1～3は、この
repeat 1のPCAを共有する。B0はfitするパラメータを持たない。

工程名と保存先の契約は[runbookのPCA fit](experiment_runbook.md#5-b1-pca-fitとbaseline変換の検証)を参照する。

続いてB0・B1の各5 folds × 3 repeats、計30組合せについて全7Kのclusteringと全test評価を完了した。
各condition・repeatで5 foldsを合わせると49試料・3,902,250有効画素を1回ずつ覆う。

評価は各foldでB0・B1の3反復、計6 consumerをまとめ、同じfold内では共有入力SHA-256が全consumerで
一致した。最終consumer保存までの共同評価実測値は次のとおりである。

| fold | test試料 | test画素 | joint evaluation wall | peak allocated | peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 906,428 | 915.94 s | 4,476.37 MiB | 4,524.00 MiB |
| 2 | 10 | 781,665 | 782.67 s | 3,861.13 MiB | 3,892.00 MiB |
| 3 | 10 | 667,682 | 643.36 s | 3,296.52 MiB | 3,340.00 MiB |
| 4 | 10 | 750,105 | 752.47 s | 3,703.39 MiB | 3,740.00 MiB |
| 5 | 9 | 796,370 | 795.86 s | 3,931.75 MiB | 3,962.00 MiB |

B0は10,290 score rowsが全件definedだった。B1は10,290 rows中10,281 rowsがdefinedで、残る9 rowsは
fold 2のKYOw02789・$K=2$における補正LLA（窓3・5・9、3反復）が`single_cluster`のためundefinedだった。
これは理由付き未定義値として保持し、0で補完しない。B0・B1だけを用いた性能順位は確定せず、
主7条件の全fold・反復が揃ったOOF snapshotで計画比較する。

## 6. 本文代表試料とOOF sanity

代表試料は2026-09-06に、各樹種で保存有効画素数が最大の7試料へ固定した。
試料一覧、選択規則、metadata・manifestのSHA-256は[可視化設計](design/visualization_and_interpretation.md#representative-samples)に集約する。

2026-09-10、B0・B1の既存OOF mapとscoreから、[sanity可視化](design/oof_sanity_visualization.md)を生成した。
保存物は条件別label sheet 2枚、silhouette 1枚、数値CSV 3つで、各label sheetの試料下にKYOw名を表示する。
この作業ではCodexが可視化を実行し、ラベル図を目視確認した。試料名追加後の対象テストは1 passed・11 deselected、
Ruffはpassedだった。この確認は図の出力・配置に関するものであり、劣化との対応は[解釈メモ](interpretation_notes.md)に分ける。
今回の文書整理で学習・評価・可視化を再実行したものではない。

<a id="chemomae-022-validation"></a>

## 7. 実行環境の確認

### 7.1 ChemoMAE v0.2.2の採用

2026-09-07、実行環境をChemoMAE v0.2.2へ更新した。
[v0.2.1](https://github.com/Mantis-Ryuji/ChemoMAE/commit/942804a176750e4f79ee530ca650e0e317efbf90)
から[v0.2.2](https://github.com/Mantis-Ryuji/ChemoMAE/commit/4ec7f6acecb82035c85001f5aee508910d40adac)
へのPythonソース差分は`_version.py`と`clustering/vmf_mixture.py`だけだった。
主実験で使うモデル・loss・Trainer・augmentation・Cosine-KMeans・正規化・silhouetteのソースは同一で、
導入済みv0.2.2の対象20ファイルともCRLFをLFへ揃えたhashが一致した。依存変更はChemoMAEだけだった。

このpackage更新自体では、モデル、loss、augmentation、800 epoch、split、seed、共通K、指標を
変更していない。
vMFの735 fitsは合意済みの補助工程であり、ニューラル再学習を追加せず、主実験のCosine-KMeansも維持する。

### 7.2 更新直後の検証結果

2026-09-07、ユーザーが次のコマンドを実行し、終了code 0と出力を共有した。Codexは再実行していない。

| 確認 | ユーザー実行結果 |
| --- | --- |
| `uv run pytest tests/experiments -q` | **383 passed、43 warnings、98.25秒** |
| `prepare_manifests.py check --experiment-id production_v1` | `validated_existing_manifest`、49試料、5-fold |

manifestはfold 1–4でtrain 39試料・319,488画素、fold 5で40試料・327,680画素だった。
43 warningsはTransformerの`norm_first=True`に伴うnested tensorの通知で、テストの失敗はなかった。
vMFのfitと数値検証は未実行である。
