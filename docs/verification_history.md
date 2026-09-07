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

### 5.1 A0・M00、fold 1

2026-09-06から2026-09-07にかけて、A0・M00のfold 1・repeat 1–3、計6 runの学習と
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

6 runすべてで`optimizer_updates + amp_skips == attempted_updates`が成立し、nonzero LR updatesは
optimizer updatesより1少なかった。`training_seconds`は各epochの処理時間の合計であり、CLI全体の
wall timeではない。本番学習のcompletionはGPU peakを保存しないため、第4.1節のpreflight値で補わない。

A0・repeat 2はepoch 26のcheckpoint保存後、`checkpoint.json.tmp`から`checkpoint.json`への置換で
Windowsの`PermissionError`が発生した。`last.pt`、一時記録、training historyのepoch・更新数・
SHA-256を照合し、同じcheckpointを明示してepoch 27から再開した。失敗attemptと完了attemptは
両方保存され、最終completionと重みのhashも一致した。

### 5.2 clean test map・全test評価

6組合せすべてでfold 1のtest 10試料・906,428画素と
$K\in\{2,4,6,8,10,12,14\}$を処理した。clusteringは各runで
`clean_test_maps_completed`・`checks_passed=true`、保存済み成果物は
`validated_existing_clustering`だった。GPU peak allocated / reservedは全runで
255.81 / 388.00 MiBだった。

2026-09-08、6組合せを同じ固定configと共通摂動でまとめて評価した。各runで
`full_test_evaluation_completed`・`checks_passed=true`、保存済み成果物は
`validated_existing_evaluation`だった。最後のconsumer保存時点の共同評価経過時間は
2,672.56秒、GPU peak allocated / reservedは495.77 / 684.00 MiBだった。この時間は
source検証を除く共同評価開始から各consumer保存までの累積値であり、run別の処理時間ではない。

共有入力SHA-256は6組合せとも
`4d11b22228242221662bbeb0dbe634064963ab98e37deec3f5ef949dedc894e7`で一致した。
これは実行・保存契約の確認であり、条件間の性能比較は全fold・反復が揃ったOOF snapshotで行う。

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
