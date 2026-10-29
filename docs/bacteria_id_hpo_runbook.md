# Bacteria-ID Table 5の実行手順

[分類プロトコル](design/bacteria_id_hpo_protocol.md)に従う、reference30の事前学習、
CVでの設定選択、最終学習・評価の手順を示す。入力準備は[Table 4 runbook](bacteria_id_runbook.md#入力と準備)を参照する。
すべてリポジトリrootから、構築済み環境でPowerShellを使って実行する。
各コマンドの終了codeが非0なら後続工程へ進まない。

## 入力と保存先

| 用途 | パス |
| --- | --- |
| SNV入力・分割 | `data/processed/bacteria_id_v1/` |
| reference30の事前学習 | `outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/` |
| HPO・最終学習・test評価 | `outputs/experiments/bacteria_id_table5_hpo_cv_v1/` |

事前学習と最終学習は単一CUDA GPUを使用する。HPOは1設定につき5 foldを各50 epoch学習するため、
長時間のGPU処理となる。出力には学習重み、run記録、Optuna study、評価値を含む。
`outputs/`と重みはGit管理対象外であり、保存済み実験を確認するには成果物を別途配置する。
保存JSON内の絶対パスと現在位置の対応が必要な場合は、各出力先の`relocation_manifest.json`を参照する。

## reference30の事前学習

M00/M11それぞれseed 0–4で、referenceの60,000件だけを使って800 epoch学習する。
Table 4用checkpointは使用しない。以下は1 runの例である。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id.py pretrain `
    --corpus reference30 --condition M11 --seed 0 --device 0 `
    --output-dir outputs/experiments/bacteria_id_table5_reference30_pretrain_v1
```

新規runだけを`pretrain`へ渡す。中断した同じrunは、そのrunの`checkpoints/last.pt`を
`--resume`に明示して再開する。完了済みrunの再学習・上書きは行わない。

## CVによる設定選択

選択対象、探索空間、seed、epochと選択に用いた記録の範囲は
[分類プロトコル](design/bacteria_id_hpo_protocol.md#設定選択)に従う。
保存済みstudyから採用設定を確認・固定するコマンドは次のとおり。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py select --completed-trials 60
```

`select`はCV学習を追加せず、完了設定をtrial番号順に並べた先頭60件を評価し、
`selected.json`に保存する。既存の選択と異なる場合は停止する。
採用設定はtrial 38であり、以下をM00/M11に共通適用する。

| 項目 | 採用値 |
| --- | --- |
| 更新する末尾block数 $k$ | 8 |
| head学習率 | $3\times10^{-4}$ |
| encoder block学習率 | $8\times10^{-5}$ |
| batch size | 8 |
| TGN / FS | off / off |
| weight decay | $10^{-4}$ |

新規studyの探索には`bacteria_id_hpo.py search`を使うが、実装上限は100設定である。
このコマンドは保存済み60設定選択の照合ではない。cutoffによる選択が確定した出力先では、
`search`は停止する。探索の乱数履歴・実行環境を含む再現には、保存されたstudyとrun記録を参照する。

## 最終学習・test評価・集計

採用設定でM00/M11×5 seedをfinetuneの全3,000件に適応させる。
50 epochの最終重みだけを使い、配布test 3,000件を各runで一度採点する。

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py train --device 0
if ($LASTEXITCODE -ne 0) { throw 'Table 5 final training failed' }
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py evaluate --device 0
if ($LASTEXITCODE -ne 0) { throw 'Table 5 evaluation failed' }
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py tables
if ($LASTEXITCODE -ne 0) { throw 'Table 5 reporting failed' }
```

学習・評価の再開可否は各runの契約と完了記録で判定する。出力先や採用設定を書き換えて再利用しない。
集計だけを行う場合は`tables`を使う。非採用設定をtestで採点しない。

## 成果物と確認点

HPO出力先にある次の記録を対応付ける。

| 保存物 | 内容 |
| --- | --- |
| `selected.json` | 採用trial、選択対象、fold別・平均validation Accuracy、設定 |
| `cv_splits.json` | 各foldのtrain/validation index |
| `final/` | M00/M11×5 seedの分類学習記録・重み |
| `evaluation/` | 各seedのtest評価 |
| `table5.md` | 文献値と本研究の値を併記した主表 |
| `table5_statistics.json` | seed別値、平均・95%信頼区間、探索結果 |

各seedのAccuracyと、5 seedの標本SDから求めた自由度4のt信頼区間を照合する。
文献値との条件差、M11で選んだ共通設定の意味は[結果の解釈](bacteria_id_interpretation_plan.md)を参照する。
