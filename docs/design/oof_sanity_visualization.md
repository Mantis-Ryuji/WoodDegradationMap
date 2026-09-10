# B0・B1 OOF sanity可視化

**Fixed / 実装・生成済み（2026-09-10）**

B0・B1の全CV完了後に、既存のclean OOF mapと評価成果物から探索的な確認図を作る。
PCA・KMeansのfit、表現抽出、評価を再実行しない。実行方法は[runbook](../experiment_runbook.md#oof-sanity)、
観察した所見は[解釈メモ](../interpretation_notes.md#b0-b1-observations)を参照する。

## 1. 保存物

出力先は `outputs/sanity_checks/b0_b1_oof_visualization/`。
保存物は次の3枚のPNGと3つのCSVだけとし、sanity用のログやrun/completion/failure JSONは作らない。
過去の表示形式との互換性は持たせない。

| ファイル | 内容 |
| --- | --- |
| `labels/B0_representatives_k8_repeat1.png` | B0の代表7試料を1行に並べたlabel map |
| `labels/B1_representatives_k8_repeat1.png` | 同じ7試料・順序のB1 label map |
| `silhouette_k_sweep.png` | 単一axesのK-sweep。下段subplotなし |
| `occupancy.csv` | 条件・試料・fold・repeat・K別のcluster画素数、使用数、最大占有率、単一cluster flag |
| `matching.csv` | foldごとのB1 raw IDからB0表示IDへの対応、contingency、overlap |
| `metrics_summary.csv` | silhouette、LLA-3/5/9、補正LLA-3/5/9、LFR noise・shift・両方の集計 |

LLA、LFR、occupancyはCSVで確認する。boundary mapと代表スペクトルは、この可視化の対象に含めない。

## 2. 代表試料のlabel map

$K_0=8$、repeat 1を使用し、[事前固定した7試料](visualization_and_interpretation.md#representative-samples)を
クリ、ケヤキ、スギ、ツガ、ヒノキ、マツ、モミの順に並べる。
試料ごとの図には分割せず、B0とB1を別々のPNGへ保存する。

各試料の下に`KYOw...`試料IDを表示し、背景0・クラスタ1〜8の共通凡例を付ける。
[共通描画規約](visualization_and_interpretation.md#figure-style)に従い、figure titleとaxes titleは付けない。

## 3. fold内Hungarian matching

各foldの全test試料について、B0・B1の共通有効画素のcontingencyを合算し、一致画素数を最大にする
1対1対応を求める。B0を基準にB1を整列し、同じfoldの全試料で同じ対応を使う。
代表7試料だけでmatchingしたり、試料ごとに別の対応を使ったりしない。

対応は表示用mapのコピーに適用する。保存済みCVラベル・指標は変更せず、occupancyのcluster別画素数はraw ID順とする。
fold間の同じ色が同じ状態を表すとは限らない。overlapが0または弱い対応はCSVで確認する。
全体fit後の[共通基準A0＋Cosine-KMeans](visualization_and_interpretation.md#matching-reference)とは適用範囲が異なる。

## 4. 数値集計とsilhouette

全49試料・5 folds・3反復・全7Kを対象とし、[評価の集計規約](evaluation_metrics.md#reporting)を使う。
条件・K・指標ごとに3反復共通の定義済み試料を集計し、平均、試料間SD、反復間SD、
反復別平均、有効対象数、除外試料と未定義理由をCSVへ保存する。未定義値を0へ置換しない。

silhouette図の太線は3反復の試料macro平均、細線は各反復のmacro平均、帯は反復間SD（`ddof=1`）。
平均の対象試料と未定義理由はCSVで確認する。この図を条件・K・best runの選択に使わない。

## 5. 入出力の扱い

入力のcompletion・score・OOF割当・map対応を検証し、既存CV成果物を変更しない。
出力先が既に存在する場合は拒否する。再生成では明示した新規出力先を使う。
全体fit後の解釈図や全CV完了後の論文用図表とは別の成果物として扱う。
