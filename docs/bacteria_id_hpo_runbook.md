# Bacteria-ID Table 5 実行記録と成果物

[分類条件](design/bacteria_id_hpo_protocol.md)に従う。reference30の事前学習は
`outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/`、HPO・最終学習・評価は
`outputs/experiments/bacteria_id_table5_hpo_cv_v1/`に保存した。
SNV入力と分割は`data/processed/bacteria_id_v1/`にある。
Table 4の主結果は[80/20手順](bacteria_id_runbook.md)で別に生成した。

当初はM11 seed 0で100設定×5-fold CVを予定した。実際には64件の異なる設定が完了し、
ユーザー指定で最初の60件からepoch 50のvalidation Accuracy平均が最大の設定を選んだ。
未完了の`trial_280`に残る4 fold分の記録もそのまま保存した。
採用設定は`selected.json`のtrial 38（$k=8$、head LR $3\times10^{-4}$、
encoder LR $8\times10^{-5}$、batch 8、TGN/FSともoff、weight decay $10^{-4}$）。
M00/M11×5 seedをそれぞれ全3,000件で50 epoch学習し、10件のtest評価を終えた。
既報値を添えた主表は`table5.md`、個別seed値と探索記録は`table5_statistics.json`、
学習重みは`final/`、test値は`evaluation/`にある。

旧出力先からの移動と全ファイルの照合結果は
[成果物整理記録](bacteria_id_artifact_reorganization_plan.md)を参照する。
既存JSONの絶対パスは実行時の位置を表し、移動後の位置は各`relocation_manifest.json`で追跡する。
現行CLIの既定出力先は上記の新ディレクトリに変更済みである。

成果物の再照合は次のコマンドで行う。標準ライブラリのみを使い、約5.5 GBを読み取る。

```powershell
python scripts/experiments/bacteria_id_relocate.py --verify
```

既存の`search`を再実行すると当初上限の100設定へ向けた探索が続くため、
確定した60設定選択・主表の再現確認には使わない。新しい探索をする場合は、
出力先と選択規則を別途定めてから実行する。
