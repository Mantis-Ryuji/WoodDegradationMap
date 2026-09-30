# Bacteria-ID成果物の整理記録

2026-09-30に[移動・照合ツール](../scripts/experiments/bacteria_id_relocate.py)のdry-run後、
旧出力を同一volume内で移動した。実験条件・分割・採用結果・既存のrun/score JSONは変更していない。

| 役割 | 現在の出力先 |
| --- | --- |
| 旧60/20/20 Table 4（履歴） | `outputs/experiments/bacteria_id_table4_60_20_20_legacy_v1/` |
| 新80/20 Table 4（主結果、移動なし） | `outputs/experiments/bacteria_id_table4_80_20_v1/` |
| Table 5用reference30事前学習 | `outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/` |
| Table 5 HPO・最終結果 | `outputs/experiments/bacteria_id_table5_hpo_cv_v1/` |
| SNVと保存済み旧分割（移動なし） | `data/processed/bacteria_id_v1/` |

移動前の全ファイルの相対パス・サイズ・SHA-256は
`outputs/experiments/bacteria_id_relocation_inventory_20260930.json`に保存した。
移動した3ディレクトリの`relocation_manifest.json`に旧・新の絶対パスと各ファイルの
サイズ・SHA-256を記録した。移動前後の全ファイルSHA-256は一致し、新80/20 Table 4の
166ファイルも移動前後で一致した。元の2出力先は残っていない。

旧Table 4には20件のscore、20件の事前学習完了記録・最終重み、
Table 5用reference30には10件の事前学習完了記録・最終重みが揃う。
Table 5は64件の異なる設定が完了し、ユーザー指定で**最初の60件**から設定を選択した。
未完了の`trial_280`にも4 fold分の記録があり、除外せず保存した。
最終学習10件、test評価10件、主表が揃う。当初の100設定完走計画とは区別する。
既存のcompletion/metricsに記録された重み・クラスタ中心のSHA-256とSQLite studyの
完了trial数も移動前に照合した。移動後の独立した`--verify`も成功した。

既存JSONの`weights_file`等には実行当時の絶対パスが残る。現在位置は各
`relocation_manifest.json`と上表で追跡し、過去の記録を書き換えない。
主結果は新80/20 Table 4とTable 5であり、旧60/20/20 Table 4は履歴である。
旧一括CLI `bacteria_id_all.py`は廃止し、旧単発CLI `bacteria_id.py`は
`--output-dir`の明示指定を必須にした。既存の旧Table 4主表は保存済みの`table4.md`を参照する。

再照合するときは、標準ライブラリだけを使う次のコマンドをリポジトリrootで実行する。
全成果物のSHA-256を読み直すため、約5.5 GBのディスク読み取りが発生する。

```powershell
python scripts/experiments/bacteria_id_relocate.py --verify
```

checkpoint payloadは引き続き[`.gitignore`](../.gitignore)の対象である。
Gitのadd/commit/pushはこの整理では行っていない。
