# 研究・実装 ToDo

残作業はmask率補助実験と位置対応FT-IRである。条件は[研究設計](docs/design/README.md)、
操作は[runbook](docs/experiment_runbook.md)、論文へ渡す資料は[引き継ぎ](docs/manuscript_handoff.md)を参照する。

## 完了済み成果物

| 対象 | 状態と参照先 |
| --- | --- |
| 主8条件のCV・OOF | 5 folds×3反復のNN 90 runs、クラスタリング・評価120組合せ。主図表は`outputs/experiments/production_v1/results/figures/main_oof_v1/` |
| 全体fit・K8 | B0・B1・A0・A1・M00・M11の6条件で全49試料のマップ・スペクトル・PCAを生成。`outputs/experiments/global_v1/` |
| Bacteria-ID Table 4 | M00/M11×Bacteria-4/6×5 seedの新80/20結果を[主表](outputs/experiments/bacteria_id_table4_80_20_v1/table4.md)に集計。旧60/20/20は履歴として保存 |
| Bacteria-ID Table 5 | M11の5-fold CVで64設定完了、先頭60設定から共通設定を選択。M00/M11×5 seedの最終結果は[主表](outputs/experiments/bacteria_id_table5_hpo_cv_v1/table5.md)。[成果物整理記録](docs/bacteria_id_artifact_reorganization_plan.md)に移動・ハッシュ照合を記載 |

## 3. 全体fitと解釈

全体fitの6条件を比較し、M11・K8の個別試料の領域差を中心に記述する。
CVの未知試料評価と全体fitによる記述的マップを区別する。
成果物・確認コマンドは[全体fit runbook](docs/experiment_runbook.md#global-fit-pipeline)、
解釈の範囲は[可視化設計](docs/design/visualization_and_interpretation.md)に従う。

## 4. Mask率補助実験

主条件M11の50%を再利用し、25%・75%を各5 folds×3反復で比較する。
中断・完了runとcheckpointの状態は再開前に確認する。

- [ ] 中断・完了runと再開可能なcheckpointを照合する。
- [ ] M11-25とM11-75を各15 runs、800 epochと後続のクラスタリング・評価まで完了する。
- [ ] 50%の既存M11と合わせて`mask_rate_oof_v1`を作成・checkする。
- [ ] mask率依存性の図表を生成・照合する。

主条件とは別のOOF snapshotとして報告する。結果を見て主条件M11やKを選び直さない。
[固定条件と実施手順](docs/experiment_runbook.md#mask-rate-sweep)を参照する。

## 5. FT-IR

- [ ] 対象試料・領域、NIRとの位置対応、測定と反復の条件を確定する。
- [ ] 測定・前処理・比較指標を実施し、化学的対応と解釈の限界を記録する。

[位置対応FT-IRの設計](docs/design/visualization_and_interpretation.md#ftir-interpretation)に従う。
原稿の執筆進捗は`C:\Users\PC_User\Python\Thesis`で管理する。
