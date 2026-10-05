# 研究・実装 ToDo

残る実験は位置対応FT-IRである。条件は[研究設計](docs/design/README.md)、
操作は[runbook](docs/experiment_runbook.md)、論文へ渡す資料は[引き継ぎ](docs/manuscript_handoff.md)を参照する。

## 完了済み成果物

| 対象 | 状態と参照先 |
| --- | --- |
| 主8条件のCV・OOF | 5 folds×3反復のNN 90 runs、クラスタリング・評価120組合せ。主図表は`outputs/experiments/production_v1/results/figures/main_oof_v1/` |
| 全体fit・K8 | B0・B1・A0・A1・M00・M11の6条件で全49試料のマップ・スペクトル・PCA座標を生成。`outputs/experiments/global_v1/` |
| Mask率補助・CV | M11-25・M11-75の追加NN 30 runsと既存M11を合わせ、49試料・45 source runs・全7Kを集計。`outputs/experiments/production_v1/results/figures/mask_rate_oof_v1/` |
| Mask率補助・全体fit | M11-75を1回学習し、固定表現でK8を3 seed比較。M11(-50)との2×7図と数値表は`outputs/experiments/global_v1/results/figures/global_mask_rate_k8_v1/`。付録用 |
| Bacteria-ID Table 4 | M00/M11×Bacteria-4/6×5 seedの新80/20結果を[主表](outputs/experiments/bacteria_id_table4_80_20_v1/table4.md)に集計。旧60/20/20は履歴として保存 |
| Bacteria-ID Table 5 | M11の5-fold CVで64設定完了、先頭60設定から共通設定を選択。M00/M11×5 seedの最終結果は[主表](outputs/experiments/bacteria_id_table5_hpo_cv_v1/table5.md)。[成果物整理記録](docs/bacteria_id_artifact_reorganization_plan.md)に移動・ハッシュ照合を記載 |

## 3. 全体fitと解釈

全体fitの6条件を比較し、M11・K8の個別試料の領域差を中心に記述する。
CVの未知試料評価と全体fitによる記述的マップを区別する。
成果物・確認コマンドは[全体fit runbook](docs/experiment_runbook.md#global-fit-pipeline)、
解釈の範囲は[可視化設計](docs/design/visualization_and_interpretation.md)に従う。
PCA座標は保存済みだが、2026-10-05の文書整理時点でPCA図の`pca-latent-2d/`は現行出力に存在しない。
掲載する場合は[再描画手順](docs/experiment_runbook.md#global-pca)を使う。今回の文書更新では再生成していない。

## 4. Mask率補助実験

保存済みの学習・OOF・図表の完了記録に基づき、次の成果物は完成している。

- [x] CVのM11-25とM11-75を各15 runs、800 epochと後続のクラスタリング・評価まで完了する。
- [x] 50%の既存M11と合わせて`mask_rate_oof_v1`を集計し、完了記録の`checks_passed=true`を確認する。
- [x] CVのmask率依存性をPNG 1枚・CSV 2個と出典・完了記録にまとめる。
- [x] 全体fitのM11-75を既存50%と同じ抽出画素・学習設定・事前固定seedで1回学習する。
- [x] 表現を固定したK8・3 seedのLLA・silhouette・使用クラスタ数・最大占有率を比較する。
- [x] 初回fitの固定7試料を2×7で並べ、上段を`M11(-50)`、下段を`M11-75`とした付録用図を生成する。

CVは主条件と別のOOF snapshot、全体fitは付録の記述的比較として報告する。
結果を見て主条件M11やKを選び直さない。図の生成と正式な目視評価の実施は区別する。
[CV手順](docs/experiment_runbook.md#mask-rate-sweep)、[全体fit手順](docs/experiment_runbook.md#global-mask-rate)、
[Thesisへ渡すファイルと完了記録](docs/manuscript_handoff.md#mask-rate-handoff)を参照する。

## 5. FT-IR

- [ ] 対象試料・領域、NIRとの位置対応、測定と反復の条件を確定する。
- [ ] 測定・前処理・比較指標を実施し、化学的対応と解釈の限界を記録する。

[位置対応FT-IRの設計](docs/design/visualization_and_interpretation.md#ftir-interpretation)に従う。
原稿の執筆進捗は`C:\Users\PC_User\Python\Thesis`で管理する。
