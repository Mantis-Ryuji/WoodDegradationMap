# ドキュメント案内

研究の問い・採用理由・現行仕様・実行手順をまとめる。論文の原稿と執筆進捗は
`C:\Users\PC_User\Python\Thesis` で管理する。

2026-09-25のユーザー完了報告により、A1を含む主8条件のCV・OOF図表、全体6条件のfit・K8図表・PCAまで完了した。
残るToDoはmask ratio sweepと位置対応FT-IRの2件とし、[ToDo](../ToDo.md)で管理する。
[執筆への引き継ぎ](manuscript_handoff.md)から既存図表を参照できる。未採用の追加可視化案を残作業に含めない。

## 目的別の入口

| 目的 | 文書 | 内容 |
| --- | --- | --- |
| 研究の全体像をつかむ | [研究概要](research_overview.md) | 問い、方法、証拠の対応と日英の説明文 |
| 手法の理由を知る | [ChemoMAEの位置づけ](chemomae_positioning.md)・[関連研究](related_work.md) | 採用理由、先行研究との関係、主張の範囲 |
| 条件と定義を確認する | [研究設計](design/README.md) | 前処理、実験条件、評価、可視化、Open事項 |
| 実験を進める | [ToDo](../ToDo.md) → [runbook](experiment_runbook.md) | 現在の状態、残作業、CLI、再開・完了判定・保存先 |
| 結果を解釈する | [解釈メモ](interpretation_notes.md) | マップ・指標・再構成lossの読み方と限界 |
| 数理・数値処理を確認する | [数理的補足](mathematical_notes.md)・[数値実装の補足](numerical_implementation_notes.md) | 導出、混合精度、loss・inertiaの意味 |
| 論文へ資料を渡す | [執筆への引き継ぎ](manuscript_handoff.md) | 出典、図表の扱い、原稿で確認する事項 |

## 文書の管理

- 研究条件・定義・データ契約は`docs/design/`を正とする。
- 進捗はToDo、操作はrunbookに集約する。実測値と実行由来は`outputs/`の各成果物を参照する。
- 完了記録の読み合わせ、過去の実行・検証結果、今回新たに実行した検証を区別する。
- 仕様を更新するときは該当箇所へ直接反映し、変更ログ・旧仕様・不採用案を残さない。重複する詳細は定義先へリンクする。
- **Fixed**は採用済み、**Open**は未確定を表す。実装・実行・検証の完了状況とは区別する。

## 用語

| 用語 | 本リポジトリでの意味 |
| --- | --- |
| raw SNV / B0 | 前処理済み256次元SNVを直接使うbaseline。センサのraw強度ではない |
| 試料 | `KYOw...`で識別する分割・集計単位。同一原材との関係は確認できた範囲で扱う |
| repeat | 同じsplit・画素集合に対する3回の学習・クラスタリング反復 |
| draw | LFR用に各摂動を5回生成する評価摂動反復 |
| OOF | 各試料をtrainに含めないfoldモデルから得る予測・評価 |
| 全体fit | 全49試料からの共通抽出画素でfit・学習し、全試料を記述する解析 |
| matching | クラスタ所属を変えずに表示番号を対応づける操作。劣化順序や共通の化学的意味を与えない |
| 物理化学的な特徴 | 化学組成に限定せず、組織・表面性状・散乱なども含めた解釈対象。クラスタやCV指標から原因を同定したことは意味しない |
| clean | 追加noise・shiftを加える前の観測。測定ノイズのない真値ではない |

可視化の仕様は[OOF sanity](design/oof_sanity_visualization.md)と
[全体fit後の可視化](design/visualization_and_interpretation.md)で使い分ける。
