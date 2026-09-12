# 研究設計

本ディレクトリは研究条件・計算方法・データ契約を管理する。
**Fixed**は採用済みの仕様、**Open**は実施前に決定が必要な事項を表す。Fixedは実装・実行済みという意味ではない。
研究目的は[研究の目的と説明文](../research_overview.md)、進捗は[ToDo](../../ToDo.md)を参照する。

## 設計文書と定義の置き場所

| 文書 | 状態 | 定義する内容 |
| --- | --- | --- |
| [前処理](preprocessing.md) | Fixed | 200 Hz入力、mask、負値除外、256点補間、SNV、保存schema |
| [実験プロトコル](experiment_protocol.md) | 主条件Fixed / vMF数値仕様Open | 条件、split、seed、共通画素、学習、クラスタリング、実行記録 |
| [評価指標](evaluation_metrics.md) | Fixed / 任意形状診断Open | LLA、LFR、silhouette、ARI、occupancy、未定義値、集約、比較 |
| [OOF sanity可視化](oof_sanity_visualization.md) | Fixed・実装済み | B0・B1のPNG・CSV、fold内B0基準のmatching |
| [全体可視化と解釈](visualization_and_interpretation.md) | Fixed / 任意責務マップ・FT-IR・正式目視評価の詳細Open | 全体fit、A0基準のmatching、代表例、スペクトル、FT-IR計画 |
| [表現幾何の補助診断の見直し](representation_geometry_diagnostics.md) | 現行計画から除外 / 未採用候補 | 既定評価との関係、実施項目から外した理由、数理的補足と旧案 |
| [設計上の決定記録](decisions.md) | 記録 | 決定日、CV開始後の追加事項と適用範囲 |

## 条件の参照先

固定値は次の定義先を正とする。

- データ：[CV・seed](experiment_protocol.md#cv-design)、[共通画素抽出](experiment_protocol.md#pixel-sampling)。
- モデル：[比較条件](experiment_protocol.md#conditions)、[表現・PCA](experiment_protocol.md#representations)、[構成](experiment_protocol.md#model-architecture)、[学習recipe](experiment_protocol.md#training-recipe)。
- 操作と評価：[augmentation・クラスタリング](experiment_protocol.md#augmentation-clustering)、[数値精度](experiment_protocol.md#evaluation-precision)、[K](experiment_protocol.md#cluster-counts)、[指標・集約](evaluation_metrics.md)。

## 主実験・補助実験・解釈

| 区分 | 対象 | 追加NN学習 | クラスタリング |
| --- | --- | ---: | --- |
| 主実験 | B0、B1、A0、M00、M10、M01、M11の5-fold・3反復 | 75回 | Cosine-KMeans、全7K |
| Mask率補助 | M11-25・M11-75。50%はM11を再利用 | 30回 | Cosine-KMeans、全7K |
| vMF補助 | 主7条件の既存表現、5-fold・3反復 | 0回 | vMF、全7Kの735 fits |
| OOF sanity | 完了済みB0・B1のOOF map・指標 | 0回 | 既存成果物のみ |
| 全体解釈 | B0、B1、A0、M00、M11を全49試料でfit・学習 | 3回 | $K_0=8$、各手法5 fits |

CVは105学習、全体解釈を含めると108学習。PCA、KMeans、vMF、表現抽出、評価摂動はこの学習数に含めない。
全体fitの表示番号はA0のCosine-KMeansへ直接整列する。B0・B1 OOF sanityはfold内B0基準とする。
CV開始後の対象拡張・追加決定は[決定記録](decisions.md)を参照する。

<a id="open-items"></a>

## 残るOpen事項

| Open事項 | 実施前に決める・確認する内容 | 定義先 |
| --- | --- | --- |
| vMF数値仕様・実装 | 精度、EM停止条件、集中度設定、修正版検証、専用pipeline。利用版・範囲・退化成分の扱いはFixed | [vMF](experiment_protocol.md#vmf-supplementary) |
| 任意の形状診断 | 採用する場合の近傍・connectivity・閾値・分母 | [診断](evaluation_metrics.md#occupancy) |
| 任意の責務マップ | 採用する場合の表示範囲・配色・背景 | [責務マップ](visualization_and_interpretation.md#vmf-responsibility-maps) |
| 位置対応FT-IR | 対象、位置対応、測定・反復条件、前処理・指標、解釈範囲 | [FT-IR](visualization_and_interpretation.md#ftir-interpretation) |
| 正式な目視評価 | 評価者、rubric、条件名・提示順、意見不一致の扱い | [証拠の統合](visualization_and_interpretation.md#evidence-triangulation) |

ライブラリの既定値でOpen事項を暗黙に埋めない。
探索的な原因仮説は[解釈メモ](../interpretation_notes.md)で扱い、実験条件として採用したことにはしない。
付録B.5の表現幾何診断は現行の実施項目・Open事項から外した。[判断理由と旧案](representation_geometry_diagnostics.md)を参照する。

## 評価と解釈の範囲

CVは空間的一貫性・指定摂動への安定性を比較し、分離性・反復間再現性・退化を診断する。
全体fitのマップ・スペクトルと位置対応FT-IRは化学的解釈のために用いる。
劣化検出精度・化学成分量・他材料への有効性の主張範囲は[ChemoMAEの位置づけ](../chemomae_positioning.md)を参照する。
