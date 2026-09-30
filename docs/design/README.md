# 研究設計

本ディレクトリは研究条件・計算方法・データ契約を管理する。
**Fixed**は採用済みの仕様、**Open**は実施前に決定が必要な事項を表す。Fixedは実装・実行済みという意味ではない。
**本研究の補完条件**は、参照論文の未記載事項をこちらで具体化した採用条件を表す。原論文の実設定とは区別する。
研究目的は[研究の目的と説明文](../research_overview.md)、進捗は[ToDo](../../ToDo.md)を参照する。

## 設計文書と定義の置き場所

| 文書 | 状態 | 定義する内容 |
| --- | --- | --- |
| [前処理](preprocessing.md) | Fixed | 200 Hz入力、mask、負値除外、256点補間、SNV、保存schema |
| [実験プロトコル](experiment_protocol.md) | 主条件・mask率補助Fixed | 条件、split、seed、共通画素、学習、クラスタリング、実行記録 |
| [Bacteria-ID補助実験計画](bacteria_id_experiment_plan.md) | 事前学習・Table 4 Fixed | RamanのM00/M11、出版版SMAE Table 4・5との報告値比較、事前学習とクラスタリング |
| [Bacteria-ID Table 5 HPO](bacteria_id_hpo_protocol.md) | 分類条件・実績 | M11の5-fold CVで64設定完了、先頭60設定で選択し、M00/M11を全件学習してtest評価 |
| [評価指標](evaluation_metrics.md) | Fixed / 任意形状診断Open | LLA（補正後）、LFR、ARI、silhouette、occupancy、補正前LLA、未定義値、集約、比較・報告 |
| [OOF sanity可視化](oof_sanity_visualization.md) | Fixed・実装済み | B0・B1・A0・M00のPNG・CSV、fold内B0基準のmatching |
| [全体可視化と解釈](visualization_and_interpretation.md) | 全体fit・K8比較・PCA一枚Fixed / 詳細図・連続指標map・FT-IR等Open | M00基準のmatching、代表例、試料別／全体スペクトル、PC1/PC2、追加解析候補 |

## 条件の参照先

固定値は次の定義先を正とする。

- データ：[CV・seed](experiment_protocol.md#cv-design)、[共通画素抽出](experiment_protocol.md#pixel-sampling)。
- モデル：[比較条件](experiment_protocol.md#conditions)、[表現・PCA](experiment_protocol.md#representations)、[構成](experiment_protocol.md#model-architecture)、[学習recipe](experiment_protocol.md#training-recipe)。
- 操作と評価：[augmentation・クラスタリング](experiment_protocol.md#augmentation-clustering)、[数値精度](experiment_protocol.md#evaluation-precision)、[K](experiment_protocol.md#cluster-counts)、[指標・集約](evaluation_metrics.md)。

## 主実験・補助実験・解釈

| 区分 | 対象 | 追加NN学習 | クラスタリング |
| --- | --- | ---: | --- |
| 主実験 | B0、B1、A0、A1、M00、M10、M01、M11の5-fold・3反復 | 90回 | Cosine-KMeans、全7K |
| Bacteria-ID補助 | RamanのM00・M11。出版版SMAE Table 4・5へ追加 | 旧事前学習30回は記録として保持。新80/20 Table 4の20回とTable 5のHPO・分類は完了 | 新Table 4はCosine-KMeans、K=4/6、計20 fits完了 |
| Mask率補助 | M11-25・M11-75。50%はM11を再利用 | 30回 | Cosine-KMeans、全7K |
| OOF sanity | 完了済みB0・B1・A0・M00のOOF map・指標 | 0回 | 既存成果物のみ |
| 全体解釈 | B0、B1、A0、A1、M00、M11を全49試料でfit・学習 | 4回 | Cosine-KMeans、$K_0=8$、6 fits |

実験の完了状態は[ToDo](../../ToDo.md)、全体fitのmatching・表示規約は
[可視化設計](visualization_and_interpretation.md)を参照する。
A1は既存7条件の結果を得た後に追加したablationであり、主比較との区別は
[実験プロトコル](experiment_protocol.md#planned-comparisons)に定義する。

<a id="open-items"></a>

## Open事項と実施範囲

Open事項のうち現在の残作業に含むのは位置対応FT-IR（mask ratio sweepの条件はFixed）。
Bacteria-IDの未記載条件は[本研究の補完条件](bacteria_id_experiment_plan.md#open-items)として具体化済み。
参照資料間の不一致は文献比較の留保として記載し、実装・実行の待機理由にはしない。
以下のその他の項目は未採用の候補であり、完了したことや追加実施が必要なことを意味しない。

| Open事項 | 実施前に決める・確認する内容 | 定義先 |
| --- | --- | --- |
| 任意の形状診断 | 採用する場合の近傍・connectivity・閾値・分母 | [診断](evaluation_metrics.md#occupancy) |
| 試料ごとの詳細図 | M11・K8で注目する試料・領域、示したい差、panel構成。試料別平均CSVは既存成果物を利用可能 | [スペクトルの要約](visualization_and_interpretation.md#representative-spectra) |
| PCAの追加色分け | PC1/PC2のPNG一枚は生成済み。追加metadata・帯域指標による図の採否・内容は執筆時に判断 | [表現空間の可視化](visualization_and_interpretation.md#latent-spectral-maps) |
| 連続スペクトル指標map | 採否、3×3平均またはGaussianの選択・数値設定、帯域選択、積分・符号・共通color scale。クラスタ所属で近傍を制限しない | [帯域選択](visualization_and_interpretation.md#spectral-band-selection) |
| 位置対応FT-IR | 対象、位置対応、測定・反復条件、前処理・指標、解釈範囲 | [FT-IR](visualization_and_interpretation.md#ftir-interpretation) |
| 正式な目視評価 | 評価者、rubric、条件名・提示順、意見不一致の扱い | [証拠の統合](visualization_and_interpretation.md#evidence-triangulation) |

ライブラリの既定値でOpen事項を暗黙に埋めない。
探索的な原因仮説は[解釈メモ](../interpretation_notes.md)で扱い、実験条件として採用したことにはしない。

## 評価と解釈の範囲

CVは空間的一貫性・指定摂動への安定性を比較し、分離性・反復間再現性・退化を診断する。
全体fitのマップ・スペクトルと位置対応FT-IRは化学的解釈のために用いる。
劣化検出精度・化学成分量・他材料への有効性の主張範囲は[ChemoMAEの位置づけ](../chemomae_positioning.md)を参照する。
