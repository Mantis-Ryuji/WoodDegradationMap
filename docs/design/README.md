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
| [全体可視化と解釈](visualization_and_interpretation.md) | Fixed / 任意責務マップ・FT-IR詳細Open | 全体fit、A0基準のmatching、代表例、スペクトル、FT-IR計画 |
| [設計上の決定記録](decisions.md) | 記録 | 決定日、CV開始後の追加事項と適用範囲 |

## 固定条件の早見表

| 項目 | 要点 | 詳細 |
| --- | --- | --- |
| 入力・分割 | 採用49試料、KYOw単位のランダム5-fold。同一splitで3反復、樹種層化なし | [CV](experiment_protocol.md#cv-design) |
| 画素 | trainは各試料8192画素を一様・非復元抽出し、全条件・K・反復で共有。testは全有効画素 | [抽出](experiment_protocol.md#pixel-sampling) |
| 表現 | B0はSNV 256次元、PCA・NNは16次元。クラスタリングへ渡す表現をL2正規化 | [表現](experiment_protocol.md#representations) |
| PCA | train平均中心化、追加autoscalingなし、whiten=False、solver=auto | [PCA](experiment_protocol.md#representations) |
| NN | CLS由来16次元の単位潜在、線形1層decoder、全可視で表現抽出 | [構成](experiment_protocol.md#model-architecture) |
| 学習 | 800 epoch、単一GPU、batch 1024、AdamW、peak lr $6\times10^{-4}$、40 epoch warmup | [学習設定](experiment_protocol.md#training-recipe) |
| augmentation | noise角度$U(0,5^\circ)$、shift幅$U(-2,2)$チャネル。各操作後に平均とnormを復元 | [操作仕様](experiment_protocol.md#augmentation-clustering) |
| K | 共通集合$\{2,4,6,8,10,12,14\}$、代表表示$K_0=8$。結果による選び直しなし | [Kの方針](experiment_protocol.md#cluster-counts) |
| 数値設定 | 学習はFP16 AMP。抽出・評価はFP32、抽出AMP・評価TF32は無効 | [数値設定](experiment_protocol.md#evaluation-precision) |
| 主評価 | LLA-3/5/9、LFR noise・shift・両方（各5 draws） | [評価](evaluation_metrics.md) |
| 集約 | OOF試料macro、paired差、試料間SDと3反復間SDを分離。有意差検定・総合scoreなし | [報告](evaluation_metrics.md#reporting) |

主要比較はM11対B0・B1・M00、構成要素の説明はAE/MAE比較とnoise・shiftの2×2 ablationで行う。
条件IDと全contrastは[実験プロトコル](experiment_protocol.md#conditions)を正とする。
augmentation強度はSNVスペクトルのsanity checkを通じて恣意的に固定した値であり、
実測誤差の同定値やCV指標で最適化した値ではない。

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

ライブラリの既定値でOpen事項を暗黙に埋めない。
探索的な原因仮説は[解釈メモ](../interpretation_notes.md)で扱い、実験条件として採用したことにはしない。

## 評価と解釈の範囲

CVが比較するのは未知試料の分離性・空間的一貫性・指定摂動への安定性・反復間再現性である。
劣化検出精度や化学成分量は直接評価していない。全体fitのマップとスペクトルは探索的に解釈する。
位置対応FT-IRはその解釈を深める計画で、詳細・結果は未確定。他材料への有効性も未検証である。
採用理由と主張の範囲は[ChemoMAEの位置づけ](../chemomae_positioning.md)を参照する。
