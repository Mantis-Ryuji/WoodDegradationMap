# 研究設計

本ディレクトリは、論文に対応する研究条件、計算方法、データ契約を定義する。
研究全体の位置づけは[研究概要](../research_overview.md)、操作方法は[実験手順](../experiment_runbook.md)を参照する。

## 仕様の参照先

| 文書 | 定義する内容 |
| --- | --- |
| [前処理](preprocessing.md) | 200 Hz入力、試料mask、反射率、画素除外、256点補間、SNV、保存schema |
| [実験プロトコル](experiment_protocol.md) | 比較条件、split、seed、共通画素、モデル、学習、クラスタリング |
| [評価指標](evaluation_metrics.md) | LLA、LFR、ARI、silhouette、occupancy、未定義値、集計、対応差 |
| [OOF診断図](oof_sanity_visualization.md) | B0・B1・A0・M00のOOF map、fold内B0基準の表示整列 |
| [全体fitの可視化・解釈](visualization_and_interpretation.md) | M00基準の表示整列、代表試料、観測スペクトル、PCA、NIR解釈 |
| [空間マップ再現性](spatial_map_reproducibility_plan.md) | M11・K8のencoderとKMeansの反復、領域・境界の比較 |
| [Bacteria-ID補助実験](bacteria_id_experiment_plan.md) | Raman入力、事前学習、クラスタリング、参照論文との比較条件 |
| [Bacteria-ID分類・HPO](bacteria_id_hpo_protocol.md) | 30分離株の分類、5-fold CVによる設定選択、最終test評価 |

## 解析の構成

| 解析 | 条件・対象 | 反復と評価範囲 |
| --- | --- | --- |
| 主実験 | 主要6条件 B0・B1・A0・A1・M00・M11と、摂動ablation M10・M01 | 試料単位5-fold・3反復、K=2/4/6/8/10/12/14、未知試料のOOF評価 |
| Mask率感度解析・CV | M11の25%・50%・75% | 同じsplit・画素集合・3学習反復。50%は主実験M11を共用 |
| 全体fit | 主要6条件、全49試料 | 各表現を1回学習・fitし、K8のクラスタリングを3初期化。マップは初回を使用 |
| Mask率感度解析・全体fit | M11の50%・75% | 各表現を固定したK8・3初期化。fitに用いた試料群の記述的比較 |
| 空間マップ再現性 | M11、全49試料、K8 | 3 encoder×3 KMeans初期化の9マップを比較 |
| Ramanクラスタリング | Bacteria-4/6、M00・M11 | train 80%・test 20%、5 seed、K=4/6 |
| Raman分類 | 30分離株、M00・M11 | M11の5-fold CVで共通設定を選択し、各5 seedで最終学習・評価 |

各解析のseedと入力範囲は該当仕様で定義する。比較の位置づけや設定選択に関わる来歴は、
[主実験プロトコル](experiment_protocol.md#planned-comparisons)およびRamanの各仕様に記載する。

## 条件と解釈の境界

- データ分割・抽出：[CV・seed](experiment_protocol.md#cv-design)、[共通画素](experiment_protocol.md#pixel-sampling)。
- モデル：[比較条件](experiment_protocol.md#conditions)、[表現・PCA](experiment_protocol.md#representations)、[構成](experiment_protocol.md#model-architecture)、[学習設定](experiment_protocol.md#training-recipe)。
- 操作と評価：[摂動・クラスタリング](experiment_protocol.md#augmentation-clustering)、[数値精度](experiment_protocol.md#evaluation-precision)、[K](experiment_protocol.md#cluster-counts)、[指標・集計](evaluation_metrics.md)。

CVは、学習に含めない試料での空間的一貫性と指定摂動への安定性を評価する。
全体fitとNIR解析は、観測した試料群の化学状態を探索・解釈する。
クラスタ番号を劣化度や化学成分量として扱わず、劣化モデルは因果検証を伴わない仮説として位置づける。
Ramanの「本研究の補完条件」は、参照論文で未記載の事項を本研究で具体化した条件であり、原論文の実設定とは区別する。
