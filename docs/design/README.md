# 研究設計文書

## ステータス

このディレクトリは、本リポジトリにおける現行の研究・実装設計を管理する。
各文書で `Open` と明示した事項を除き、記載内容を **Fixed** とする。

研究目的は、古材NIRスペクトルから得る教師なし領域分割の空間的一貫性と、指定したスペクトル摂動への
安定性を比較することである。外部の正解劣化ラベルや独立した劣化測定による定量評価は行わず、
劣化との対応は最後のスペクトル・マップに基づく探索的解釈に限定する。

## 文書一覧

| 文書 | 状態 | 内容 |
| --- | --- | --- |
| [preprocessing.md](preprocessing.md) | Fixed | 200 Hz本番前処理、mask、自動cutoff、256点補間、SNV |
| [experiment_protocol.md](experiment_protocol.md) | Fixed / 実行定数Open | 主比較、5-fold・3反復、均等画素抽出、事前固定K、補助実験 |
| [evaluation_metrics.md](evaluation_metrics.md) | Fixed / 実行定数Open | 主評価LLA・LFR、silhouette・補正LLA・ARI・occupancy、pairedな報告 |
| [visualization_and_interpretation.md](visualization_and_interpretation.md) | Fixed / 表示設定Open | 全体学習後の4条件比較、Hungarian matching、探索的解釈 |

## 設計の要約

- 分割・集計単位は `KYOw...` で識別される試料とする。上位の採取関係がある場合の扱いはsplit前に確認する。
- 試料単位のランダム5-fold CVを行い、同一試料をfold間で分割しない。同じsplitで3反復する。
- 各train試料から同数$q$の画素を抽出し、学習・PCA・KMeansで全条件・全反復に共通利用する。
  test評価には各試料の全有効画素を使用する。
- 主比較は本番前処理済みSNVを直接使うraw SNV、PCA、AE、MAE、およびaugmentationの
  2×2 ablationとする。
- 提案M11とB0、B1、M00の直接比較を主要比較とし、その他の計画比較で構成要素の効果を説明する。
- mask率は主比較では50%に固定し、M11の25/50/75%比較を同じ3反復の補助実験とする。
- CVの主評価はLLA-3/5/9と3種類のlabel flip rateとし、劣化検出精度の代用にしない。
- cosine-silhouetteは各表現空間の幾何学的診断とする。補正LLA、反復間ARIおよびoccupancyを補助報告する。
- 共通K集合$\mathcal{K}$と代表表示値$K_0$をCV開始前に固定する。全試料elbowによるK校正は行わない。
- 試料別のpairedな差、試料間SDおよび3反復間SDを区別して報告する。有意差検定による採否判定は行わない。
- CV結果から「best条件」を事後選択しない。
- 全体学習後はraw SNV、PCA、標準MAE、提案Aug-MAEの4条件を解釈・可視化する。
- A0はChemoMAE v0.2.1の全領域再構成lossを使用する。
- リポジトリ内の可視化にはfigure titleおよびaxes titleを付けず、説明はcaptionまたはファイル名で管理する。

## 主実験・補助実験・解釈の区分

| 区分 | 内容 | 追加するニューラルネット学習 |
| --- | --- | --- |
| 主実験 | B0、B1、A0、M00、M10、M01、M11の5-fold・3反復。主要比較と2×2 ablationを同じrunから報告 | 75回 |
| 補助実験 | M11のmask率25%・75%。50%は主実験を再利用 | 30回 |
| 補助解析 | 共通K集合内のK依存性、反復間ARI、補正LLAおよびoccupancy診断 | 追加なし |
| 探索的解釈 | B0、B1、M00、M11を全試料でfitし、事前指定K・seedでマップとスペクトルを解釈 | 2回 |

CVは合計105学習、全体解釈まで含めると107学習となる。PCAとKMeansのfit、表現抽出、評価摂動の計算は
この回数に含めない。Kごとに表現を再学習しない。

## Open事項

$\mathcal{K}$、$K_0$、各試料の抽出画素数$q$、seed一覧、PCA次元、モデル・学習定数、augmentationの
数値設定およびLFRの摂動反復数は、CV開始前に確定する。
一覧は[experiment_protocol.md](experiment_protocol.md)第11節を参照する。
本文用の可視化例の基準と試料ID、任意の形状診断の実行定義もOpenとする。
これらを既定値や結果から暗黙に決定しない。
