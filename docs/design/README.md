# 研究設計文書

## ステータス

このディレクトリは、本リポジトリにおける現行の研究・実装設計を管理する。
各文書で `Open` と明示した事項を除き、記載内容を **Fixed** とする。

本研究は、古材NIRスペクトルのマスク再構成にnoise・shiftからのdenoisingを組み合わせ、
化学状態をより安定して反映する表現の学習につながるかを問う。これはaugmentationの導入動機であり、
指定した摂動への耐性そのものの獲得を主目的とするものではない。
採用理由と仮定の詳細は[ChemoMAEの位置づけ](../chemomae_positioning.md)にまとめる。

実験では、得られた表現による教師なし領域分割の空間的一貫性と、指定したスペクトル摂動への
label安定性を比較し、表現・マップの性質からその効果を調べる。
これらの指標だけで化学的な表現品質を直接検証したとはしない。
外部の正解劣化ラベルや独立した劣化測定による定量評価は行わず、
劣化との対応は最後のスペクトル・マップに基づく探索的解釈に限定する。

## 文書一覧

| 文書 | 状態 | 内容 |
| --- | --- | --- |
| [preprocessing.md](preprocessing.md) | Fixed | 200 Hz本番前処理、mask、自動cutoff、256点補間、SNV |
| [experiment_protocol.md](experiment_protocol.md) | 主条件Fixed / vMF数値仕様Open | 主比較、5-fold・3反復、均等画素抽出、事前固定K、mask率・vMF補助実験 |
| [evaluation_metrics.md](evaluation_metrics.md) | Fixed / 任意診断Open | 主評価LLA・LFR、silhouette・補正LLA・ARI・occupancy、pairedな報告 |
| [visualization_and_interpretation.md](visualization_and_interpretation.md) | Fixed | 全体学習後の4条件比較、Hungarian matching、本文代表例、探索的解釈 |

## 設計の要約

- 分割・集計単位は `KYOw...` で識別される試料とする。上位の採取関係がある場合の扱いはsplit前に確認する。
- 試料単位のランダム5-fold CVを行い、同一試料をfold間で分割しない。同じsplitで3反復する。
- 各train試料から$q=8192$画素を一様ランダム・非復元抽出し、学習・PCA・KMeansで全条件・全反復に共通利用する。
  test評価には各試料の全有効画素を使用する。
- 主比較は本番前処理済みSNVを直接使うraw SNV、PCA、AE、MAE、およびaugmentationの
  2×2 ablationとする。
- PCA・AE・すべてのMAE条件は16次元に統一し、次元削減後の表現をL2正規化する。
  ChemoMAEは全可視でCLSを潜在射影して抽出する。B0は256次元SNVのcosine幾何を用いる。
- PCAは`PCA(n_components=16)`の既定設定を採用し、train平均による中心化、`whiten=False`、
  `svd_solver="auto"`とする。波長ごとの追加autoscalingは行わない。
- noise角度は$U(0,2.5^\circ)$、shiftは既定の$U(-2,2)$チャネルに固定する。その他のAug操作設定は
  ChemoMAE v0.2.2の既定値を採用し、LFRでも同じ強度分布を使う。強度の追加ablationは行わない。
- 提案M11とB0、B1、M00の直接比較を主要比較とし、その他の計画比較で構成要素の効果を説明する。
- mask率は主比較では50%に固定し、M11の25/50/75%比較を同じ3反復の補助実験とする。
- 学習はMAE論文・公式PRETRAIN.mdの800 epoch recipeに従う。単一GPU、batch size 1024、勾配蓄積なし、
  AdamW、peak lr $6\times10^{-4}$、40 epoch warmup後のcosine decayとする。
- CVの主評価はLLA-3/5/9と3種類のlabel flip rateとし、劣化検出精度の代用にしない。
- LFRの評価摂動は各種類$R=5$回とする。表現抽出・評価はFP32、抽出AMPと評価TF32は無効にする。
- cosine-silhouetteは各表現空間の幾何学的診断とする。補正LLA、反復間ARIおよびoccupancyを補助報告する。
- 共通K集合は$\mathcal{K}=\{2,4,6,8,10,12,14\}$、主表・全体可視化の代表値は$K_0=8$に固定する。
  全試料elbowによるK校正は行わない。
- 試料別のpairedな差、試料間SDおよび3反復間SDを区別して報告する。有意差検定による採否判定は行わない。
- CV結果から「best条件」を事後選択しない。
- 全体学習後はraw SNV、PCA、標準MAE、提案Aug-MAEの4条件を解釈・可視化する。
- A0はChemoMAE v0.2.2の全領域再構成lossを使用する。
- リポジトリ内の可視化にはfigure titleおよびaxes titleを付けず、説明はcaptionまたはファイル名で管理する。

## 主実験・補助実験・解釈の区分

| 区分 | 内容 | 追加するニューラルネット学習 |
| --- | --- | --- |
| 主実験 | B0、B1、A0、M00、M10、M01、M11の5-fold・3反復。主要比較と2×2 ablationを同じrunから報告 | 75回 |
| 補助実験 | M11のmask率25%・75%。50%は主実験を再利用 | 30回 |
| 補助実験 | 主7条件の既存表現へvMF mixtureを適用。5-fold・3反復・共通7Kの735 fits。数値実装はOpen | 0回 |
| 補助解析 | 共通K集合内のK依存性、反復間ARI、補正LLAおよびoccupancy診断 | 追加なし |
| 探索的解釈 | B0、B1、M00、M11を全試料でfitし、事前指定K・seedでマップとスペクトルを解釈 | 2回 |

CVは合計105学習、全体解釈まで含めると107学習となる。PCA、KMeans、vMFのfit、表現抽出、評価摂動の計算は
この回数に含めない。Kごとに表現を再学習しない。

## 残るOpen事項

主実験の研究条件はFixedである。実装値は[実験プロトコル](experiment_protocol.md)第3・4・6節、
実行時に保存するseed・manifest・環境などは同第11.2節を参照する。

| Open事項 | 確定・確認する内容 | 定義先 |
| --- | --- | --- |
| vMF補助実験 | 数値精度、EM停止条件、集中度設定、修正版の検証と専用pipeline。範囲・利用版・退化成分の扱いはFixed | [実験プロトコル第5.2節](experiment_protocol.md#vmf-supplementary) |
| 任意の形状診断 | 孤立label・連結成分shapeの定義。採用する場合だけ、結果を見る前に固定する | [評価指標第7節](evaluation_metrics.md) |

vMFの比較と解釈は[評価指標第8.4節](evaluation_metrics.md#vmf-evaluation)に従う。
Open事項をライブラリの既定値で暗黙に埋めて実行しない。

## 実行文書

設計から実行を分離し、本番CLIと再開・完了判定は[../experiment_runbook.md](../experiment_runbook.md)、
テストとpreflightの要約は[../verification_history.md](../verification_history.md)、現在の進捗は
[../../ToDo.md](../../ToDo.md)で管理する。
