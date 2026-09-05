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
| [experiment_protocol.md](experiment_protocol.md) | Fixed | 主比較、5-fold・3反復、均等画素抽出、事前固定K、補助実験 |
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
  ChemoMAE v0.2.1の既定値を採用し、LFRでも同じ強度分布を使う。強度の追加ablationは行わない。
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

## 残るOpen事項

各試料の抽出画素数$q=8192$と一様ランダム・非復元抽出をFixedとし、これまでOpenとしていた
主実験の条件選択は解消した。ユーザーが共有した確認結果では、現行49試料すべてで抽出可能だった。
確認結果と確定状況は[experiment_protocol.md](experiment_protocol.md)第3.4節・第11.1節を参照する。
用途別seed、manifest、試料の対応・採取関係、GPU・library情報などは同文書第11.2節の
実行記録事項として整理した。
optimizer・学習率・800 epochのschedule・batch size・学習精度は同文書第4.2節でFixedとした。
共通K集合と代表表示値は同文書第6節でFixedとした。
16次元出力・L2正規化・全可視抽出は同文書第4.1.1節でFixedとした。
PCAの既定設定とAug強度・操作設定は同文書第4.1.1節・第4.1.3節でFixedとした。
幅256・8層・8head・16patchのencoder、線形1層decoder、dropout=0.0、ChemoMAE既定初期化は第4.1.2節でFixedとした。
Cosine-KMeansは既定の初期化1回・`max_iter=500`・`tol=1e-4`を採用し、Kと反復用seedだけを実験計画に合わせる。
抽出・評価FP32とlibrary既定epsilonは第4.1.4節、LFRの$R=5$は評価文書第5.3節でFixedとした。
同文書第4.1.3節・第4.2節にv0.2.1の実装確認と、MAE学習recipeに合わせる際の差を記録した。
本文用の可視化例は、各樹種の保存有効画素数が最大の7試料として事前固定済みである。
残るOpen事項は、評価文書第7.1節に示した任意の孤立label・連結成分shape診断の実行定義だけである。
採用する場合は結果を見る前に定義し、採用しない場合は主評価へ追加しない。

## 実行文書

設計から実行を分離し、本番CLIと再開・完了判定は[../experiment_runbook.md](../experiment_runbook.md)、
テストとpreflightの要約は[../verification_history.md](../verification_history.md)、現在の進捗は
[../../ToDo.md](../../ToDo.md)で管理する。
