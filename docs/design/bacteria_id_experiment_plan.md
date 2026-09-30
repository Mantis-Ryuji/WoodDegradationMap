# Bacteria-IDを用いたRaman補助実験計画

> Table 5の分類学習条件は[5-fold CVプロトコル](bacteria_id_hpo_protocol.md)を正とする。
> 本書は事前学習・Table 4・引用値を定義する。

参照版は**SMAE出版版PDF**とする。
**Fixed**は採用条件を示す。公開資料に記載がない事項は、ユーザーの補完方針に従い
**本研究の補完条件**として具体化する。論文・公式コードから確認した設定と区別して記録する。
SMAE補足PDF、RamanCluster本文PDF・補足DOCXも照合した。実験の進捗は[ToDo](../../ToDo.md)で管理する。
結果の研究上の位置づけとThesisへの引き継ぎ方針は
[Raman補助実験の解釈計画](../bacteria_id_interpretation_plan.md)に記す。

## 1. 目的・実施範囲・優先順位

古材NIR以外の公開スペクトルでもChemoMAEによる自己教師あり表現学習を評価するため、
Bacteria-IDのRamanスペクトルを補助実験に用いる。新たに実験する手法は
**ChemoMAE(M00)とChemoMAE(M11)の2条件のみ**とする。

主な比較表は、SMAEの次の2表へ両条件を追加したものとする。

- **Table 4相当**：Bacteria-4・Bacteria-6のクラスタリングACC・NMI・AMI。
- **Table 5相当**：Bacteria-IDの30クラス分類Accuracy。両条件は`w/ pretraining`列へ追加する。

**Table 4の主結果はvalidationなし・train 80%／test 20%の新設計とする。** Table 5では、M11の5-fold CVで選んだ
分類設定をM00/M11に共通に適用する。設定選択と全件学習は第7節に記す。

他手法はSMAE掲載の報告値を引用し、SMAE・ResNet・RamanCluster等を再実験しない。
ChemoMAEのscratch学習、追加baseline、潜在次元・Aug強度のsweep、追加データセット、
Table 3や追加可視化は実施範囲に含めない。
M00もmasked reconstructionによる事前学習を行う条件であり、
`w/o pretraining`ではない。

Bacteria-IDを古材のmask ratio sweepより先に実施した。
中断したmask sweepの再開前には、既存run・checkpointの状態を確認する。
古材の主実験・mask sweepの固定条件は[既存プロトコル](experiment_protocol.md)に従う。

### 参照版と比較条件の扱い（Fixed）

参照論文はRen, Zhou, and Li, *A self-supervised learning method for Raman spectroscopy based on masked autoencoders*,
**Expert Systems With Applications 292 (2025), 128576**、
[DOI: 10.1016/j.eswa.2025.128576](https://doi.org/10.1016/j.eswa.2025.128576)。
ユーザー提示の出版版PDF（全12ページ）のp.8でTable 4・5、§3.3で反復条件を照合した。
報告値・反復条件をarXiv版から採用しない。

補助的な参照資料は、SMAE補足のS-5・Fig. S3と、Sun, Wang, and Jiang,
*RamanCluster: A deep clustering-based framework for unsupervised Raman spectral identification of pathogenic bacteria*,
**Talanta 275 (2024), 126076**、[DOI: 10.1016/j.talanta.2024.126076](https://doi.org/10.1016/j.talanta.2024.126076)
の§3.1・§3.3・補足Fig. S2とする。

公開資料から確認できるデータ定義・分割比率・指標・反復と報告形式・分類head・下流学習手順は
SMAEを参照する。未記載の設定や資料間の不一致については、**第8節の補完条件を採用して実験を進める**。
著者回答や非公開の実行情報の取得を着手条件にしない。補完した設定をSMAEの実設定として引用しない。
SMAEの出版版本文・補足資料、公式コードを参照し、Bacteria-6の資料間不一致はSMAE補足図の構成を優先する。

公式コードの確認版は
[`pengjuRen99/SpectraMAENet@7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3`](https://github.com/pengjuRen99/SpectraMAENet/tree/7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3)
（2025-06-23のcommit）とする。このcommitが論文掲載値を生成した実行版であることまでは確認できていない。

方法として意図的に異なるのは、既に指定済みのChemoMAE構成・SNV・WoodDegradationMapのpretraining recipe・
TGN/FSと、Table 4のCosine-KMeansである。既知の評価条件を揃え、これらの手法上の差と第8節の補完条件を
脚注へ明示した文献比較とする。SMAE等を同一環境・同一計算予算で再実験した比較とは記述しない。

## 2. モデル・前処理（Fixed）

設定の基準は**WoodDegradationMapの採用済みモデルとpretraining recipe**とする。
ChemoMAEライブラリの引数既定値やSMAEのpretraining設定で置き換えない。
参照版はChemoMAE v0.2.2、commit
[`4ec7f6acecb82035c85001f5aee508910d40adac`](https://github.com/Mantis-Ryuji/ChemoMAE/commit/4ec7f6acecb82035c85001f5aee508910d40adac)。

| 項目 | Bacteria-IDでの設定 |
| --- | --- |
| 入力 | 1スペクトル1000チャネル、`seq_len=1000` |
| 前処理 | スペクトルごとのSNV。標本標準偏差（`ddof=1`）を使用 |
| patch | `n_patches=20`、1 patchは連続50チャネル |
| encoder幅 | `d_model=256` |
| attention / 層数 | `nhead=8` / `num_layers=8` |
| FFN / dropout | `dim_feedforward=1024` / `dropout=0.0` |
| 潜在 | CLSから`to_latent`で128次元、`latent_dim=128`、`latent_normalize=True` |
| decoder | `decoder_num_layers=1`、`Linear(128, 1000)` |
| pretraining mask | 50%、20 patch中10 patchを隠す |
| pretraining loss | maskedチャネルだけのMSE、patch内target正規化なし |
| 初期化 | WoodDegradationMapと同じChemoMAE v0.2.2の初期化 |
| 通常の特徴抽出・分類 | 全20 patchを可視にする。ランダムmaskなし |

入力長・patch構成・潜在次元以外のモデル設定は
[古材の構成](experiment_protocol.md#model-architecture)を引き継ぐ。
潜在128次元は本補助実験の固定条件であり、SMAE等とのbottleneckの同一性や最適性を意味しない。
encoder幅256と、単一潜在の次元128を区別する。
128次元単位潜在はpretrainingの再構成とTable 4のクラスタリングに使用する。
Table 5の分類headは投影前CLSへ接続する（第7節）。

配布済みスペクトルを入力原本として保持し、SNVを適用する。古材固有の256点補間・空間mask・
画素抽出を流用しない。波数軸・チャネル順・shapeを確認し、追加の平滑化や0〜1 clippingは導入しない。
SNV後の負値は有効値である。非有限値・標準偏差ゼロ等が見つかった場合、黙って除外して
SMAEと異なる評価集合にせず、対象と件数を記録して取り扱いを決める。

## 3. TGN・Fractional Shift（Fixed）

**強度もWoodDegradationMapと同じ値**を使用する。Raman用に弱めたり、学習段階ごとに変えたりしない。
TGNはTangent Gaussian Noise、FSはFractional Shiftを指す。

| 項目 | 採用設定 |
| --- | --- |
| TGN角度 | 一様分布0〜5°、`noise_angle_deg_range=(0.0, 5.0)` |
| FS移動量 | 一様分布−2〜2チャネル、`shift_delta_range=(-2.0, 2.0)` |
| 適用確率 | 有効な各操作についてスペクトルごとに独立に0.5 |
| 操作順 | `shuffle_order_per_batch=True` |
| 再中心化・norm | `recenter_after_each_op=True`、`renorm_to_input_norm=True` |
| FS補間・端点 | 線形補間・端点値延長。循環shiftなし |
| 数値安定化 | `eps=1e-8` |

処理順は**配布入力 → SNV → TGN/FS**とする。各操作後に平均ゼロ・一定normへ戻す。
原入力へSMAEのAugをかけてからSNVする処理は使用しない。
FSの単位は1000点入力のチャネルindexであり、古材と同じ物理的な波長・波数移動量を意味しない。
定義と実装上の扱いは[augmentation仕様](experiment_protocol.md#augmentation-clustering)を参照する。

| 段階 | ChemoMAE(M00) | ChemoMAE(M11) |
| --- | --- | --- |
| Pretraining | 追加Augなし、mask 50% | TGN・FS各0.5、mask 50% |
| Finetuningのtrain | TGN・FS各0.5、全patch可視 | TGN・FS各0.5、全patch可視 |
| Table 4 test・クラスタリング用特徴抽出、Table 5 validation・test | Augなし、全patch可視 | Augなし、全patch可視 |

M00/M11は事前学習条件を表す。finetuningのAugを共通にし、両条件の差を
事前学習時の追加Augの有無として比較する。
pretrainingのtargetは両条件とも**追加摂動前の観測SNV**とし、targetの強度・チャネル位置は動かさない。
ここでいうcleanは測定ノイズのない真値を意味しない。

## 4. Pretraining recipe（Fixed）

[WoodDegradationMapの学習recipe](experiment_protocol.md#training-recipe)と
[experiments/config.py](../../src/wood_degradation_map/experiments/config.py)の採用値を引き継ぐ。

| 項目 | 採用設定 |
| --- | --- |
| epoch / batch size | 800 / 1024 |
| GPU / gradient accumulation | 単一GPU / `accum_iter=1` |
| optimizer | AdamW、betas=(0.9, 0.95)、eps=1e-8、AMSGradなし |
| weight decay | 0.05。biasと正規化層のパラメータは0 |
| base / peak learning rate | 1.5e-4 / 6e-4（effective batch 1024） |
| schedule | 40 epochの線形warmup、その後760 epochのcosine decay、最小学習率0 |
| learning rate更新 | 各batchの処理前、epoch内進捗を含めて更新 |
| 精度 | FP32パラメータ、FP16 autocast、GradScaler |
| gradient clipping / EMA | ともに使用しない |
| train DataLoader | shuffleあり、`drop_last=True` |
| 終了・checkpoint | early stoppingなし、800 epoch完了時のraw weights（`last_model.pt`） |

M00/M11で入力集合・batch順序・mask等の対応を取れるよう、用途別seedを記録する。
**Table 5は出版版に合わせて各条件5つの異なるseedで評価する。Table 4も補完条件として5 seedとする。**
run seedは`[0, 1, 2, 3, 4]`。各seedで事前学習から独立に反復し、Table 5では対応する同じseedの
pretraining checkpointを1回のfinetuningへ渡す。単一checkpointに対するfinetuningだけの5反復とはしない。
Table 4のBacteria-4/6とTable 5は学習コーパスを分ける。用途別seed・実行数は第8節に定義する。
古材の5 folds×3反復を自動的に持ち込まない。
実際の学習件数、1 epochのbatch数、予定・実更新数、AMPによるskip数を保存する。
800 epochをSMAEと同じ計算予算と表現しない。

## 5. データと分割

データの出典は[原著論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6960993/)と
[Bacteria-ID公式リポジトリ](https://github.com/csho33/bacteria-ID)とする。
30クラスは細菌・酵母の**isolateクラス**であり、30の異なる菌種とは記述しない。

### Table 5の配布subsetと分割（Fixed）

| 配布subset | 件数 | Table 5での役割 |
| --- | ---: | --- |
| reference | 60,000（30クラス×2,000） | ラベルを損失へ使わないpretraining |
| finetune | 3,000（30クラス×100） | 5-fold CVの学習・validation、採用後の全件学習 |
| test | 3,000（30クラス×100） | 最終Accuracyの評価 |

Table 5ではfinetune subsetへ`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`を
適用し、5分割すべてでM11 seed 0のHPOを行う。各foldはtrain 2,400・validation 600で、
設定選択後はfinetune 3,000件すべてでM00/M11を学習する。出版版の「5つの異なるseed」を
5 foldsでの評価と読み替えず、本研究の5-fold HPOとして区別する。
入力ファイル内の行順と分割indexを保存する。SMAE出版版の実際の分割は公開資料から未確認である。

testはpretraining、HPO、設定選択、最終学習には使用しない。
SNVは各スペクトル内の統計量だけを用いる。配布subset間の同一個体・同一測定群の関係は
出典で確認できた範囲で記録し、古材の試料単位CVや未知患者評価と同一視しない。

### Table 4の件数・分割比率とクラス構成

SMAE本文§3.2と補足S-5は、Bacteria-4/6/8/10をRamanClusterの分割方法に従うと記載する。
RamanCluster本文§3.1（pp.5–6）は、10種それぞれ2,000件、4・6・8・10クラスのデータセット、
**train / validation / test = 60% / 20% / 20%**と定義している。
本研究では保存済みの旧60/20/20分割から、旧trainと旧validationだけを統合する。
旧testの元行indexと配列順を一切変更せず、最終分割を**train 80%／test 20%**とする。
下表は本研究で使用する最終的な件数であり、文献手法の学習割合が一致するとの主張ではない。

| 対象 | 分割前総数 | 新train | test |
| --- | ---: | ---: | ---: |
| Bacteria-4 | 8,000 | 6,400 | 1,600 |
| Bacteria-6 | 12,000 | 9,600 | 2,400 |

この80/20はTable 4用であり、Table 5の配布済みreference/finetune/testの役割と混同しない。
2,000件/クラスと整合する**配布reference subset**から対象isolateを抽出する。
使用ファイルは[公式配布案内](https://github.com/csho33/bacteria-ID/blob/3c00a712a6dbad9aefa19ac878a1e6db20590ca9/data/data.md)の
`X_reference.npy`・`y_reference.npy`とする。元ファイルの指定は本研究の補完条件である。

最終的なTable 4の分割は、保存済みindexを入力として次の手順で固定する。

1. `data/processed/bacteria_id_v1/splits.npz`の旧train・旧validation・旧testを読み、旧分割の非重複・対象全件の網羅・クラス別件数を検査する。
2. 新trainは旧trainと旧validationの和集合とし、元行index順に保存する。
3. testは旧testの元行index配列を順序も含めてコピーする。新たな乱数分割は行わない。
4. 新train/testの非重複、対象全件の網羅、旧testとの完全一致、各クラスtrain 1,600件・test 400件を確認する。

Bacteria-4/6でそれぞれ分割し、各分割をM00/M11・5 seedの全runで共有する。
事前学習とCKmeansのfitは**新train全件**を用い、testでは固定したクラスタ中心への割当と採点を行う。
Table 4にvalidation・HPO・early stoppingを設けず、初期化から800 epoch学習した最終raw weightsを使用する。
Table 5のreference全体で学習したcheckpointをTable 4へ流用しない。

**クラス構成には、SMAEの「RamanClusterに従う」という本文記述と補足図の間に不一致がある。**
両方の補足図を確認した結果を次に示す。

| 対象・資料 | 図のクラス名・読み取れる構成 |
| --- | --- |
| RamanCluster補足Fig. S2(a)：Bacteria-4 | C. albicans、C. glabrata、K. aerogenes、E. coli |
| SMAE補足Fig. S3(a)：Bacteria-4 | C. albicans、C. glabrata、K. aerogenes、E. coli 1 |
| RamanCluster補足Fig. S2(b)：Bacteria-6 | 上記4種にE. faecium・E. faecalisを追加 |
| SMAE補足Fig. S3(b)：Bacteria-6 | 上記4 isolateクラスにE. coli 2・E. faeciumを追加。E. faecalisは含まれない表示 |

RamanCluster本文§3.1が列挙する先頭6種もFig. S2(b)と整合する。
一方、SMAE Fig. S3は全panel共通の凡例と色の対応から上記の構成と読め、6クラス中2クラスがE. coliである。
[Bacteria-ID公式config.py](https://github.com/csho33/bacteria-ID/blob/3c00a712a6dbad9aefa19ac878a1e6db20590ca9/config.py)
の`STRAINS`と照合すると、SMAE図の表示はBacteria-4がID `[0, 1, 2, 3]`、
Bacteria-6がID `[0, 1, 2, 3, 4, 5]`に対応する。
本研究ではSMAEへの追加比較を目的とするため、**この図の表示に基づくIDを採用する（補完条件、Fixed）**。
SMAE Table 4の数値を算出した実行IDを確認できた、という意味ではない。
RamanClusterのE. coliはID 3/4、E. faecalisはID 6/7のいずれかを、資料だけでは特定できない。
SMAE Fig. S3のcaptionも自己注意機構の模式図と記載され、実際のクラスタ図と一致していない。

図の誤記か実データ構成の差かは未解明のまま記録し、Bacteria-6を両論文で同一集合と断定しない。
本実験のクラス構成は上記IDで固定し、RamanCluster側の構成による追加実験は行わない。
表の脚注にSMAE補足図に基づく構成であることと、比較元の資料間不一致を簡潔に明記する。

## 6. Table 4：教師なしクラスタリング

### 評価手順（Fixed）

1. 第5節の各subsetの新trainでM00・M11をseed 0–4の各runで初期化から800 epoch事前学習する。ラベルは保存済み分割の検証と採点にのみ使用し、
   pretraining lossやクラスタ中心更新に使わない。
2. 事前学習checkpointから、Augなし・全patch可視で128次元の単位潜在を抽出する。
   分類finetuning後のencoderをこの評価へ流用しない。
3. **CKmeansとしてChemoMAEのCosine-KMeans**を新train全件の潜在でfitする。Bacteria-4はK=4、Bacteria-6はK=6。
   潜在を直接入力し、t-SNE・UMAP等の追加次元削減は行わない。test潜在はcosine類似度最大の学習済み中心へ割り当て、
   test上で中心を更新・再fitしない。このfit/採点集合の扱いは本研究の補完条件である。
4. test全件のACC・NMI・AMIを百分率で算出する。ACCはクラスタ番号と正解ラベルの最適な一対一対応に基づく。
   ラベル対応は採点に使用し、クラスタ所属を学習し直さない。**AMIをARIへ置き換えない。**
5. 各subset・各条件の5 seedの算術平均と標本SD（`ddof=1`）をChemoMAE行へ掲載する。
   seed別の3指標も記録し、最良seedを選ばない。SMAE掲載の引用行は原表どおり点推定値のみとし、
   未報告のばらつきを推定して付け足さない。Table 4の「±」は標本SDであり、Table 5の95% CIとは異なる。

Cosine-KMeansの初期化はk-means++型、1 fitあたり1初期化、`max_iter=500`、`tol=1e-4`とし、
[既存仕様](experiment_protocol.md#augmentation-clustering)に従う。追加restartによる最良値選択は行わない。
ACCの対応づけは[公開notebookのcluster_acc](https://github.com/pengjuRen99/SpectraMAENet/blob/7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3/Unsupervised_Mapping/t_SNE.ipynb)
と同じHungarian matchingに基づく。testの分割全体で対応づけを求め、batchごとのACCを平均しない。
特徴抽出・採点では`drop_last=False`とし、trainのfit対象も含め端数batchを欠落させない。

### NMI・AMIの定義（Fixed）

SMAEの比較元であるRamanCluster本文§3.3、式(15)・(16)に明記された定義を採用する。
正解ラベルを$Y$、クラスタ割当を$\hat{Y}$、相互情報量を$I$、エントロピーを$H$とすると、

$$
\mathrm{NMI}=\frac{I(Y;\hat{Y})}{[H(Y)+H(\hat{Y})]/2}
$$

$$
\mathrm{AMI}=\frac{I(Y;\hat{Y})-\mathbb{E}[I(Y;\hat{Y})]}
{\max\{H(Y),H(\hat{Y})\}-\mathbb{E}[I(Y;\hat{Y})]}
$$

実装では[NMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)に
`average_method="arithmetic"`、[AMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)に
`average_method="max"`を明示し、百分率へ変換する。AMIをライブラリの既定値へ任せない。
AMIの負値を0にclipしない。出力前の値と使用ライブラリのversionを保存する。
この定義の根拠はRamanClusterの数式であり、SMAEの未公開の採点コードまで同一と確認した意味ではない。

### 主表と比較上の留保

既存6手法の数値は[SMAE出版版](https://doi.org/10.1016/j.eswa.2025.128576) **p.8、Table 4**の報告値。
全セルを提示PDFと照合済み。単位は%、出版版と同じ小数2桁で表示する。
既存5手法（SMAE自身を除く）の30数値のうち29数値は
[RamanCluster Table 1](https://doi.org/10.1016/j.talanta.2024.126076)と一致する。
この一致は既報値を踏襲した可能性を示すが、SMAEが既存5手法を再実行したかどうかは本文・補足資料から確認できない。
したがって、これらを「SMAEが同一条件で再計算したbaseline」とは記載しない。
SimCLRのBacteria-4 NMIはSMAE Table 4で54.40、RamanCluster Table 1で54.5と相違するが、
参照版の指定に従いSMAEの54.40を保持する。元論文の値へ黙って修正しない。

実測値と引用値を合わせた表は[Table 4主表](../../outputs/experiments/bacteria_id_table4_80_20_v1/table4.md)を正本とする。

脚注で、既存手法はSMAE Table 4からの引用値、ChemoMAEは5 seedの平均±標本SDであり、
ChemoMAEにはSNV・128次元単位潜在・
Cosine-KMeansを使用することを明記する。SMAE補足図に基づくisolate ID、最終的な80/20分割、
trainでのCKmeans fitとtest採点を補完条件として脚注に記載する。
クラス構成の資料間不一致と未公開のsplit indexがあるため、元実験と完全に同一のデータ集合とは記載しない。
全手法を同じ前処理・クラスタリング・計算予算で再実験した比較とは記述しない。
引用行にばらつきがなく、分割・クラス構成にも未確認事項があるため、点推定値の大小だけで
統計的な優越性や同一条件下のSOTAを主張しない。既報値にばらつきがない理由や著者の意図も推測しない。

## 7. Table 5：事前学習後の分類

分類学習の確定条件は[5-fold CVと全件学習のプロトコル](bacteria_id_hpo_protocol.md)を正とする。
既存のreference30事前学習重みを使い、M11 seed 0で5-fold CVのepoch 50 Accuracy平均により
設定を選んだ。当初100件を予定したが、64件の異なる設定が完了した時点で、
ユーザー指定により最初の60件から選んだ。採用設定をM00/M11の各5 seedへ
共通に適用し、finetune subset 3,000件すべてで50 epoch学習する。
別のtest 3,000件は、採用したepoch 50の重みだけで評価する。

既存手法の数値は[SMAE出版版](https://doi.org/10.1016/j.eswa.2025.128576)
p.8、Table 5の報告値であり、本研究では再実行しない。

実測値と引用値を合わせた表は[Table 5主表](../../outputs/experiments/bacteria_id_table5_hpo_cv_v1/table5.md)を正本とする。

ChemoMAEは事前学習から独立した5 seedのtest Accuracyを百分率で集計し、
標本標準偏差を$s$として、自由度4のt分布による半幅を用いる。

$$
\bar{x}\ \pm\ t_{0.975,4}\frac{s}{\sqrt{5}}
$$

TCLPの82.30 ± 0.5%はSMAE著者がencoderを変更して報告した値であり、
TCLP原論文の値ではない。出版版Table 5のencoder更新範囲、checkpoint選択、
反復の開始点、CI計算法は特定できない。Fig. 10の評価subsetも明記されていない。
比較上の留保は[解釈計画](../bacteria_id_interpretation_plan.md)にまとめる。

<a id="open-items"></a>

## 8. 未記載条件の補完と実行数（Fixed）

ユーザーの「書いていない部分はこちらで補完する」という方針に基づき、以下を本研究の条件として採用する。
公開資料に情報がないことや、SMAEとRamanClusterの図が一致しないことは文献比較の留保として残す。
追加調査・著者照会を実験開始の前提にしない。

| 対象 | 本研究の補完条件 |
| --- | --- |
| Table 4のisolate | SMAE補足Fig. S3を優先。Bacteria-4はID `[0, 1, 2, 3]`、Bacteria-6はID `[0, 1, 2, 3, 4, 5]` |
| Table 4の入力・分割 | referenceから抽出した保存済み旧train∪旧validationを新trainとし、旧testのindex・順序を保持する80/20。M00/M11・全runで固定 |
| Table 4のfit・採点 | 20 runを初期化から800 epoch学習し、新train全件の潜在でCKmeans fit、旧test全件で採点。validation・HPO・early stoppingなし |
| 反復・run seed | 両表とも`[0, 1, 2, 3, 4]`の5 seed。Table 4のChemoMAE行は5 seedの平均±標本SDを掲載 |
| Pretrainingの対応 | 各seed・各コーパス・各条件で独立に800 epoch。Table 5のCVではM11 seed 0、全件学習ではM00/M11のseed 0–4を使用 |
| Table 5の評価checkpoint | 50 epoch固定。CVはepoch 50のvalidation、全件学習はepoch 50の重みを使用 |
| Table 5の95% CI | 5 seedのAccuracyの標本SDから求める自由度4のt信頼区間 |

Table 5の5反復・95%信頼区間という報告形式は出版版由来であり、seed値・反復の開始点・CIの式が補完部分である。
Table 5の5-fold CVは第5節のfinetune subsetに限定し、Table 4用の分割へ置き換えない。

### 乱数の管理

run seedを`s=0..4`、コーパス識別子を`bacteria4`・`bacteria6`・`reference30`とする。
用途別seedは既存の[seed生成方針](experiment_protocol.md#seed-plan)と同様にSHA-256から生成する。
入力は`["bacteria-id-seeds-v1", corpus, s, purpose]`のASCII JSON（空白なし）とし、digest先頭4 byteをbig-endian整数にする。
purposeは`model_init`・`train_order`・`mask`・`pretrain_augmentation`・`kmeans`・
`finetune_head`・`finetune_order`・`finetune_augmentation`で区別する。
M00/M11の条件名はseed生成に含めず、同じコーパス・run seed・用途では同じseedを共有する。
未使用のAugが他用途の乱数列を変えないよう、batch順序・mask・Augの乱数状態を分離する。
Table 5では`reference30`のpretrainingとfinetuningを同じrun seedで対応づける。
split用42はこの用途別生成の対象外とし、実際のseed一覧と分割indexを学習開始前に保存する。
seedの衝突を検出した場合は開始前に扱いを修正・記録し、結果を見たseedの選び直しは行わない。

### 学習回数

| 用途 | 学習コーパス | 条件×反復 | Pretraining | Finetuning |
| --- | --- | ---: | ---: | ---: |
| Table 4：Bacteria-4 | reference内の対象4クラスの新train 6,400件 | 2×5 | 10回（新規） | 0回 |
| Table 4：Bacteria-6 | reference内の対象6クラスの新train 9,600件 | 2×5 | 10回（新規） | 0回 |
| Table 5 HPO | reference全60,000件 → finetuneの5 folds（各train 2,400件） | M11 seed 0×64完了設定×5 folds、選択対象は先頭60設定 | 1件のcheckpointを再利用 | 320回完了 |
| Table 5全件学習 | reference全60,000件 → finetune全3,000件 | M00/M11×5 seed | 10件のcheckpointを再利用 | 10回 |
| 合計 | 3つの事前学習コーパス | — | **30回** | **330回完了** |

新Table 4の事前学習20回は初期化から完了した。Table 5用のreference30事前学習10回は
旧実験から独立した既存checkpointを維持した。
Table 4のCKmeans fitは計20回である。Table 5には未完了の`trial_280`の4 fold記録も
保存されており、上表の完了設定64件×5 foldsには含めない。
比較手法はM00/M11のみとする。
Table 4用の事前学習は各subsetのtrainに限定し、Table 5のreference全体を見たcheckpointとの共有を避ける。
新Table 4事前学習1 runの予定optimizer stepはBacteria-4が4,800（6 batch/epoch）、
Bacteria-6が7,200（9 batch/epoch）である（800 epoch、batch 1024、`drop_last=True`）。
実更新数・AMP skip数をrun別に記録する。Table 5用reference30の設定・更新数は変更しない。
不調なseed・中断runを除外して集計せず、予定した各5 runを揃えて表を作る。

## 9. 成果物と由来

前処理済み入力は`data/processed/bacteria_id_v1/`、Table 4の主結果は
`outputs/experiments/bacteria_id_table4_80_20_v1/`、Table 5用事前学習は
`outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/`、Table 5の探索・最終結果は
`outputs/experiments/bacteria_id_table5_hpo_cv_v1/`に保存した。
実行コマンドと確認点は[専用runbook](../bacteria_id_hpo_runbook.md)、
移動前後の対応は[成果物整理記録](../bacteria_id_artifact_reorganization_plan.md)を参照する。

Table 4では旧60/20/20の結果を取得した後、旧trainと旧validationを統合した80/20を
主結果に採用した。旧testの行indexと順序は維持し、旧結果は
`outputs/experiments/bacteria_id_table4_60_20_20_legacy_v1/`に別保存した。
Table 5は完了64設定のうち先頭60設定で選択した。両表の経緯を区別して報告する。
公式BoxリンクはHTTP 404であり、使用した公開ミラーと公式原本のバイト単位の同一性は未確認である。
