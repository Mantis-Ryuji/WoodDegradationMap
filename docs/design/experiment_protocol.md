# 実験プロトコル

## 1. ステータス

**Fixed**

前処理および030 Hz/200 Hzの使用方法だけは未決定であり、
[preprocessing.md](preprocessing.md)で管理する。

## 2. 研究目的

近赤外ハイパースペクトル画像から得られるSNV後スペクトルについて、
ChemoMAEが教師なしセグメンテーションに有用な表現を学習できるかを検証する。

中心となる問いは次の3点である。

1. ChemoMAEの表現はraw SNVおよびPCAより有用なクラスタ構造を与えるか。
2. mask modelingは、maskなしのautoencoderより有用か。
3. Gaussian noiseと波長方向shiftは、それぞれ単独または併用で表現の分離性、空間的一貫性、摂動安定性に寄与するか。

mask率の最適化は主題にせず、提案条件の感度を確認する補助実験として扱う。

## 3. 実験単位とcross-validation

### 3.1 試料単位

- 独立な分割単位は、ファイル名の `KYOw...` で識別される試料とする。
- 同一 `KYOw` に属する画像、測定条件、画素は、必ず同じfoldへ割り当てる。
- 画素を独立なsplit単位として使用しない。
- metadataの参照元は `data/metadata/古材メタデータ.csv` とする。

### 3.2 5-fold CV

- 試料をランダムな5-foldに分割する。
- 各foldは概ねtrain:test = 8:2となり、各試料は一度だけtestに現れる。
- 樹種による層化抽出は行わない。
- split生成時のseedを明示的に固定し、split manifestとともに記録する。
- すべての比較条件で同一のfoldを使用する。

各foldでは、表現学習、PCAのfit、Cosine-KMeansのfitをtrain試料だけで行う。
test試料は、trainで得た変換器、encoder、クラスタ中心を固定した状態で評価する。

## 4. 主比較条件

主比較ではmask率を50%に固定する。augmentationは学習時だけ適用し、
各augmentationの適用確率は、有効な条件においてそれぞれ0.5とする。

| ID | 表現 | mask率 | Gaussian noise | shift | 役割 |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | raw SNV | - | - | - | 入力空間baseline |
| B1 | PCA | - | - | - | 線形次元削減baseline |
| A0 | AE | 0% | なし | なし | maskなし再構成baseline |
| M00 | MAE | 50% | なし | なし | 標準MAE |
| M10 | Noise-MAE | 50% | あり | なし | Gaussian noiseの単独効果 |
| M01 | Shift-MAE | 50% | なし | あり | shiftの単独効果 |
| M11 | Aug-MAE | 50% | あり | あり | 提案条件 |

A0はChemoMAE v0.2.1の全領域再構成lossを使用する。
`ChemoMAE(n_mask=0)`と`TrainerConfig(loss_region="all")`を組み合わせ、
全パッチをencoderへ入力してclean targetの全スペクトルに再構成lossを計算する。
M00を含むMAE条件では`loss_region="masked"`を使用する。

### 4.1 条件間で固定するもの

- ChemoMAE系条件のencoder、decoder、latent次元および学習budget
- 学習データ、fold、前処理および画素mask
- Cosine-KMeansの実装、初期化方針および共通クラスタ数
- 評価対象の試料および画素
- augmentationの強度とseed方針

augmentation強度は結果を見て条件ごとに変更せず、CV開始前に固定する。
強度sweepは行わない。

### 4.2 計画比較

| 比較 | 検証する内容 |
| --- | --- |
| M00 vs B0 | 表現学習を導入する効果 |
| M00 vs B1 | 非線形表現学習を導入する効果 |
| M00 vs A0 | mask modelingの効果 |
| M10 vs M00 | Gaussian noiseの単独効果 |
| M01 vs M00 | shiftの単独効果 |
| M11 vs M00 | 2種類のaugmentationを併用する効果 |
| M11 vs M10 | shiftの追加効果 |
| M11 vs M01 | Gaussian noiseの追加効果 |

M00、M10、M01、M11は、Gaussian noiseの有無とshiftの有無による2×2要因計画として扱う。
必要に応じ、各評価指標について次の交互作用contrastを併記する。

$$
(M11-M10)-(M01-M00)
$$

## 5. mask率の補助実験

提案条件M11について、mask率25%、50%、75%を比較する。

| ID | mask率 | Gaussian noise | shift |
| --- | ---: | ---: | ---: |
| M11-25 | 25% | あり | あり |
| M11-50 | 50% | あり | あり |
| M11-75 | 75% | あり | あり |

M11-50は主比較のM11と同一条件であり、新たな条件として重複学習させない。
この実験は「最適mask率」の選択ではなく、提案条件のmask率依存性を確認する感度解析として扱う。
結果は補助実験として報告し、主条件M11を事後的に置き換えない。
fold、K range、学習budgetおよび評価指標は主比較と同一にする。

Gaussian noiseのみ、shiftのみの条件ではmask率sweepを行わない。

## 6. クラスタ数

### 6.1 基準クラスタ数

全試料のraw SNVを対象としてCosine-KMeansのelbowを一度だけ計算し、基準クラスタ数を

$$
K_0
$$

とする。この決定に樹種または劣化ラベルを使用しない。

### 6.2 共通range

全fold、全条件で次の共通rangeを使用する。

$$
\mathcal{K}=
\{K_0-2,K_0-1,K_0,K_0+1,K_0+2\}
$$

ただし、2未満の値は除外する。条件ごと、foldごと、評価指標ごとに都合のよいKを選ばない。

- $K_0$を中心値として表形式で報告する。
- $\mathcal{K}$全体は、Kに対して結論が反転しないか確認する曲線として報告する。
- 5点を単一の総合scoreへ自動的に集約しない。

全試料を用いたK決定は、ラベルを使わないtransductiveな共通K校正として明記する。

## 7. fold内の処理

各fold、各Kについて次を行う。

1. train試料だけを用いて、必要な表現変換器またはChemoMAEを学習する。
2. train/testの木材領域画素から、条件に対応する表現を抽出する。
3. train表現だけでCosine-KMeansをfitする。
4. 固定したtrain centroidsを用いてtest表現へラベルを割り当てる。
5. test試料に対して主評価指標を計算する。
6. 試料単位で集計し、foldおよび全out-of-fold試料の結果を保存する。

B1のPCAはfoldごとにtrain試料だけでfitし、同じ変換をtest試料へ適用する。
PCA次元はCV開始前に固定し、結果を見て変更しない。

## 8. 条件選択の方針

- CV結果から単一の「best条件」を選定しない。
- cosine-silhouette、LLA、label flip rateを恣意的に合成した総合scoreを作らない。
- 計画比較ごとに、各指標の方向とKに対する傾向を報告する。
- 解釈・可視化結果を用いて主条件を事後選択しない。

全体学習および解釈の対象はCV順位にかかわらず、事前に定めた4条件
B0、B1、M00、M11とする。詳細は
[visualization_and_interpretation.md](visualization_and_interpretation.md)に定義する。

## 9. 実行しない探索

- Encoder/Decoderの幅またはdepth sweep
- Gaussian noise強度sweep
- shift強度sweep
- Gaussian noise単独条件のmask率sweep
- shift単独条件のmask率sweep
- 条件別のK選択
- CV結果によるbest条件探索

seedは再現性のため固定・記録するが、seed自体を実験因子とするsweepは行わない。

## 10. 実行順序

1. 前処理仕様と030 Hz/200 Hzの使用方法を確定する。
2. augmentation強度、PCA次元、split seedなどの実行定数を固定する。
3. 全試料raw SNVから$K_0$と共通$\mathcal{K}$を決める。
4. 試料単位の共通5-fold splitを作成する。
5. 主比較7条件を同一プロトコルで評価する。
6. M11のmask率補助実験を行う。
7. B0、B1、M00、M11を全試料でfitまたは学習し、$K_0$でクラスタリングする。
8. ラベルを整列したうえで、スペクトル、ラベルマップ、潜在空間を解釈する。
