# ChemoMAEの特徴とケモメトリクスにおける位置づけ

本書はChemoMAE v0.2.2の採用理由と、構成から言えること・評価で確かめることを整理する。
固定条件は[実験プロトコル](design/experiment_protocol.md)、導出は[修論付録B](../thesis/appendices/mathematical_details.md)、
文献の確認範囲は[関連研究](related_work.md)、観察は[解釈メモ](interpretation_notes.md)を参照する。

- [MAEと追加corruptionの採用理由](#1-この構成をどう捉えるか)
- [モデルと損失](#2-実装で確認できる構成)
- [PCA・L2正規化・SAM](#pca-comparison)
- [TGN・shift](#spectral-augmentation)・[VRM的な解釈](#augmentation-vicinity)
- [主張と検証範囲](#5-本研究で主張できる特徴と評価を待つ事項)

## 1. この構成をどう捉えるか

ChemoMAEは、状態の区分や正解ラベルをあらかじめ定めにくい材料について、帯域間の予測関係から
スペクトルを圧縮し、状態差を探索する座標系を学ぶ役割を担う。古材NIR-HSIを実証対象とし、
固定encoderによるクラスタマップをCVで比較し、NIRと位置対応FT-IRから領域差を解釈する計画である。
研究全体の問いと証拠の対応は[研究概要](research_overview.md#research-focus)に示す。

**PCAと同じく、ラベルなしで得た低次元座標を可視化・クラスタリングに使う**という用途である。
この用途には非線形PCAの前例があり、Raman解析にもMAEによるクラスタリングや、非線形encoderと
線形decoderの組合せがある。
([Kramer, 1991](https://doi.org/10.1002/aic.690370209);
[Ren et al., 2025, §3.2](https://arxiv.org/html/2504.16130v1);
[Georgiev et al., 2024, Methods](https://arxiv.org/html/2403.04526v1))

本構成は非線形encoder・アフィンdecoder・単位潜在を持ち、PCAを数学的に包含するモデルではない。
**「PCAの役割を担う、線形再構成制約付きの非線形スペクトル表現学習」**と位置づける。
学習後のencoderを固定して使う手順と、追加corruptionが表現を改善するという仮説は区別する。

### 1.1 MAEを選んだ理由: view間で何を不変にするかを定める難しさ

古材では劣化・樹種・表面性状などに関わる変動が重なり、どの情報を状態差として残し、何に不変な
表現を求めるかが自明でない。今回のTGN・shiftは、同じ化学状態を保つ十分に多様なviewの族として
妥当性を確立したものではない。問題は摂動幅だけでなく、保持する情報、変換の多様性、組合せ・強度・確率の根拠にある。

本研究は対象固有のviewレシピ探索を中心に置かず、可視帯域から元の不可視帯域を予測するMAEを基礎とする。
SimCLR・BYOL・DINOは今回比較しない。これは研究範囲の判断であり、それらに対するMAEの優位性ではない。

| 学習方法 | viewの役割と必要な設計 |
| --- | --- |
| [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) | 同じ入力のviewを正例として対応づける。原論文もaugmentationの組合せを重要な要素とする |
| [BYOL](https://arxiv.org/html/2006.07733v3) | 一方のviewから別viewのtarget表現を予測する。負例がなくてもview設計は必要 |
| [DINO](https://arxiv.org/html/2104.14294v2) | 異なるviewのteacher・student出力分布を対応づける。保持する情報とview・cropの設計が必要 |
| M00 | 可視帯域から隠した帯域の元の値を予測する。maskの単位・割合とtargetを定める |
| M10・M01・M11 | TGN・shiftを加えた可視帯域から、追加摂動前の隠した帯域を予測する |

上三手法に共通するのはview間で対応づける情報の設計であり、負例の有無、損失、augmentation感度は異なる。
対応づけはprojection head等にも依存し、encoderの全情報を厳密に不変にするという意味ではない。
MAEにもpatch分割、mask率、SNV、MSE、16次元圧縮、線形decoderの帰納的制約があり、
予測しやすい帯域間相関が目的の化学状態だけに由来する保証はない。

### 1.2 noise・shiftを加えた理由: denoisingを表現学習の課題にする

**Random maskもdenoisingのcorruptionである。TGN・shiftは欠落に追加するcorruptionとして扱う。**
追加摂動前の観測をtargetに固定し、変形された可視帯域から不可視帯域を予測させることで、
化学状態の探索に有用な帯域間関係を学ぶことを期待する。ノイズ除去性能や指定摂動への耐性自体を主目的とはしない。

Corruptionからの復元を中間表現の学習課題にする原理はDenoising AEに対応する。
Vincentらの分類実験等はこの原理を支えるが、古材NIRで化学状態を抽出できるという実証ではない。
([Vincent et al., 2008, §2–4](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf);
[Vincent et al., 2010, §3](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf))

この課題では復元の基準を明示できる一方、「指定corruptionから元の観測を復元すべきだ」という仮定がある。
同じtargetへの復元は潜在にも間接的な制約を与えるので、view間対応より常に弱い仮定で済むわけではない。
ここでcleanは追加摂動前の観測であり、測定ノイズのない真値ではない。

研究の問いは、**MAE群のcorruption条件が、全可視の表現から得るマップの性質をどう変え、
その領域差を状態差の探索・解釈にどう結びつけられるか**である。CLSは1画素の情報を単一bottleneckへ集約するtokenであり、
化学成分や教師ラベルを表さない。「構造推論」も帯域間関係の推定を指し、分子構造の同定ではない。
再構成lossはクラスタの生成・分離を直接要求しない。

既定の主要比較M11対B0・B1・M00と、M00・M10・M01・M11の2×2 ablationで調べる。
A1等は追加せず、MAEへの追加corruptionの効果をLLA・LFRと既定診断で評価する。
どの入力差が強調・抑制されたかを追う補助診断一式は、現行比較に必須ではないため実施計画から外した
（[必要性の見直し](design/representation_geometry_diagnostics.md)）。

## 2. 実装で確認できる構成

### 2.1 スペクトル全体を単一の潜在ベクトルへ圧縮する

| 部分 | 採用構成 |
| --- | --- |
| 入力 | 1画素の256チャネルSNV。座標・近傍画素は入力しない |
| patch・mask | 連続16チャネル×16 patch。主比較のMAE条件では8 patch可視、CLSを加え9 token |
| encoder | 幅256、8層、8 head、FFN幅1024、GELU、pre-norm、dropout 0、学習可能な位置埋め込み |
| bottleneck | 最終CLSを16次元へ線形射影し、L2正規化 |
| decoder | bias付き `Linear(16, 256)` |
| loss | 追加摂動前のSNVへのMSE。MAEは不可視チャネル、A0は全チャネル |
| 利用時 | 全patch可視・augmentationなしの17 token。encoderを固定して単位潜在をCosine-KMeansへ渡す |

Decoderへ渡る入力由来の情報は単一の $z$ のみで、patch別出力、skip connection、decoder用mask tokenは使わない。
`decoder_num_layers=1` とする本研究の設定についての説明であり、ライブラリの全設定を指すものではない。
根拠は[固定設定](../src/wood_degradation_map/experiments/config.py)、
[mask生成・抽出](../src/wood_degradation_map/experiments/neural.py)、[学習処理](../src/wood_degradation_map/experiments/training.py)、
[参照モデル](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/models/chemo_mae.py)である。

### 2.2 学習している写像と損失

追加摂動前の観測を $x_i\in\mathbb{R}^{256}$、追加corruptionを $g$、可視patch集合を $V_i$、
CLSの射影までを含むencoderを $f_\theta$ とする。通常の非ゼロnorm領域では、

$$
u_i=f_\theta(g(x_i);V_i),\qquad z_i=\frac{u_i}{\lVert u_i\rVert_2},\qquad
\hat{x}_i=Wz_i+b,\quad W\in\mathbb{R}^{256\times16},\quad b\in\mathbb{R}^{256}.
$$

L2正規化にはepsilon保護があり、抽出時は正規化前の非有限値・ゼロnorm・極小normを検査する。
上式は保護が作動しない場合を表す。隠したチャネル集合 $M_i$ は主比較のMAEでは128要素なので、
batch sizeを $B$ とすると、

$$
\mathcal{L}_{\mathrm{MAE}}
=\frac{1}{B}\sum_{i=1}^{B}\frac{1}{128}\sum_{j\in M_i}
\left([Wz_i+b]_j-x_{ij}\right)^2.
$$

M00では $g$ は恒等写像、M10・M01・M11ではTGN・shift・両方を用いる。Targetの値と波長位置は動かさない。
適用確率・強度は[実験プロトコル §4.1.3](design/experiment_protocol.md)に従う。
A0は追加摂動なし・全可視で、内側の和と分母を全256チャネルへ変える。

各stepのlossは不可視帯域だけだが、画素・stepごとのrandom maskにより全帯域が復元対象となる。
ただし予測自体がmaskに依存するため、mask平均した目的関数も全可視・全帯域lossとは同一でない。
学習時のmask課題と利用時の全可視表現は分けて評価する。

Lossにはクラスラベル、KMeans割当、空間項、対比学習項を含まない。AdamWのweight decayとも区別する。
実装は `Trainer._compute_loss` の `loss_type="mse"`、`reduction="mean"`、条件別 `loss_region` に対応する。
([参照Trainer](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/training/trainer.py);
[参照loss](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/models/losses.py))

## 3. PCAとの共通点と、数学的に異なる点

<a id="pca-comparison"></a>

### 3.1 「スペクトルの座標と復元方向を学ぶ」という見方

16成分PCAのtrain平均を $\mu$、直交主成分を $P$ とすると、

$$
t_i=P^{\mathsf T}(x_i-\mu),\qquad \hat{x}^{\mathrm{PCA}}_i=\mu+Pt_i,\qquad P^{\mathsf T}P=I_{16}.
$$

ChemoMAEでは非線形encoderが座標 $z_i$ を求め、共通の $W$ で復元する。行に画素を並べれば、

$$
\hat{X}=ZW^{\mathsf T}+\mathbf{1}b^{\mathsf T}
$$

であり、非線形に推定した座標による制約付き低ランク再構成と読める。

| 観点 | PCA baseline B1 | 本研究のChemoMAE |
| --- | --- | --- |
| 座標推定 | train平均で中心化した線形射影 | 可視patchに依存する非線形写像 |
| 復元 | PCA部分空間への直交射影 | 単位潜在からのアフィン写像 |
| 学習目標 | 全帯域二乗誤差に対応する分散最大化 | A0は全帯域MSE、MAE条件はmasked MSE |
| 軸 | 直交・分散順序を持つ | 直交性・分散順序を課さない |
| L2正規化 | fit後のクラスタリング用scoreへ適用 | 再構成学習のbottleneck内部から適用 |

クラスタリングにはどちらも16次元単位ベクトルを用いるが、PCAの復元は正規化前のscoreで定義する。
線形AEとPCAの同値性は写像・目的関数の条件に依存し、非線形encoder・mask・単位潜在へそのまま拡張できない。
([Baldi & Hornik, 1989](https://doi.org/10.1016/0893-6080(89)90014-2))

### 3.2 非線形なのは座標推定であり、復元可能な範囲には強い制約がある

$$
\hat{x}\in b+\operatorname{col}(W),\qquad \operatorname{rank}(W)\le16.
$$

復元は高々16次元のアフィン部分空間に含まれ、bottleneck前後を非線形にする
[Kramer (1991)](https://doi.org/10.1002/aic.690370209)とは制約が異なる。
同じtrainデータ・前処理・画素重みの全帯域二乗誤差では、正規化前のscoreによる厳密PCAが
同次元以下のアフィン部分空間近似を最適化する。ただし、masked loss・未知試料・クラスタリング品質の優劣は別である。

本研究では、corruptionから座標を推定する学習課題の有用性を、既定CVによるマップの性質と観測スペクトルの解釈から検討する。
[付録B.5](../thesis/appendices/mathematical_details.md#latent-decoder)は、入力差・潜在差・残差とlossの関係を示す数理的補足である。
関係式の成立を、TGN・shiftによる改善やその機構の実証とはしない。対応する診断一式は[未採用候補](design/representation_geometry_diagnostics.md)として残す。

<a id="snv-geometry"></a>

### 3.3 L2正規化の主理由: SNVの一定normを踏まえた自由度の制限

**入力がSNVで一定normになることを踏まえ、潜在にもnormの自由度を持たせず、方向へ情報を集約する**
という設計判断である。標本標準偏差によるSNVでは、非定数スペクトルについて、

$$
\mathcal{M}_{\mathrm{SNV}}
=\{x\in\mathbb{R}^{C}:\mathbf{1}^{\mathsf T}x=0,\ \lVert x\rVert_2=\sqrt{C-1}\}.
$$

$C=256$ では平均ゼロ超平面内の半径 $\sqrt{255}$、内在次元254の球面となる。
方向から共通定数倍でSNV入力を復元できるが、非線形圧縮後のnormも必ず不要になるわけではない。
潜在の正規化は意図的な制約であり、圧縮や正規化の無損失性を意味しない。
潜在に平均ゼロ制約は課していない。
([前処理仕様 §5.6](design/preprocessing.md))

SNVがscore plotに曲線状構造を誘導しうることには既報があり、SNV幾何自体を新規提案としない。
([Fearn et al., 2009](https://doi.org/10.1016/j.chemolab.2008.11.006))

学習中から単位化すれば、復元に使う情報が潜在normのみに載り、後段cosineで無視される食い違いを抑えられる。
単位潜在では、

$$
\lVert z_i-z_j\rVert_2^2=2\left(1-z_i^{\mathsf T}z_j\right),\qquad
\lVert\hat{x}_i-\hat{x}_j\rVert_2^2
=(z_i-z_j)^{\mathsf T}W^{\mathsf T}W(z_i-z_j).
$$

前式は方向によるクラスタリングとの整合性を示す
（[Banerjee et al., 2005](https://jmlr.org/papers/v6/banerjee05a.html)）。
後式の $W^{\mathsf T}W$ には等方性がなく、復元とcosineで各方向の重みは異なる。
入力角度の保存もcollapse防止も保証されない。表現内の分離と退化は既定のsilhouette・occupancy等で診断し、
どの入力差が強調・抑制されたかは、それらの指標や式だけから結論づけない。PCA側の操作の読み方は
[解釈メモ](interpretation_notes.md#pca-snv-geometry-note)に示す。

### 3.4 SAMとの関係: 方向によるスペクトル比較の先行例

Spectral Angle Mapper（SAM）は、非ゼロ観測スペクトル $r$ と参照 $s$ の角度を用いる。

$$
\alpha(r,s)=\arccos\left(\frac{r^{\mathsf T}s}{\lVert r\rVert_2\lVert s\rVert_2}\right).
$$

正のスカラーgainに不変な方向比較の先行例であり、NIR専用ではない。ENVIの標準的説明は反射率を対象とする。
波長依存の照明変化や散乱全般への不変性は意味しない。
([ENVI公式解説](https://www.nv5geospatialsoftware.com/docs/spectralanglemapper.html);
[公式チュートリアル pp. 8–9](https://www.nv5geospatialsoftware.com/portals/0/pdfs/envi/Mapping_Methods.pdf))

SNVは平均も除くため、SNV後の角度は元反射率へのSAMと一般に異なる。
同じ256帯域上の非定数スペクトル $r_i,r_k$ とそのSNV $x_i,x_k$ では、

$$
\frac{x_i^{\mathsf T}x_k}{\lVert x_i\rVert_2\lVert x_k\rVert_2}
=\frac{x_i^{\mathsf T}x_k}{255}=\rho(r_i,r_k)
$$

となり、$\rho$ はPearson相関係数である。学習潜在のcosineがこの値を保存する制約はない。
SAMは方向比較の前例として引用し、潜在正規化の必要性・最適性の根拠とはしない。

## 4. SNVの幾何をaugmentationの設計へ結びつける

### 4.1 augmentationの仮定: 幾何的制約と実際のスペクトル変動を分ける

今回のcorruptionは**SNVの平均ゼロ・一定norm制約に基づく設計**であり、装置の誤差分布や
物理化学的生成過程から導いたものではない。TGNの角度とshift幅は事前固定条件である。

| 実測での変動 | 文献の例と今回のcorruptionとの違い |
| --- | --- |
| 測定noise | 光子・暗電流のshot noiseや読み出しnoiseがある。低SNRはnoiseの相対寄与を表す。本TGNは帯域別SNR・信号依存性を再現しない（[Hamamatsu Photonics §1.2–1.3](https://hub.hamamatsu.com/us/en/technical-notes/image-sensors/image-sensors-product-selection.html)） |
| 装置由来の波長位置ずれ | HISUIのspectral smileは検出器列・波長に依存する。本shiftは画素内で一様な軸方向移動であり、その依存性や応答幅を再現しない（[Yamamoto et al., 2022 §I](https://doi.org/10.1109/TGRS.2022.3190486)） |
| 試料状態によるピーク変化 | 水・glucose水溶液の昇温に伴う見かけの移動は重なった帯域の相対強度変化とも解釈される。状態情報を含み、一様shiftや除去すべき誤差とは限らない（[Cui et al., 2016 §3.1](https://pubs.rsc.org/en/content/articlehtml/2016/ra/c6ra18912a)） |

これらは現象の例であり、本古材データの変動原因・大きさを同定したものではない。
SNV制約を満たす生成点が実在の化学状態に対応する保証もない。幾何保存を化学状態保存と同一視せず、
指定corruptionからの復元が有用な差を残すか、重要な微小差まで弱めるかを評価・解釈で検討する。

<a id="spectral-augmentation"></a>

### 4.2 提案するaugmentation: Tangent Gaussian NoiseとFractional Shift

提案内容は、SNV制約を保つTGN・Fractional Shiftの定義とmasked denoisingへの組込みである。
操作の発想は[著者解説 §3.2–3.3](https://zenn.dev/mantis_ryuji/articles/e17b4d223cd7da)、設定は
[固定仕様](design/experiment_protocol.md)とChemoMAE v0.2.2の `SpectraAugmenter` に基づく。

理想演算で $C=256$、$r=\sqrt{C-1}$、平均ゼロ・norm $r$ の入力 $x$、$u=x/r$ とする。
TGNはGaussian方向から平均方向と半径方向を除き、接方向へ回転する。

$$
P=I-\frac{\mathbf{1}\mathbf{1}^{\mathsf T}}{C},\qquad
\epsilon\sim\mathcal{N}(0,I),\qquad
v=P\epsilon-u\bigl(u^{\mathsf T}P\epsilon\bigr),\qquad q=\frac{v}{\lVert v\rVert_2}.
$$

$$
T_{\mathrm{TGN}}(x)=r\bigl(\cos\theta\,u+\sin\theta\,q\bigr),\qquad
\theta\sim U(0,\pi/36).
$$

$\lVert v\rVert_2>0$ なら $q$ は平均ゼロ・入力に直交・単位normで、出力もSNV制約を保つ。
入力との角度は $\theta$、球面弧長は $r\theta$。Gaussianは方向候補の分布であり、角度や最終残差の分布ではない。

等間隔波長grid上の線形補間・端点値延長を $S_\delta$ とすると、Fractional Shiftは、

$$
y=P S_\delta(x),\qquad
T_{\mathrm{FS}}(x)=r\frac{y}{\lVert y\rVert_2},\qquad \delta\sim U(-2,2).
$$

$\lVert y\rVert_2>0$ ならSNV制約へ戻るが、同じshift幅でも角度変化は入力形状に依存する。
実装は `eps=1e-8` で退化候補を保護し、元入力を保持する。等式の保存は理想演算についての記述である。

著者記事との違いは次のとおりであり、その設定をそのまま本研究へ移さない。

| 項目 | 記事 | 本研究 |
| --- | --- | --- |
| SNV半径 | 母標準偏差に対応する $\sqrt C$ | 標本標準偏差に対応する $\sqrt{C-1}$ |
| TGN強度の抽選 | cosine $\rho$ の一様分布 | 角度 $\theta$ の一様分布。$\rho=\cos\theta$ でも分布は一致しない |
| shift軸 | 波数軸の説明 | 等間隔波長grid |

M11は各操作を画素ごとに確率0.5で適用し、順序をbatchごとにランダム化する。
固定順序で常に両操作を行う構成ではない。角度制御・制約保存の導出は
[付録B](../thesis/appendices/mathematical_details.md)を参照する。
現行ablationには加法noiseや再投影なしshiftとの比較がなく、幾何保存そのものの優位性は単独に検証しない。

<a id="augmentation-vicinity"></a>

### 4.3 VRM的な解釈: 表現の幾何に沿って学習信号を広げる

観測点の周囲に、SNVに即した近傍と復元targetを定める設計としても読める。
これは2026-09-11の事後的な解釈であり、VRMから採用条件を導いたという記録ではない。

Mask前の追加corruption分布を $\nu(\tilde{x}\mid x_i)$（操作しない確率を含む）、
encoder・decoder全体を $h_\psi$、隠した帯域のMSEを $\ell_M$ とすれば、

$$
\widehat R_\nu(\psi)=\frac1n\sum_{i=1}^n
\mathbb E_{\tilde x\sim\nu(\cdot\mid x_i)}\mathbb E_M
\left[\ell_M\bigl(h_\psi(\tilde x;V(M)),x_i\bigr)\right].
$$

$V(M)$ は可視patch集合である。近傍は生成入力だけでなく、元観測 $x_i$ への復元という仮定も含む。
VRMは学習信号の広げ方を説明する枠組みとして用い、教師ありVRMの保証を移さない。

SNV制約集合上の二点には、

$$
d_{\mathcal M}(x,\tilde x)=r\arccos\left(\frac{x^{\mathsf T}\tilde x}{r^2}\right),\qquad
\lVert x-\tilde x\rVert_2=2r\sin\left(\frac{d_{\mathcal M}(x,\tilde x)}{2r}\right)
$$

が成立する。TGN単独は $d_{\mathcal M}=r\theta$ で強度を指定できるが、距離だけでは生成方向・targetは決まらない。
弦長と測地距離は単調対応するため、Euclidean距離自体を不適切とはしない。
再投影なしの非退化Gaussian加算は制約集合を確率1で外れるが、corruptionとして無意味という結論にはならない。
加算後の再SNVなら集合へ戻るため、TGNとの差は生成方向・角度分布まで含めて扱う。

区別すべき点は、既知の平均・norm制約への整合、復元課題としての有用性、物理化学的な実現可能性である。
保証するのはmask前の全帯域入力の制約であり、可視部分・decoder出力・実データのsupportではない。
この制約に沿う近傍を採用する価値は、次節の比較と解釈で検討する。

## 5. 本研究で主張できる特徴と、評価を待つ事項

| 主張・問い | 根拠と上限 |
| --- | --- |
| 提案する設計 | SNV制約を保つTGN・shiftをmasked denoisingへ組み込み、単一16次元単位潜在へ集約する |
| 固定表現の利用 | 学習後のencoderとtrainでfitした中心をtestへ適用する。後段fine-tuningを要しない現行pipelineの事実 |
| 追加corruptionの効果 | MAE群の2×2比較・交互作用をLLA・LFRと既定診断で評価する。化学状態をより安定して反映することは仮説であり、指標や再構成lossからは保証されない |
| 指定摂動への安定性 | LFRで割当の維持を測る。学習と同じ種類・強度の人工摂動への結果であり、実測誤差全般や化学情報保持へ外挿しない |
| どの差が強調・抑制されたか | 現行実験の実証範囲に含めない。[補助診断の旧案](design/representation_geometry_diagnostics.md)は未採用で、付録B.5は数理的補足に留める |
| 化学的な対応 | NIR代表・差スペクトルと位置対応FT-IRによる解釈。FT-IRの測定設計はOpen、結果未確認 |
| 構成の最適性 | 層数・潜在次元・線形decoder・正規化の最適性や、未比較SSLへの優位性は扱わない |

5-fold・3反復のマップ評価は[評価指標](design/evaluation_metrics.md)に従う。
LLA・LFRだけでなくoccupancy・反復間ARI等を併読し、単一クラスタ化等の退化を区別する。
Cosine-silhouetteは各表現内の幾何診断であり、化学的妥当性の共通尺度ではない。
化学的な対応は[FT-IR計画](design/visualization_and_interpretation.md#ftir-interpretation)に基づき別途検討する。

比較・解釈に必要な制約は次の四つである。

- **学習と利用：** MAEは学習時8 patch、利用時16 patch可視。全可視maskを明示し、`eval()`だけで切り替わるとはしない。
- **軸の非一意性：** 直交 $Q$ による $z'=Qz,\ W'=WQ^{\mathsf T}$ は復元とcosineを変えない。Decoder列を純粋成分や分散順序付きの軸とみなさない。
- **欠測への外挿：** 全帯域SNV後にmaskするため、未測定帯域があり同じSNVを計算できない条件への対応を実証しない。
- **比較単位：** M00対A0は入力maskとloss領域の両方が違い、B1対NNも非線形性・正規化段階等が異なる。単一要素の効果に読み替えない。

### 5.1 論文で説明する貢献と、新規性の確認範囲

貢献は、(1) SNV制約を保つTGN・Fractional Shiftの具体的設計、(2) masked denoisingへの統合、
(3) 正解劣化ラベルを用いない試料単位マップ比較、(4) 古材への適用と位置対応FT-IRによる解釈、に分けて説明する。
(4)の未実施部分は計画であり、実証済みの貢献に含めない。

提案の具体性、先行研究との差、有用性は別に確認する。Denoising一般・spectral MAE・SNV幾何には既報があり、
TGNや統合構成の網羅的な優先性調査は未完了である。「初」を検討する場合は、名称の異なる操作も含め、
SNV後のnoiseと再正規化、接方向・角度制御、shift再投影、masked targetを照合し、検索範囲・日付・差分を残す。

方法論上は「既知のSNV幾何を、学習用近傍と復元課題の設計へ結びつける」と説明できる。
この接続が従来存在しなかったという歴史的断定は、現在の文献確認からは支持できない。

## 6. クラスタリング方法への依存性

同じ学習済み表現へのvMF mixtureを補助実験として計画する。表現についての結論が分割方法を変えても
保たれるかを調べるものであり、実施範囲・利用版・採用前検証は
[実験プロトコル §5.2](design/experiment_protocol.md#vmf-supplementary)で管理する。

## 7. 出典と確認範囲

### 7.1 関連研究との対応

[関連研究](related_work.md)は非線形PCAの用途、Denoising AEの学習原理、Raman unmixing AEの線形decoder、
Raman SMAEのmask・クラスタリング、LeafVAEの空間マッピングを整理する。各研究の性能は本古材NIRへ外挿しない。

### 7.2 実装・文献の確認範囲

実装の根拠は本書のコード参照とChemoMAE v0.2.2の参照ソースである。
文献の書誌・確認箇所・未照合範囲は[関連研究 §2](related_work.md#2-参考文献と確認範囲)に残す。
今回の文書整理で学習・評価・文献再調査を行ったものではない。
