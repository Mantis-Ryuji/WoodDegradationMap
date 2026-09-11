# 3.5 Cosineクラスタリングと空間マッピング

## 3.5.1 クラスタリングへ渡す表現

各画素に対し、前節の全帯域可視で得る16次元潜在をクラスタリングへ渡す。比較対象として、SNVを直接用いるB0と、PCAによる16次元の表現を用いるB1も同じcosineクラスタリングの枠組みで扱う。PCAはfit対象の画素から求めた平均で中心化し、追加のautoscalingとwhiteningは行わない。

画素の添字を $p$、その最終SNVスペクトルを $\boldsymbol{x}_p$、全帯域可視で抽出した潜在を $\boldsymbol{z}^{\mathrm{full}}_p$ とする。表現の次元を $d_{\mathrm{feat}}$、B1の中心化平均を $\overline{\boldsymbol{x}}_{\mathrm{fit}}$、主成分方向を列に持つ行列を $U_{\mathrm{PCA}}\in\mathbb{R}^{256\times16}$ とする。正規化前の表現 $\boldsymbol{\psi}_p$ とクラスタリング入力 $\boldsymbol{\xi}_p$ は

$$
\boldsymbol{\psi}_p=
\begin{cases}
\boldsymbol{x}_p, & \mathrm{B0},\\
U_{\mathrm{PCA}}^{\mathsf T}
(\boldsymbol{x}_p-\overline{\boldsymbol{x}}_{\mathrm{fit}}),
& \mathrm{B1},\\
\boldsymbol{z}^{\mathrm{full}}_p, & \text{学習した表現},
\end{cases}
\qquad
\boldsymbol{\xi}_p
=
\frac{\boldsymbol{\psi}_p}{\|\boldsymbol{\psi}_p\|_2}
\tag{3.16}
$$

である。$d_{\mathrm{feat}}=256$ はB0、$d_{\mathrm{feat}}=16$ はPCAおよび学習した表現に対応する。$\overline{\boldsymbol{x}}_{\mathrm{fit}}$ はPCAのfit対象画素の平均であり、CVではtrain試料のfit対象画素だけから求める。NNの潜在はすでに学習中から単位normであり、式(3.16)の単位化は後段で共通の方向表現を扱うことを示している。

正規化前後に非有限値、ゼロnorm、および数値的に単位化できない極小normを検査する。これらはcosine比較を定義できない表現として記録し、条件ごとに画素を無言で除外することはしない。数値保護の設定は付録Cに示す。

B0の単位化は元のSNVのcosine関係を変えない。一方、PCAやencoderによる変換は画素間の角度関係を変え得る。B1とNNの最終的なクラスタリング入力は、いずれも16次元空間内の単位球面 $\mathbb{S}^{15}$ 上にある。ただし、B1はfit平均で中心化したSNVの線形射影を後から単位化し、NNは単位潜在をdecoderへ渡す復元課題を通じて座標を学習する。出力次元とnormが同じでも、異なる表現が同じ幾何を持つことは前提にしない。

PCA score $\boldsymbol{\psi}_p$ のnormは、fit平均からのスペクトル形状のずれが、選んだ主成分部分空間にどれだけ含まれるかを表す。SNV前の反射率の絶対的な大きさではない。後段の単位化はこのscoreの大きさをクラスタリングに用いない処理である。導出と解釈の範囲は[付録B.5.4](../../appendices/mathematical_details.md#snv-coordinate-learning)に示す。

## 3.5.2 Cosine-KMeans

クラスタ数を $K$、その添字を $k=1,\ldots,K$、第 $k$ クラスタの単位中心を $\boldsymbol{c}_k$ とする。単位表現 $\boldsymbol{\xi}_p$ と中心のcosine不類似度 $d_{\cos}$ を

$$
d_{\cos}(\boldsymbol{\xi}_p,\boldsymbol{c}_k)
=
1-\boldsymbol{\xi}_p^{\mathsf T}\boldsymbol{c}_k,
\qquad
\|\boldsymbol{\xi}_p\|_2=\|\boldsymbol{c}_k\|_2=1
\tag{3.17}
$$

とする。中心推定に用いる画素集合を $\mathcal{F}$、その画素数を $N_{\mathcal{F}}$ とし、画素 $p$ の割当ラベルを $\ell_p\in\{1,\ldots,K\}$ とする。Cosine-KMeansは平均cosine不類似度 $J$

$$
J=
\frac{1}{N_{\mathcal{F}}}
\sum_{p\in\mathcal{F}}
\left(1-\boldsymbol{\xi}_p^{\mathsf T}\boldsymbol{c}_{\ell_p}\right)
\tag{3.18}
$$

を小さくするように、割当と中心更新を反復する。固定中心への割当は

$$
\ell_p=
\underset{k\in\{1,\ldots,K\}}{\operatorname{arg\,max}}
\ \boldsymbol{\xi}_p^{\mathsf T}\boldsymbol{c}_k
\tag{3.19}
$$

で与える。$\operatorname{arg\,max}$ は内積が最大になるクラスタの添字を選ぶ操作を表す。第 $k$ クラスタに割り当てられたfit画素集合を $\mathcal{F}_k$、その表現の和を $\boldsymbol{g}_k$ とする。非空かつ和が非ゼロの通常の場合、中心更新は

$$
\boldsymbol{g}_k=\sum_{p\in\mathcal{F}_k}\boldsymbol{\xi}_p,
\qquad
\boldsymbol{c}_k
=
\frac{\boldsymbol{g}_k}{\|\boldsymbol{g}_k\|_2}
\tag{3.20}
$$

と表せる。入力表現を固定した下で割当と中心を求めるため、この段階からencoderへの学習は行わない。

初期化にはcosine不類似度に基づくk-means++型の方法を用い、1 fit内の初期化は1回とする。最大反復数は500、相対変化の許容値は $10^{-4}$ とする。空クラスタへの対応と停止の詳細は付録Cに示す。式(3.18)は最小化の対象を示すものであり、大域的な最小値への到達を保証するものではない。

比較では $K\in\{2,4,6,8,10,12,14\}$ を共通に用い、代表表示を $K=8$ とする。これらは共通条件として固定し、各表現の結果から異なる $K$ を選んで比較することはしない。クラスタ数依存性の評価は第4章で定義する。

CVではPCA、encoder、クラスタ中心のfitをtrain試料に限定する。Test画素には固定した変換器と中心を適用し、試料ごとのクラスタリング再fitを行わない。全体fitによる解釈では全試料からの共通抽出画素をfit対象とし、その結果はCVとは区別する。

## 3.5.3 画素位置へのラベル配置

有効画素集合 $\Omega$ に対して得たラベルを、保存しておいた座標対応へ戻す。行・列の座標 $(u,v)$ の画素が $p$ に対応するとき、ラベルマップ $\operatorname{Label}(u,v)$ を

$$
\operatorname{Label}(u,v)=
\begin{cases}
\ell_p, & p\in\Omega,\\
0, & \text{背景または品質条件による除外画素}
\end{cases}
\tag{3.21}
$$

と定義する。クラスタラベルは $1,\ldots,K$、解析対象外は0とする。実装が返す0始まりのクラスタ番号は、マップ保存時に1を加えて区別する。

この処理では、画素座標を使ってラベルを元の位置へ配置する。空間情報はencoderやCosine-KMeansの特徴として使わず、近傍の多数決、ラベルの平滑化、空間正則化も加えない。したがって、マップに現れる空間的なまとまりは、共通の表現変換とクラスタ割当を各画素のスペクトルへ適用した結果として観察する。

クラスタ番号は任意の識別子であり、大きさの順序や劣化の強弱を表さない。異なる条件・fit間で表示番号を整列する場合も、分割そのものを変更せず、対応を見やすくするための操作として扱う。表示の基準と解釈範囲は第4章で定義する。

本章で得られるのは、スペクトル表現に基づく空間的な分割である。マップの空間的一貫性、指定摂動への安定性、学習・クラスタリング反復間の再現性は別途比較し、化学的意味は観測スペクトルや位置対応測定から検討する。

---

執筆メモ：PCAおよび球面クラスタリングの原典は最終稿で引用を追加する。現行仕様は[実験プロトコル](../../../docs/design/experiment_protocol.md)、実装は[baseline](../../../src/wood_degradation_map/experiments/baselines.py)、[クラスタリング](../../../src/wood_degradation_map/experiments/clustering.py)と照合した。vMFは数値仕様が未確定のため、本節では主手法のCosine-KMeansを記述する。
