# 付録B 数理的補足

第3章の定義から導かれる性質を、入力の制約、学習課題、復元モデルの順に示す。有限な入力と非ゼロの正規化分母を仮定する理想演算を扱い、数値保護はB.6で補足する。性能や化学的妥当性は実験で検討する。

| 論点 | 導出する内容 | 対応節 |
| --- | --- | --- |
| 入力と摂動 | SNVが定める集合と、TGN・Fractional Shiftがその制約を保つ条件 | [B.1](#snv-geometry)〜[B.3](#fractional-shift) |
| 学習課題 | 不可視帯域の誤差が全帯域にわたる復元課題を作る仕組み | [B.4](#masked-objective) |
| 表現と復元 | 入力差の拡大・縮小、クラスタ内外の変動、SVD・残差とmasked lossの対応 | [B.5](#latent-decoder) |
| 数値実装 | 理想式の照合と退化条件 | [B.6](#numerical-geometry) |

記号の対応は[共通記号表](../notation.md)を参照する。

<a id="snv-geometry"></a>

## B.1 SNVの制約と幾何

### B.1.1 平均とnorm

以下では一つの有効画素に注目し、第3.2節の画素添字 $p$ を省略する。補間後のチャネル数を $C=256$、チャネル添字を $j=0,\ldots,C-1$、補間反射率を $\widetilde{\boldsymbol{R}}\in\mathbb{R}^{C}$、その第 $j$ 成分を $\widetilde{R}_j$ とする。その画素内平均 $\mu$ と標本標準偏差 $s>0$ は、本文の $\mu_p,s_p$ と同じ量である。$\boldsymbol{1}$ は全成分1の $C$ 次元ベクトル、上付き $\mathsf T$ は転置、$\|\cdot\|_2$ はL2 normを表す。SNVスペクトルを $\boldsymbol{x}=(\widetilde{\boldsymbol{R}}-\mu\boldsymbol{1})/s$ とすると、

$$
\boldsymbol{1}^{\mathsf T}\boldsymbol{x}
=
\frac{1}{s}
\left(\sum_{j=0}^{C-1}\widetilde{R}_j-C\mu\right)
=0
\tag{B.1}
$$

である。また、$s^2=(C-1)^{-1}\sum_j(\widetilde{R}_j-\mu)^2$ なので、

$$
\|\boldsymbol{x}\|_2^2
=
\frac{\sum_j(\widetilde{R}_j-\mu)^2}{s^2}
=C-1.
\tag{B.2}
$$

したがって、SNVスペクトルのnormを $\rho=\sqrt{C-1}$ とすれば、SNVの平均・norm制約を満たす集合 $\mathcal{S}_{\mathrm{SNV}}$ を

$$
\mathcal{S}_{\mathrm{SNV}}
=
\left\{
\boldsymbol{x}\in\mathbb{R}^{C}:
\boldsymbol{1}^{\mathsf T}\boldsymbol{x}=0,\ 
\|\boldsymbol{x}\|_2=\rho
\right\}
\tag{B.3}
$$

と書ける。SNV入力はこの集合に属する。平均ゼロの超平面は $C-1$ 次元であり、その中の半径 $\rho$ の球面は $C-2$ 次元である。$C=256$ では $\rho=\sqrt{255}$、球面の次元は254となる。この集合はSNVの制約を満たす点の集合であり、実在する材料スペクトルの集合そのものではない。

### B.1.2 中心化の射影

平均を除く中心化行列 $P=I_C-\boldsymbol{1}\boldsymbol{1}^{\mathsf T}/C$ を用いる。ここで $I_C$ は $C$ 次元単位行列である。この行列は

$$
P^{\mathsf T}=P,\qquad
P^2=P,\qquad
P\boldsymbol{1}=\boldsymbol{0}
\tag{B.4}
$$

を満たす。$\boldsymbol{0}$ は $C$ 次元ゼロベクトルである。中心化された補間反射率 $P\widetilde{\boldsymbol{R}}$ は平均ゼロなので、SNVは

$$
\boldsymbol{x}
=
\rho\frac{P\widetilde{\boldsymbol{R}}}{\|P\widetilde{\boldsymbol{R}}\|_2}
\tag{B.5}
$$

と書き換えられる。ただし $P\widetilde{\boldsymbol{R}}\neq\boldsymbol{0}$ とする。

SNV前のスペクトルに、全帯域で共通する正のscale $\kappa>0$ とoffset $\beta$ を与えると、
$P(\kappa\widetilde{\boldsymbol{R}}+\beta\boldsymbol{1})=\kappa P\widetilde{\boldsymbol{R}}$ である。したがって、

$$
\operatorname{SNV}(\kappa\widetilde{\boldsymbol{R}}+\beta\boldsymbol{1})
=
\operatorname{SNV}(\widetilde{\boldsymbol{R}})
\tag{B.6}
$$

となる。この性質は画素内の共通offset・scaleを除くことを表し、波長依存の歪みやすべての測定変動を除くという意味ではない。

<a id="tgn-geometry"></a>

## B.2 TGNの接方向と制約保存

### B.2.1 接空間への射影

チャネル数を $C=256$、SNVの制約集合を $\mathcal{S}_{\mathrm{SNV}}$ とする。SNV入力を $\boldsymbol{x}\in\mathcal{S}_{\mathrm{SNV}}$、そのnormを $\rho=\sqrt{C-1}$、入力の単位方向を $\boldsymbol{n}=\boldsymbol{x}/\rho$ とする。$\boldsymbol{x}$ における接方向は、平均ゼロの超平面内にあり、かつ半径方向 $\boldsymbol{n}$ と直交する方向である。その方向の集合（接空間）を $\mathcal{T}_{\boldsymbol{n}}$ とすると、

$$
\mathcal{T}_{\boldsymbol{n}}
=
\left\{
\boldsymbol{q}_{\mathrm{raw}}\in\mathbb{R}^{C}:
\boldsymbol{1}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}=0,\ 
\boldsymbol{n}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}=0
\right\}
\tag{B.7}
$$

である。集合を記述する $\boldsymbol{q}_{\mathrm{raw}}$ は未正規化の接方向ベクトルを表す。Gaussian方向候補 $\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},I_C)$ を用いる。$\mathcal{N}(\boldsymbol{0},I_C)$ は平均ゼロ・共分散が単位行列の $C$ 次元正規分布、$P$ は式(B.4)の中心化行列である。この候補から

$$
\boldsymbol{q}_{\mathrm{raw}}
=
P\boldsymbol{\epsilon}
-
\boldsymbol{n}
\left(\boldsymbol{n}^{\mathsf T}P\boldsymbol{\epsilon}\right)
\tag{B.8}
$$

を作ると、$\boldsymbol{1}^{\mathsf T}P=\boldsymbol{0}^{\mathsf T}$ と
$\boldsymbol{1}^{\mathsf T}\boldsymbol{n}=0$ により
$\boldsymbol{1}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}=0$ となる。また、

$$
\boldsymbol{n}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}
=
\boldsymbol{n}^{\mathsf T}P\boldsymbol{\epsilon}
-
(\boldsymbol{n}^{\mathsf T}\boldsymbol{n})
\boldsymbol{n}^{\mathsf T}P\boldsymbol{\epsilon}
=0
\tag{B.9}
$$

である。したがって、$\|\boldsymbol{q}_{\mathrm{raw}}\|_2>0$ のとき
$\boldsymbol{q}=\boldsymbol{q}_{\mathrm{raw}}/\|\boldsymbol{q}_{\mathrm{raw}}\|_2$ は平均ゼロの単位接方向となる。

$P\boldsymbol{n}=\boldsymbol{n}$ なので、式(B.8)は
$\boldsymbol{q}_{\mathrm{raw}}=(P-\boldsymbol{n}\boldsymbol{n}^{\mathsf T})\boldsymbol{\epsilon}$ とも書ける。
行列 $P-\boldsymbol{n}\boldsymbol{n}^{\mathsf T}$ は対称かつべき等であり、
$\mathcal{T}_{\boldsymbol{n}}$ への直交射影である。接空間の次元は $C-2$ である。

### B.2.2 回転後の平均とnorm

回転角を $\alpha$（radian）、単位接方向を $\boldsymbol{q}$ として、TGN出力を $\boldsymbol{x}_{\mathrm{TGN}}=\rho(\cos\alpha\,\boldsymbol{n}+\sin\alpha\,\boldsymbol{q})$ とする。
両方向の平均がゼロなので $\boldsymbol{1}^{\mathsf T}\boldsymbol{x}_{\mathrm{TGN}}=0$ である。
また、$\boldsymbol{n}^{\mathsf T}\boldsymbol{q}=0$ から、

$$
\begin{aligned}
\|\boldsymbol{x}_{\mathrm{TGN}}\|_2^2
&=
\rho^2\left(
\cos^2\alpha\,\|\boldsymbol{n}\|_2^2
+
\sin^2\alpha\,\|\boldsymbol{q}\|_2^2
+
2\cos\alpha\sin\alpha\,\boldsymbol{n}^{\mathsf T}\boldsymbol{q}
\right)\\
&=\rho^2
\end{aligned}
\tag{B.10}
$$

となる。したがってTGNは $\mathcal{S}_{\mathrm{SNV}}$ 内の変換となる。

### B.2.3 角度、弧長、残差

入力と出力のcosine類似度は

$$
\frac{\boldsymbol{x}^{\mathsf T}\boldsymbol{x}_{\mathrm{TGN}}}
{\|\boldsymbol{x}\|_2\|\boldsymbol{x}_{\mathrm{TGN}}\|_2}
=\cos\alpha
\tag{B.11}
$$

である。採用範囲 $0\leq\alpha\leq\pi/36$ では両者の角度は $\alpha$、半径 $\rho$ の球面上の短い弧長は $\rho\alpha$ となる。一方、差ベクトルのnormは

$$
\|\boldsymbol{x}_{\mathrm{TGN}}-\boldsymbol{x}\|_2^2
=
2\rho^2(1-\cos\alpha)
=
4\rho^2\sin^2(\alpha/2)
\tag{B.12}
$$

である。角度、弧長、Euclideanな残差normは異なる量であり、強度を記述する際には区別する。

本研究では $\alpha\sim\mathcal{U}(0,\pi/36)$ とする。$\mathcal{U}$ は指定区間の連続一様分布、$\pi$ は円周率であり、上限は5度に対応する。Cosine類似度 $\cos\alpha$ を一様に抽選する分布とは異なる。方向候補がGaussianであっても、射影・単位化・回転を経た最終残差を独立な加算Gaussian雑音とみなすことはできない。

<a id="fractional-shift"></a>

## B.3 Fractional Shiftの補間と再投影

### B.3.1 参照位置と端点値延長

SNV入力を $\boldsymbol{x}$、そのチャネル数を $C=256$、出力チャネル添字を $j=0,\ldots,C-1$、チャネル単位のshift量を $\delta$ とする。出力位置 $j$ が参照する連続位置を $\zeta_j$、その左右の整数indexを $\iota_j^-,\iota_j^+$、右側の値に掛ける補間重みを $\omega_j^{\mathrm{FS}}$ として、

$$
\zeta_j=j-\delta,\qquad
\iota_j^-=\lfloor\zeta_j\rfloor,\qquad
\iota_j^+=\iota_j^-+1,\qquad
\omega_j^{\mathrm{FS}}=\zeta_j-\lfloor\zeta_j\rfloor
\tag{B.13}
$$

と定義する。$\lfloor\cdot\rfloor$ は床関数であり、その値以下の最大の整数を返す。$\iota_j^+$ は常に左側indexに1を加えた値であり、参照位置が整数のときもこの定義を用いる。参照範囲を画像ではなくスペクトルの両端に収めたindexを
$\bar{\iota}_j^-=\min(C-1,\max(0,\iota_j^-))$、
$\bar{\iota}_j^+=\min(C-1,\max(0,\iota_j^+))$
とする。上付きの $-,+$ は左右の参照点の区別であり、数値の符号ではない。線形補間操作 $S_\delta$ の第 $j$ 成分は、

$$
[S_\delta(\boldsymbol{x})]_j
=
(1-\omega_j^{\mathrm{FS}})x_{\bar{\iota}_j^-}
+\omega_j^{\mathrm{FS}}x_{\bar{\iota}_j^+}
\tag{B.14}
$$

である。$x_{\bar{\iota}_j^-},x_{\bar{\iota}_j^+}$ はそれぞれ、入力スペクトルの参照indexの値を表す。範囲外では二つの参照indexが同じ端点になるため、その端点値が延長される。正の $\delta$ は大きいindex側へ、負の $\delta$ は小さいindex側へ特徴を移す向きに対応する。

等間隔gridの隣接チャネル間の波長差を $\Delta\lambda$ とすれば、補間操作の軸方向移動量は $\delta\Delta\lambda$ である。ただし、再中心化・再正規化後の変化全体を、物理的な吸収帯の単純移動や実測の校正誤差と同一視しない。

### B.3.2 制約への復帰

補間行列は一般に直交行列ではなく、端点処理も含むため、補間だけで入力の平均・normが保たれるとは限らない。$P$ を式(B.4)の中心化行列、$\rho=\sqrt{C-1}$ をSNV入力のnormとする。補間後に平均を除いたスペクトルを $\boldsymbol{y}=P S_\delta(\boldsymbol{x})$ とし、$\|\boldsymbol{y}\|_2>0$ の場合に、Fractional Shiftの最終出力 $\boldsymbol{x}_{\mathrm{FS}}$ を

$$
\boldsymbol{x}_{\mathrm{FS}}
=
\rho\frac{\boldsymbol{y}}{\|\boldsymbol{y}\|_2}
\tag{B.15}
$$

と定めれば、$P$ の性質から $\boldsymbol{1}^{\mathsf T}\boldsymbol{x}_{\mathrm{FS}}=0$、
正規化から $\|\boldsymbol{x}_{\mathrm{FS}}\|_2=\rho$ となる。この再投影により出力は $\mathcal{S}_{\mathrm{SNV}}$ に属する。

TGNと異なり、固定した $\delta$ から入力・出力の角度は一意に決まらない。補間結果が元のスペクトル形状と端点値に依存するためである。また、再投影がSNV制約を保つことは、化学的な情報が保存される証明ではない。

<a id="masked-objective"></a>

## B.4 Masked lossと全帯域の学習対象

一つの画素の追加摂動前のSNV観測を $\boldsymbol{x}$ とし、画素添字 $p$ を省略する。不可視チャネル集合を $\mathcal{H}$、その要素数を固定値 $N_{\mathrm{hide}}>0$ とする。主比較では $N_{\mathrm{hide}}=128$ である。適用抽選・順序・強度を含む追加摂動の変換を $\mathcal{A}$ とし、モデルparameterを固定した下で、摂動とmaskに依存する再構成値を $\widehat{\boldsymbol{x}}(\mathcal{A},\mathcal{H})$ と書く。一画素の不可視チャネル上の平均二乗誤差 $\mathcal{L}_{\mathrm{one}}$ は、

$$
\mathcal{L}_{\mathrm{one}}(\mathcal{A},\mathcal{H})
=
\frac{1}{N_{\mathrm{hide}}}\sum_{j=0}^{C-1}
\operatorname{Ind}[j\in\mathcal{H}]\,
\left(\widehat{x}_j(\mathcal{A},\mathcal{H})-x_j\right)^2
\tag{B.16}
$$

である。ここで $C=256$ は全チャネル数、$j$ はチャネル添字、$\operatorname{Ind}[j\in\mathcal{H}]$ は $j$ が不可視なら1、それ以外なら0を返す指示関数である。$x_j$ と $\widehat{x}_j$ はそれぞれ観測と再構成値の第 $j$ 成分を表す。

等長patchを一様に選ぶため、各チャネルが不可視となる確率は $N_{\mathrm{hide}}/C$ である。$\Pr$ はmask抽選による確率、$\mathbb{E}_{\mathcal{A},\mathcal{H}}$ は追加摂動とmaskの抽選に関する期待値、縦線の後の $j\in\mathcal{H}$ はそのチャネルを不可視とした条件を表す。二乗誤差の期待値が有限である場合、

$$
\begin{aligned}
\mathbb{E}_{\mathcal{A},\mathcal{H}}[\mathcal{L}_{\mathrm{one}}(\mathcal{A},\mathcal{H})]
&=
\frac{1}{N_{\mathrm{hide}}}\sum_j
\Pr(j\in\mathcal{H})\,
\mathbb{E}_{\mathcal{A},\mathcal{H}}
\left[
(\widehat{x}_j(\mathcal{A},\mathcal{H})-x_j)^2
\mid j\in\mathcal{H}
\right]\\
&=
\frac{1}{C}\sum_j
\mathbb{E}_{\mathcal{A},\mathcal{H}}
\left[
(\widehat{x}_j(\mathcal{A},\mathcal{H})-x_j)^2
\mid j\in\mathcal{H}
\right]
\end{aligned}
\tag{B.17}
$$

となる。この式は、すべてのチャネルについて「そのチャネルを不可視にしたときの復元誤差」が目的関数に含まれることを示す。Patch内のmask指示は相関するが、この期待値の展開にはチャネル間の独立性を必要としない。

一方、$\widehat{x}_j(\mathcal{A},\mathcal{H})$ 自体がmaskに依存するため、式(B.17)は全可視入力の再構成誤差を平均する式には置き換えられない。各stepのloss対象が不可視帯域であることと、学習課題が全帯域にわたることは両立するが、全可視denoisingと同じ目的関数を用いているわけではない。

<a id="latent-decoder"></a>

## B.5 表現幾何と再構成損失の関係

本節では、単位潜在と線形decoderのもとで、入力差・潜在差・再構成残差の関係を、画素対、クラスタ平均、同一画素への摂動について示す。これらは条件付きの数理的関係であり、TGN・shiftによって実際に何が強調・抑制されたか、またそれが指標改善を引き起こしたかを示す結果ではない。関係式ごとの補助実験を現行の実施計画には含めず、条件間の効果は第4章の既定比較・評価で検討する。

<a id="pairwise-gain"></a>

### B.5.1 入力差と潜在差の比較

画素 $p$ のSNVを $\boldsymbol{x}_p$、そのnormを $\rho=\sqrt{C-1}$、単位方向を $\boldsymbol{n}_p=\boldsymbol{x}_p/\rho$ とする。固定した全帯域可視のencoderと単位化を合わせて $F$ と書き、本節では $\boldsymbol{z}_p=F(\boldsymbol{x}_p)=\boldsymbol{z}^{\mathrm{full}}_p$ と略記する。入力・潜在とも単位方向で比較すれば、SNVのnorm $\rho$ と潜在norm 1の尺度差を除ける。二画素 $p,q$ に対して、

$$
\begin{aligned}
d_{\mathrm{in}}(p,q)&=1-\boldsymbol{n}_p^{\mathsf T}\boldsymbol{n}_q,
&d_{\mathrm{lat}}(p,q)&=1-\boldsymbol{z}_p^{\mathsf T}\boldsymbol{z}_q,\\
g_{pq}
&=\sqrt{\frac{d_{\mathrm{lat}}(p,q)}{d_{\mathrm{in}}(p,q)}}
=\frac{\|\boldsymbol{z}_p-\boldsymbol{z}_q\|_2}
{\|\boldsymbol{n}_p-\boldsymbol{n}_q\|_2},
&&d_{\mathrm{in}}(p,q)>0.
\end{aligned}
\tag{B.18}
$$

$g_{pq}>1$ は拡大、$g_{pq}<1$ は縮小を表す。これは有限な画素対の弦長比であり、微分による局所倍率ではない。入力距離が極小なら比が不安定になるため、元の二距離と併読する。

条件間の診断に用いる場合は、同じ画素対の距離を比較する。異なるモデルの潜在ベクトルを直接差し引かず、入力距離に対する応答の違いを見る。この比較が記述するのは固定した学習済み表現と入力の幾何の違いであり、学習途中の時間的な増減ではない。

<a id="svd-interpretation"></a>

### B.5.2 SVDによる入力差・潜在差・残差の対応

同じdecoderによる復元を $\widehat{\boldsymbol{x}}_p=W_{\mathrm{dec}}\boldsymbol{z}_p+\boldsymbol{b}_{\mathrm{dec}}$、残差を $\boldsymbol{e}_p=\boldsymbol{x}_p-\widehat{\boldsymbol{x}}_p$ とする。$W_{\mathrm{dec}}\in\mathbb{R}^{256\times16}$ のSVDを $U_{\mathrm{dec}}\Sigma_{\mathrm{dec}}V_{\mathrm{dec}}^{\mathsf T}$ とし、正の特異値 $\sigma_{\mathrm{dec},i}$ に対応する左右の単位特異ベクトルを $\boldsymbol{u}_{\mathrm{dec},i},\boldsymbol{v}_{\mathrm{dec},i}$ と書く。$\Delta$ を同じ順序で取った二画素の差とすると、

$$
\Delta\boldsymbol{x}
=W_{\mathrm{dec}}\Delta\boldsymbol{z}+\Delta\boldsymbol{e},
\qquad
\boldsymbol{v}_{\mathrm{dec},i}^{\mathsf T}\Delta\boldsymbol{z}
=\frac{
\boldsymbol{u}_{\mathrm{dec},i}^{\mathsf T}\Delta\boldsymbol{x}
-\boldsymbol{u}_{\mathrm{dec},i}^{\mathsf T}\Delta\boldsymbol{e}
}{\sigma_{\mathrm{dec},i}}
\quad(\sigma_{\mathrm{dec},i}>0).
\tag{B.19}
$$

共通biasは差で消え、左特異ベクトルへの射影から右側の式が得られる。入力の単位方向の尺度では、分子を $\rho$ で割った差に対する倍率は $\rho/\sigma_{\mathrm{dec},i}$ となる。ただし、小特異値だけで実際の拡大は決まらない。入力差と残差差の射影が打ち消し合えば、潜在差も小さくなる。

この対応は、潜在で大きい差がどの波長方向の観測差を伴うか、観測差がどれだけ残差へ残るかを記述するために使える。残差はdecoderの列空間に直交するとは限らず、二乗量を「保存された情報＋失われた情報」と加算する分解ではない。個々のMSEが小さくても、弱い画素間差に対する残差の比率まで小さいとはいえない。

Rankが不足する場合、decoderのkernel内の潜在差は復元から読めない。特異方向には符号・重複特異値での非一意性があり、異なるモデルの軸番号を同じ化学的意味へ対応づけない。

<a id="cluster-mean-geometry"></a>

### B.5.3 クラスタ平均と内部の広がり

非空の画素集合 $\mathcal{P}_k$ に、非負で和が1の重み $w_{kp}$ を与える。各画素をencodeした後、入力・潜在・残差を同じ重みで平均し、$\overline{\boldsymbol{x}}_k,\overline{\boldsymbol{z}}_k,\overline{\boldsymbol{e}}_k$ と書く。線形decoderでは、

$$
\overline{\boldsymbol{x}}_k
=W_{\mathrm{dec}}\overline{\boldsymbol{z}}_k
+\boldsymbol{b}_{\mathrm{dec}}+\overline{\boldsymbol{e}}_k,
\qquad
\Delta\overline{\boldsymbol{x}}
=W_{\mathrm{dec}}\Delta\overline{\boldsymbol{z}}
+\Delta\overline{\boldsymbol{e}}.
\tag{B.20}
$$

したがって、群間の平均スペクトル差にも式(B.19)を適用できる。一般に $F(\overline{\boldsymbol{x}}_k)\neq\overline{\boldsymbol{z}}_k$ なので、平均スペクトルをencoderへ入力して代用しない。この線形対応に使う平均潜在は単位化せず、既定の中央値による代表スペクトルとも区別する。

二集合から上記の重みで独立に画素を選ぶと、内積の双線形性より、

$$
\mathbb{E}_{p\in k,q\in l}[d_{\mathrm{in}}(p,q)]
=1-\frac{\overline{\boldsymbol{x}}_k^{\mathsf T}\overline{\boldsymbol{x}}_l}{\rho^2},
\qquad
\mathbb{E}_{p\in k,q\in l}[d_{\mathrm{lat}}(p,q)]
=1-\overline{\boldsymbol{z}}_k^{\mathsf T}\overline{\boldsymbol{z}}_l.
\tag{B.21}
$$

さらに、$a_k=\|\overline{\boldsymbol{z}}_k\|_2$ とし、$a_k,a_l>0$ の場合の平均方向を $\boldsymbol{c}^{\mathrm{emp}}_k=\overline{\boldsymbol{z}}_k/a_k$ とすると、

$$
\begin{aligned}
1-\overline{\boldsymbol{z}}_k^{\mathsf T}\overline{\boldsymbol{z}}_l
&=(1-a_ka_l)+a_ka_l\left(1-(\boldsymbol{c}^{\mathrm{emp}}_k)^{\mathsf T}\boldsymbol{c}^{\mathrm{emp}}_l\right),\\
\sum_{p\in\mathcal{P}_k}w_{kp}
\|\boldsymbol{z}_p-\overline{\boldsymbol{z}}_k\|_2^2
&=1-a_k^2.
\end{aligned}
\tag{B.22}
$$

入力側も $\overline{\boldsymbol{x}}_k/\rho$ で同様に計算できる。これにより、平均画素対距離の増加を、平均方向の分離と群内の拡散に分けて読む。平均差だけでは相殺される変動を、内部の広がりと画素対の診断で補う。

$\boldsymbol{c}^{\mathrm{emp}}_k$ は解析集合の平均方向であり、trainのKMeans中心とは限らない。平均がゼロなら方向は未定義となる。式(B.21)で $k=l$ とすると同一画素を選ぶ場合も含む。等重みの異なる二画素だけを対象にするときは、$n_k=|\mathcal{P}_k|>1$ として右辺を $n_k/(n_k-1)$ 倍する。

条件比較に用いる場合は、所属画素と重みを固定した共通集合を用いる。各モデル自身のクラスタによる集計は分割の記述に使えるが、群の選ばれ方も変わるため、中心間距離だけで表現の優越性を判定しない。共通分割の選択による基準モデルへの依存も明示する。

<a id="perturbation-response"></a>

### B.5.4 同一画素への摂動応答

固定画素への追加摂動を $\mathcal{A}$、全可視潜在を $\boldsymbol{z}_{p,\mathcal{A}}=F(\mathcal{A}(\boldsymbol{x}_p))$ とする。摂動についての平均を $\boldsymbol{\mu}_p=\mathbb{E}_{\mathcal{A}}[\boldsymbol{z}_{p,\mathcal{A}}]$、共分散を $\Gamma_p$ と書けば、

$$
2\mathbb{E}_{\mathcal{A}}\!\left[1-\boldsymbol{z}_{p,\mathcal{A}}^{\mathsf T}\boldsymbol{z}_p\right]
=\mathbb{E}_{\mathcal{A}}\|\boldsymbol{z}_{p,\mathcal{A}}-\boldsymbol{z}_p\|_2^2
=\operatorname{tr}(\Gamma_p)+\|\boldsymbol{\mu}_p-\boldsymbol{z}_p\|_2^2.
\tag{B.23}
$$

潜在を平均と偏差に分けると、偏差の期待値がゼロなので交差項が消える。$\operatorname{tr}$ は対角和である。摂動間のばらつきが小さくても、すべての摂動でcleanから同じ方向へずれる場合があるため、二項を分ける。有限の $D$ drawsで恒等式を照合する場合は、分母 $D$ の共分散を使う。

同じ摂動入力を各条件へ与えると、入力方向の変化、潜在のばらつきと平均のずれを対応づけられる。画素間の差と併読することは、一律の圧縮か、変動の種類に応じた応答かを検討する診断候補となる。復元を併読する場合のtargetは常に追加摂動前の $\boldsymbol{x}_p$ とする。潜在の移動量だけでは割当境界を横切るかは決まらず、label flipとは区別する。

<a id="masked-loss-variation"></a>

### B.5.5 Masked lossが測る潜在変動

学習課題との関係を見るため、ここだけはモデル・画素・不可視チャネル集合 $\mathcal{H}$ を固定し、追加摂動 $\mathcal{A}$ に期待値を取る。不可視チャネル数を $N_{\mathrm{hide}}>0$、decoderの対応する行を $W_{\mathcal{H}},\boldsymbol{b}_{\mathcal{H}}$、targetの対応成分を $\boldsymbol{x}_{p,\mathcal{H}}$ とする。このmaskで得た潜在 $\boldsymbol{z}_{p,\mathcal{H},\mathcal{A}}$ の平均・共分散を $\boldsymbol{\mu}_{p,\mathcal{H}},\Gamma_{p,\mathcal{H}}$ とすると、一画素のmasked MSEは、

$$
\mathbb{E}_{\mathcal{A}}[\mathcal{L}_{p,\mathcal{H},\mathcal{A}}]
=\frac{\|W_{\mathcal{H}}\boldsymbol{\mu}_{p,\mathcal{H}}
+\boldsymbol{b}_{\mathcal{H}}-\boldsymbol{x}_{p,\mathcal{H}}\|_2^2}
{N_{\mathrm{hide}}}
+\frac{\operatorname{tr}(W_{\mathcal{H}}\Gamma_{p,\mathcal{H}}W_{\mathcal{H}}^{\mathsf T})}
{N_{\mathrm{hide}}}.
\tag{B.24}
$$

式(B.23)と同じ平均・偏差の展開で得られる。第2項は、$W_{\mathcal{H}}$ の各右特異方向の潜在分散に、対応する特異値の二乗を掛けた和を $N_{\mathrm{hide}}$ で割った量である。Lossはdecoderを通した変動を測る一方、潜在のcosine幾何はその特異値で方向を重みづけない。小特異値やkernelの方向では、潜在が動いてもlossへ現れにくい。

これが、低い再構成lossだけでクラスタリングに適した変動を判断できない理由である。学習中はdecoderも変わるため、式(B.24)は特定方向の潜在分散が必ず減るという学習則ではない。実際に何が拡大・縮小されたかは、式(B.24)や既定のクラスタ評価指標だけからは結論づけられない。

固定maskでの条件付きの式を、mask間の不変性や全可視推論の保証へ拡張しない。Random mask自体もcorruptionであり、M00で追加摂動に関する分散がゼロでも、denoising作用の不在を意味しない。また、更新中のモデルで集計したepoch train lossを、固定モデルの式(B.24)へ代入しない。

<a id="numerical-geometry"></a>

## B.6 数値照合の条件

以上は単位normなどの制約を満たす理想式である。実装で照合する際は入力・潜在の実測norm、残差、平均と内積の計算誤差を確認する。零平均の方向、零入力距離の比、ゼロ特異値での除算は定義せず、極小距離・数値rankの扱いは診断前に固定する。

正規化の保護とFP16学習・FP32抽出の設定は[付録C](implementation_details.md)に従う。FP16の値の間隔をMSEの下限とは扱わない。固定重みでのAMP・FP32比較は別の診断候補であり、本節の恒等式の照合に自動的に追加するものではない。

---

## 執筆メモ・論点の出典

B.1〜B.4は第3章と[現行プロトコル](../../docs/design/experiment_protocol.md)の定義を補う。ChemoMAE v0.2.2のaugmentation・モデル・masked MSEと[呼出し側](../../src/wood_degradation_map/experiments/neural.py)は、既存の執筆時に読み取りで照合した。

B.5は2026-09-12の[会話](https://chatgpt.com/c/6aa41285-9758-83ee-9d0f-8ca3daa23ff6)に沿って、入力差・潜在差・残差の関係式へ改訂した。一般的な楕円体・PCA・出力制約の導出は削除した。2026-09-13の[必要性の見直し](../../docs/design/representation_geometry_diagnostics.md)で、対応する補助診断一式を現行の実施計画から外し、本節は数理的補足に位置づけた。診断は未実装・未実施であり、本節の掲載を実験・結果図表の追加理由にしない。学習条件・主評価は変更せず、A1等は追加しない。Lossの概数と過去の説明候補は[解釈メモ](../../docs/interpretation_notes.md#decoder-residual-discussion)に残す。
