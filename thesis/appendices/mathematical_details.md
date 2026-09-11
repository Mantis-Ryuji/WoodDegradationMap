# 付録B 数理的補足

本付録では、第3章の定義から導かれる性質を示す。ここでの等式は、有限な入力と非ゼロの正規化分母を仮定する理想演算についての記述であり、実験での性能や化学的妥当性を示すものではない。浮動小数演算上の保護はB.6で区別する。

論点は、入力の制約、学習課題、復元モデルの制約の順に積み上げる。

| 論点 | 導出する内容 | 対応節 |
| --- | --- | --- |
| 入力と摂動 | SNVが定める集合と、TGN・Fractional Shiftがその制約を保つ条件 | [B.1](#snv-geometry)〜[B.3](#fractional-shift) |
| 学習課題 | 不可視帯域の誤差が全帯域にわたる復元課題を作る仕組み | [B.4](#masked-objective) |
| 表現と復元 | 単位潜在・線形decoderの幾何、出力制約の成立条件、誤差とSVDの解釈 | [B.5](#latent-decoder) |
| 数値実装 | 理想式、数値的な保護、混合精度での確認を区別する条件 | [B.6](#numerical-geometry) |

記号の対応は[共通記号表](../notation.md)を参照する。各節でも、導出に必要な量と成立条件を初出時に示す。

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

## B.5 単位潜在と線形decoderの制約

本節では、SNV targetの制約集合、モデルが復元できる集合、実際のデータで使用される領域を区別する。まず表現と復元写像を定義し、次にすべての単位潜在でSNV制約を満たす条件を導く。その上で、個々の復元誤差や学習済みモデルから何を読み取れるかを整理する。

| 論理群 | 中心となる問い | 小節 |
| --- | --- | --- |
| [表現とPCA](#representation-and-pca) | 潜在、PCA score、クラスタリング入力はそれぞれ何を表すか | B.5.1〜B.5.4 |
| [Decoderの幾何と出力制約](#decoder-geometry-and-constraints) | 潜在距離は復元距離へどう写り、平均・normの保証には何が必要か | B.5.5〜B.5.7 |
| [再構成誤差](#reconstruction-diagnostics) | 誤差の成分と、norm不足の可能な原因をどう分けるか | B.5.8〜B.5.9 |
| [モデルの解釈](#model-interpretation) | SVDと小さなlossから、どこまで述べられるか | B.5.10〜B.5.11 |

<a id="representation-and-pca"></a>

### 表現の定義とPCAとの比較

#### B.5.1 潜在の自由度と復元範囲

画素添字 $p$ を省略し、潜在次元を $d_z=16$ とする。Encoderの正規化前出力 $\boldsymbol{h}\in\mathbb{R}^{d_z}$ が非ゼロのとき、L2正規化した潜在 $\boldsymbol{z}=\boldsymbol{h}/\|\boldsymbol{h}\|_2$ は
$\boldsymbol{z}\in\mathbb{S}^{15}
=\{\boldsymbol{z}\in\mathbb{R}^{16}:\|\boldsymbol{z}\|_2=1\}$
に属する。$\mathbb{S}^{15}$ は16次元空間内の単位球面を表す。16個の成分を持つが、単位normの制約により通常の連続自由度は15である。これは、平面内の円周が2成分で表せても、円周上の位置を指定する自由度は1であることに対応する。

$W_{\mathrm{dec}}\in\mathbb{R}^{C\times d_z}$ をdecoderの重み行列、$\boldsymbol{b}_{\mathrm{dec}}\in\mathbb{R}^{C}$ をbias、$\widehat{\boldsymbol{x}}$ を復元スペクトルとする。ここで $C=256$ は出力チャネル数である。$\widehat{\boldsymbol{x}}=W_{\mathrm{dec}}\boldsymbol{z}+\boldsymbol{b}_{\mathrm{dec}}$ より、

$$
\widehat{\boldsymbol{x}}
\in
\boldsymbol{b}_{\mathrm{dec}}+\operatorname{col}(W_{\mathrm{dec}}),
\qquad
\operatorname{rank}(W_{\mathrm{dec}})\leq16
\tag{B.18}
$$

である。$\operatorname{col}$ は行列の列空間、$\operatorname{rank}$ はその次元を表す。したがって、encoderを非線形にしても、decoderの復元可能範囲が任意の非線形曲面へ広がるわけではない。

さらに $W_{\mathrm{dec}}$ が列full rankであり、薄い特異値分解を
$W_{\mathrm{dec}}=U_{\mathrm{dec}}\Sigma_{\mathrm{dec}}V_{\mathrm{dec}}^{\mathsf T}$、
$\Sigma_{\mathrm{dec}}=\operatorname{diag}(\sigma_{\mathrm{dec},1},\ldots,\sigma_{\mathrm{dec},16})$、
$\sigma_{\mathrm{dec},i}>0$ とする。$U_{\mathrm{dec}}$ は左特異ベクトルを列に持つ $C\times d_z$ 行列、$V_{\mathrm{dec}}$ は右特異ベクトルを列に持つ $d_z\times d_z$ 直交行列、$\Sigma_{\mathrm{dec}}$ は正の特異値 $\sigma_{\mathrm{dec},i}$ を対角に並べた行列である。$\operatorname{diag}$ は対角行列を作る操作、ここでの $i=1,\ldots,d_z$ は特異値の添字を表す。回転後の潜在 $\boldsymbol{z}_{\mathrm{rot}}=V_{\mathrm{dec}}^{\mathsf T}\boldsymbol{z}$ も単位normなので、

$$
\sum_{i=1}^{d_z}
\left(
\frac{
[U_{\mathrm{dec}}^{\mathsf T}(\widehat{\boldsymbol{x}}-\boldsymbol{b}_{\mathrm{dec}})]_i
}{\sigma_{\mathrm{dec},i}}
\right)^2
=1
\tag{B.19}
$$

を満たす。これは16次元のアフィン部分空間内の楕円体表面の制約を表す。潜在 $\boldsymbol{z}$ 自体は単位球面上にあり、楕円体が現れるのはdecoder後の復元空間である。球面の次元が15であることと、そのアフィン包の次元が最大16であることは矛盾しない。重みが列full rankでない場合は像が退化し、この15次元の楕円体表面としての説明はそのまま適用できない。

固定したdecoderが全単位潜在から作れる集合は $\boldsymbol{b}_{\mathrm{dec}}+W_{\mathrm{dec}}\mathbb{S}^{15}$ である。Encoderは、その集合上のどの復元値を入力に対応づけるかを非線形に推定する。実際のencoder出力が単位球面全体を覆うとは限らないため、学習した観測の復元値はこの集合の一部にしか現れない場合がある。復元可能集合の幾何と、データが使用する範囲は別の論点である。

線形decoderの出力には、平均ゼロやnorm $\sqrt{255}$ への再投影を課していない。TargetがSNVであることだけで、再構成値がSNV制約を厳密に満たすとはいえない。

#### B.5.2 PCAとの対応範囲

第3.5節と同じく、画素 $p$ のSNVを $\boldsymbol{x}_p$、PCAのfit対象画素から求めた中心化平均を $\overline{\boldsymbol{x}}_{\mathrm{fit}}$、16本の直交主成分方向を列に持つ行列を $U_{\mathrm{PCA}}\in\mathbb{R}^{C\times16}$ とする。正規化前のPCA scoreを $\boldsymbol{\psi}_p$、そのscoreからの復元値を $\widehat{\boldsymbol{x}}_{\mathrm{PCA},p}$ とすると、

$$
\boldsymbol{\psi}_p
=
U_{\mathrm{PCA}}^{\mathsf T}(\boldsymbol{x}_p-\overline{\boldsymbol{x}}_{\mathrm{fit}}),
\qquad
\widehat{\boldsymbol{x}}_{\mathrm{PCA},p}
=
\overline{\boldsymbol{x}}_{\mathrm{fit}}+U_{\mathrm{PCA}}\boldsymbol{\psi}_p
\tag{B.20}
$$

である。本研究のモデルと同様、低次元座標から線形に復元するという役割を持つ。一方、本研究では座標推定が非線形であり、単位潜在、入力maskおよび追加摂動を用いる。Decoderの列にもPCAのような直交制約を課していない。

両者の違いを、decoderを固定して任意の許される座標を渡す場合について整理すると、次のようになる。表の復元集合全体が実際の試料で使われるという意味ではない。

| 観点 | PCA | 本研究の単位潜在モデル |
| --- | --- | --- |
| 座標の推定 | Fit平均で中心化した入力の線形射影 | 非線形encoderと単位化 |
| 再構成に用いる座標 | 大きさが自由な16成分のscore | Normが1の16成分潜在 |
| Decoderの列 | 互いに直交し、各列のnormは1 | 直交性・同一normを制約しない |
| 復元可能集合 | Fit平均を通る16次元アフィン部分空間 | 列full rankなら、16次元アフィン部分空間内の15次元楕円体表面 |
| クラスタリングに渡す段階 | Scoreを後からL2正規化 | 学習時から単位化された潜在を全可視で抽出 |

本研究のB1では、PCA scoreをさらにL2正規化してクラスタリングへ用いる。この後段の単位化と、PCAのfit・再構成の定義は区別する。出力次元を16にそろえた比較から、非線形性だけの効果を取り出したと解釈することはできない。

#### B.5.3 方向表現とcosine距離

前節の比較をクラスタリングの段階へ進める。第3.5節の単位クラスタリング入力を $\boldsymbol{\xi}_p$、第 $k$ クラスタの単位中心を $\boldsymbol{c}_k$ とする。$p$ は画素添字、$k$ はクラスタ添字である。両者に対し、

$$
\|\boldsymbol{\xi}_p-\boldsymbol{c}_k\|_2^2
=
2(1-\boldsymbol{\xi}_p^{\mathsf T}\boldsymbol{c}_k)
\tag{B.21}
$$

である。そのため、単位化済みの特徴と中心の間では、cosine不類似度を小さくする割当と二乗Euclidean距離を小さくする割当は対応する。ただし、中心の更新規則を単位球面へ制約しない通常のKMeansまで同一になるという意味ではない。

また、この恒等式は潜在や中心の幾何についての式である。入力と潜在の角度関係が保存されることや、SNVへ再投影していないdecoder出力に対するmasked MSEがcosine lossと同じになることを意味しない。

<a id="snv-coordinate-learning"></a>

#### B.5.4 SNVの方向情報と低次元座標の意味

ここまでの定義を用いて、SNV入力、encoderの単位潜在、PCA scoreのそれぞれで「方向」と「大きさ」が何を指すかを整理する。入力チャネル数を $C=256$、SNVスペクトルのnormを $\rho=\sqrt{C-1}$ とする。B.1の制約集合 $\mathcal{S}_{\mathrm{SNV}}$ では、全帯域の平均とnormがすでに固定されている。このため、同じnormのSNVスペクトルを区別する情報は、平均ゼロの部分空間内での方向にある。ここでいう方向は全チャネルの相対的な値の組合せを含み、スペクトル形状を一つの角度だけで表せるという意味ではない。

全帯域可視でのencoderと単位化は、この高次元の方向情報を、潜在次元 $d_z=16$ の単位球面 $\mathbb{S}^{15}$ 上の座標へ非線形に写す。入力側のSNV制約集合は254次元、潜在球面は15次元であるが、これは各制約集合の次元であり、実際の試料スペクトル分布の内在次元を推定した値ではない。この写像に完全な情報保存や可逆性を仮定しない。

したがって、潜在の単位化を「SNV入力に残っていたスペクトル全体の大きさを捨てる処理」と説明するのは適切でない。単位化が制限するのは、encoderが作る圧縮後の座標のnormを独立の情報量として使う自由度である。モデルは、定められた復元課題に必要な情報を潜在方向に符号化する。一方、吸収帯の相対的な深さなどの形状差はSNV後にも残り得るため、「大きさが一定」を「各チャネルの値の違いも消える」と解釈しない。

PCA scoreのnormが画素ごとに変わることも、このSNV制約とは矛盾しない。画素 $p$ のSNVを $\boldsymbol{x}_p$、fit画素の平均を $\overline{\boldsymbol{x}}_{\mathrm{fit}}$、主成分行列を $U_{\mathrm{PCA}}$、単位化前のscoreを $\boldsymbol{\psi}_p$ とする。式(B.20)と主成分列の直交性から、

$$
\|\boldsymbol{\psi}_p\|_2^2
=
(\boldsymbol{x}_p-\overline{\boldsymbol{x}}_{\mathrm{fit}})^{\mathsf T}
U_{\mathrm{PCA}}U_{\mathrm{PCA}}^{\mathsf T}
(\boldsymbol{x}_p-\overline{\boldsymbol{x}}_{\mathrm{fit}})
\tag{B.22}
$$

となる。右辺は、fit平均からのスペクトル形状のずれをPCA部分空間へ射影した成分の二乗normである。SNV前の反射率の絶対的な大きさではない。PCAによる復元値を $\widehat{\boldsymbol{x}}_{\mathrm{PCA},p}$ とすると、射影成分と残差は直交するため、

$$
\begin{aligned}
\|\boldsymbol{x}_p-\overline{\boldsymbol{x}}_{\mathrm{fit}}\|_2^2
&=
\|\boldsymbol{\psi}_p\|_2^2
+
\|\boldsymbol{x}_p-\widehat{\boldsymbol{x}}_{\mathrm{PCA},p}\|_2^2,\\
\|\boldsymbol{x}_p-\overline{\boldsymbol{x}}_{\mathrm{fit}}\|_2^2
&=
\rho^2+\|\overline{\boldsymbol{x}}_{\mathrm{fit}}\|_2^2
-2\boldsymbol{x}_p^{\mathsf T}\overline{\boldsymbol{x}}_{\mathrm{fit}}.
\end{aligned}
\tag{B.23}
$$

入力のnormが一定でも、fit平均を引いた後のnormや、その一部分であるPCA scoreのnormは一定とは限らない。B1のL2正規化では、クラスタリングに渡す際にこのscoreの大きさを取り除く。PCAのfit自体や、単位化前のscoreからの再構成まで単位球面に制約するものではない。

非線形encoderは、線形射影だけでは表せない入力と低次元座標の対応を学習できる。一方、線形decoderを用いるため、その座標から共通のアフィン写像でtargetを近似できることが復元課題の制約となる。入力分布が必ず分離しやすくなること、化学的な類似性に沿って配置されること、PCAより優れることは、この構成からは導けない。PCAでも射影によって不要な変動が減る場合があり、B1の最終的な単位化後の写像まで線形と呼ぶこともできない。比較結果は、表現学習の目的、mask、追加摂動、単位潜在およびdecoderの制約を含む方法全体の結果として扱う。

<a id="decoder-geometry-and-constraints"></a>

### Decoderの幾何と出力制約の成立条件

単位潜在の方向を用いることと、SNV targetを復元することは、同じ幾何を課す条件ではない。B.5.5では二つの復元値の距離を、B.5.6〜B.5.7ではすべての単位潜在に対する出力平均・normを扱う。後者の条件を、観測された一部の潜在で誤差が小さいという性質と区別する。

<a id="decoder-distance"></a>

#### B.5.5 Decoderが定める復元距離と潜在のcosine距離

同じ固定モデルで得た二つの画素の潜在を $\boldsymbol{z}_{p_1},\boldsymbol{z}_{p_2}$ とする。$p_1,p_2$ は比較する二画素の添字である。各潜在を同じdecoderへ渡した復元値を $\widehat{\boldsymbol{x}}_{p_1},\widehat{\boldsymbol{x}}_{p_2}$、潜在の差を $\Delta\boldsymbol{z}=\boldsymbol{z}_{p_1}-\boldsymbol{z}_{p_2}$ とする。Decoder重みの列間内積をまとめた行列を $G_{\mathrm{dec}}$ と定義すると、

$$
\begin{aligned}
G_{\mathrm{dec}}
&=W_{\mathrm{dec}}^{\mathsf T}W_{\mathrm{dec}},\\
\|\widehat{\boldsymbol{x}}_{p_1}-\widehat{\boldsymbol{x}}_{p_2}\|_2^2
&=\|W_{\mathrm{dec}}\Delta\boldsymbol{z}\|_2^2
=\Delta\boldsymbol{z}^{\mathsf T}G_{\mathrm{dec}}\Delta\boldsymbol{z}.
\end{aligned}
\tag{B.24}
$$

共通biasは差を取ると消える。$G_{\mathrm{dec}}$ は $d_z\times d_z$ の対称な半正定値行列であり、decoderが潜在の差を復元空間でどの方向にどれだけ拡大するかを表す。列full rankなら正定値となり、二乗距離を定める。Rankが不足すると、異なる潜在でも復元値が同じになる方向があり、厳密な距離ではなく半距離となる。B.5.1の特異値 $\sigma_{\mathrm{dec},i}$ は各特異方向の拡大率であり、化学成分の重要度を直接表す値ではない。

一方、潜在は単位normなので、

$$
\|\Delta\boldsymbol{z}\|_2^2
=
2\left(1-\boldsymbol{z}_{p_1}^{\mathsf T}\boldsymbol{z}_{p_2}\right).
\tag{B.25}
$$

式(B.25)は後段で用いるcosine幾何に対応するが、式(B.24)ではさらに $G_{\mathrm{dec}}$ による方向別の重みが入る。したがって、復元スペクトルの距離と潜在のcosine距離は一般には比例しない。$G_{\mathrm{dec}}$ が単位行列の正の定数倍となる特別な場合には比例するが、本研究ではその制約を課していない。

PCAでは、直交列を持つ $U_{\mathrm{PCA}}$ をdecoderに用いるため、単位化前の二つのscoreのEuclidean距離と、その復元値のEuclidean距離は一致する。ただし、PCAへ射影する前の入力間距離や、scoreを単位化した後のcosine距離まで保存されるという意味ではない。

ここで導出したのは、二つの復元値の全チャネルにわたる距離である。学習時のmasked MSEは各画素のtargetとの差を不可視チャネルだけで測り、その可視集合に応じて潜在自体も変わる。そのため、学習目的をそのまま潜在のcosine距離の最適化と見なすことはできない。また、線形decoderからの読み出しを課すことは、クラスタラベルや化学状態を線形に分類できることの保証ではない。

<a id="decoder-mean"></a>

#### B.5.6 平均ゼロのtargetが与える制約とmasked lossの違い

SNV targetを $\boldsymbol{x}$、任意の復元値を $\widehat{\boldsymbol{x}}$、平均を除く直交射影を $P$、$C$ 次元単位行列を $I_C$ とする。$P\boldsymbol{x}=\boldsymbol{x}$ であり、平均ゼロの成分と全チャネル共通の平均成分が直交するため、

$$
\|\boldsymbol{x}-\widehat{\boldsymbol{x}}\|_2^2
=
\|\boldsymbol{x}-P\widehat{\boldsymbol{x}}\|_2^2
+
\|(I_C-P)\widehat{\boldsymbol{x}}\|_2^2.
\tag{B.26}
$$

第2項は復元値の全チャネル共通の平均成分による誤差であり、非負である。したがって、**全チャネルの二乗誤差については**、復元値の平均を除いても誤差は増えない。全チャネルMSEでも両辺を同じチャネル数で割るだけなので、この性質は変わらない。

Decoderの出力を平均ゼロにする操作は、重みとbiasをそれぞれ $PW_{\mathrm{dec}},P\boldsymbol{b}_{\mathrm{dec}}$ に置き換えることで表せる。これは全チャネル再構成誤差の上で、平均成分を持たない出力を選べるという数学的性質である。実装にこの置換を追加したことや、最適化の結果として必ずこの形の重みが得られることを意味しない。

さらに、観測された潜在だけでなく、すべての単位潜在でdecoder出力を平均ゼロにしたい場合の必要十分条件は、

$$
\boldsymbol{1}^{\mathsf T}W_{\mathrm{dec}}=\boldsymbol{0}^{\mathsf T},
\qquad
\boldsymbol{1}^{\mathsf T}\boldsymbol{b}_{\mathrm{dec}}=0.
\tag{B.27}
$$

ここで $\boldsymbol{1}$ は全成分1の $C$ 次元ベクトル、右辺の $\boldsymbol{0}^{\mathsf T}$ は長さ $d_z$ のゼロ行ベクトルである。同じ単位潜在とその符号を反転した潜在で平均がともに0となる条件を加減すると、biasの平均と重みの各列の平均がそれぞれ0でなければならない。逆に式(B.27)を満たせば、任意の潜在で出力平均は0となる。実際にencoderが出力した潜在の一部で平均が0に近いことだけから、この行列条件を推定することはできない。

主比較で用いるmasked MSEでは、式(B.26)の直交分解をそのまま適用できない。不可視チャネルだけに内積を制限すると、全チャネルで平均ゼロの成分と平均成分が直交するとは限らないためである。例えば、不可視チャネルの予測がすべてtargetと一致していても、可視チャネルの予測誤差によって復元値の全チャネル平均が0でなくなる場合がある。その平均を全チャネルから引くと、不可視チャネルに新たな誤差が生じる。これは、出力の再中心化がmasked lossを必ず減らすわけではないことを示す。

Maskを繰り返し抽選しても、潜在と復元値が可視集合に依存するため、全可視出力の全チャネルMSEへ置き換えられるわけではない（B.4）。SNV targetの使用は出力平均が近づく可能性を考える動機になるが、平均ゼロの保証は、構造上の制約または別途の検証が必要な命題として区別する。

<a id="decoder-norm"></a>

#### B.5.7 すべての単位潜在から一定normを出力する条件

SNVのtarget normを $\rho=\sqrt{C-1}$、任意の単位潜在を $\boldsymbol{z}$ とする。一般のbias付きdecoderの出力normは、式(B.24)の $G_{\mathrm{dec}}$ を用いて、

$$
\|W_{\mathrm{dec}}\boldsymbol{z}+\boldsymbol{b}_{\mathrm{dec}}\|_2^2
=
\boldsymbol{z}^{\mathsf T}G_{\mathrm{dec}}\boldsymbol{z}
+
2\boldsymbol{z}^{\mathsf T}W_{\mathrm{dec}}^{\mathsf T}\boldsymbol{b}_{\mathrm{dec}}
+
\|\boldsymbol{b}_{\mathrm{dec}}\|_2^2.
\tag{B.28}
$$

これを**すべての**単位潜在で $\rho^2$ に一致させるための必要十分条件は、

$$
\begin{aligned}
W_{\mathrm{dec}}^{\mathsf T}\boldsymbol{b}_{\mathrm{dec}}
&=\boldsymbol{0},\\
G_{\mathrm{dec}}
&=\left(\rho^2-\|\boldsymbol{b}_{\mathrm{dec}}\|_2^2\right)I_{d_z},\\
\rho^2-\|\boldsymbol{b}_{\mathrm{dec}}\|_2^2&\geq0.
\end{aligned}
\tag{B.29}
$$

ここで $I_{d_z}$ は潜在次元の単位行列、$\boldsymbol{0}$ は長さ $d_z$ のゼロベクトルである。式(B.28)を $\boldsymbol{z}$ と $-\boldsymbol{z}$ で比較すると、一次項が常に0になることが必要である。一次項を除いた二次形式がすべての単位方向で同じ値を取るには、対称行列 $G_{\mathrm{dec}}$ がその値を対角に持つ単位行列の定数倍である必要がある。逆に式(B.29)を代入すれば、どの単位潜在でも出力normは $\rho$ となる。

特にbiasをゼロと仮定した場合は、$W_{\mathrm{dec}}^{\mathsf T}W_{\mathrm{dec}}=\rho^2I_{d_z}$ となり、decoderの列は互いに直交し、各列のnormとすべての特異値は $\rho$ になる。このとき単位潜在の像は半径 $\rho$ の球面である。Biasがある一般の場合には、biasが列空間と直交し、列空間内の球面の半径とbiasのnormの二乗和が $\rho^2$ になる。半径が0となる場合は重みがゼロとなり、出力はbias一点に退化する。

式(B.29)は、潜在球面全体について一定normを要求したときの条件である。実際のencoderが使用する潜在はその一部であり、有限の学習データを近似的に復元する課題は、この全方向の条件と同じではない。加えて、主比較ではmasked lossを用い、decoder出力をSNVへ再投影していない。したがって、「SNV targetで学習すればdecoderの特異値がそろう」「楕円体が球面に近づく」といった挙動を、未確認の学習結果として記述しない。

平均ゼロと一定normをすべての単位潜在で同時に満たすには、式(B.27)と式(B.29)の両方が必要になる。これらは本研究の実装に追加した制約ではなく、既存の線形decoderとSNV targetの関係を理解するための条件付きの結果である。

<a id="reconstruction-diagnostics"></a>

### 再構成誤差の分解とnorm不足の解釈

ここからは、復元値とtargetの間に生じる誤差を扱う。B.5.8の分解は、任意の復元値について誤差の所在を表す恒等式である。B.5.9の条件付き平均は、出力を制約しない全チャネルMSEの下で、normが縮む理由を説明する理想的な予測である。誤差の記述と原因の解釈を分けて読む。

<a id="reconstruction-error-components"></a>

#### B.5.8 再構成誤差の平均・norm・方向への分解

一画素のSNV targetを $\boldsymbol{x}$、復元値を $\widehat{\boldsymbol{x}}$ とし、画素添字を省略する。$C$ はチャネル数、$\rho=\sqrt{C-1}$ はtargetのnorm、$P$ は全チャネルの平均を除く射影である。復元値の平均を $\mu_{\mathrm{rec}}$、平均を除いた復元値のnormを $\rho_{\mathrm{rec}}$ と定義する。$\rho_{\mathrm{rec}}>0$ の場合、targetと中心化した復元値の角度を $\varphi$ とすると、

$$
\mu_{\mathrm{rec}}=\frac{\boldsymbol{1}^{\mathsf T}\widehat{\boldsymbol{x}}}{C},
\qquad
\rho_{\mathrm{rec}}=\|P\widehat{\boldsymbol{x}}\|_2,
\qquad
\cos\varphi=
\frac{\boldsymbol{x}^{\mathsf T}P\widehat{\boldsymbol{x}}}
{\rho\rho_{\mathrm{rec}}}.
\tag{B.30}
$$

$\boldsymbol{1}$ は全成分1の $C$ 次元ベクトルである。$\mu_{\mathrm{rec}}$ はSNV前の反射率平均 $\mu$ とは異なり、$\rho_{\mathrm{rec}}$ は中心化前の復元値のnormではない。式(B.26)の中心化成分を内積で展開すると、

$$
\begin{aligned}
\|\boldsymbol{x}-\widehat{\boldsymbol{x}}\|_2^2
&=C\mu_{\mathrm{rec}}^2+
\rho^2+\rho_{\mathrm{rec}}^2
-2\rho\rho_{\mathrm{rec}}\cos\varphi\\
&=\underbrace{C\mu_{\mathrm{rec}}^2}_{\text{平均のずれ}}
+\underbrace{(\rho_{\mathrm{rec}}-\rho)^2}_{\text{normのずれ}}
+\underbrace{2\rho\rho_{\mathrm{rec}}(1-\cos\varphi)}_{\text{方向のずれ}}.
\end{aligned}
\tag{B.31}
$$

全チャネルMSEへの寄与は、各項を $C$ で割った量となる。3項は非負であり、大きな成分同士が相殺して総誤差だけ小さくなることはない。

$\rho_{\mathrm{rec}}=0$ の場合は角度を定義せず、方向項を $2(\rho\rho_{\mathrm{rec}}-\boldsymbol{x}^{\mathsf T}P\widehat{\boldsymbol{x}})=0$ として扱う。この場合、中心化後の誤差 $\rho^2$ はnorm項に入る。方向項は角度だけでなく $\rho_{\mathrm{rec}}$ にも依存するため、形状の診断では角度が定義できるかとnormを併読する。

| 成分 | 直接表すもの | 解釈するときの注意 |
| --- | --- | --- |
| 平均 | 平均0のtargetに対する全チャネル共通の上下のずれ | SNV前の強度や散乱の大きさを直接表さない |
| Norm | 中心化した再構成の縮み・膨らみ | $\rho_{\mathrm{rec}}-\rho$ の符号を残す。形の平均化やモデル制約、数値演算など複数の原因があり得る |
| 方向 | Targetと再構成の波長間の相対的な形の違い | 特定の吸収帯の取りこぼしか、測定変動かは、この集計値だけでは識別できない |

この分解は全チャネルの診断であり、不可視チャネルだけで計算する学習時のmasked MSEの内訳ではない。可視集合によって復元値も変わるため、maskを平均してもその違いは解消しない。

個々の成分と波長ごとの符号付き残差を画素位置へ配置すれば、特定の帯域や場所で取りこぼしが繰り返されるかを探索できる。ただし、成分の大きさは誤差の原因を一意に定めず、化学状態や劣化の指標としての対応を確認した結果もまだない。Norm不足の可能な説明の一つを次節に示す。

<a id="conditional-mean-reconstruction"></a>

#### B.5.9 一定normのtargetと条件付き平均による縮み

出力normの不足は、球面制約からのずれという側面に加え、入力からtargetを一意に決められない場合の平均化としても解釈できる。ただし、以下は出力集合を制約しない全チャネルMSEについての理想的な議論である。

モデルへ与える可視帯域の値・位置などの観測情報を $\mathcal{O}$、その情報のもとでのSNV targetの条件付き平均を $\boldsymbol{m}(\mathcal{O})$ とする。ここでは $\boldsymbol{x}$ を観測情報に対応して複数の値を取り得るtargetとし、$\mathbb{E}[\cdot\mid\mathcal{O}]$ は観測情報を固定した条件付き期待値を表す。どのtargetも平均0・norm $\rho$ なら、

$$
\boldsymbol{m}(\mathcal{O})
=\mathbb{E}[\boldsymbol{x}\mid\mathcal{O}],
\qquad
\boldsymbol{1}^{\mathsf T}\boldsymbol{m}(\mathcal{O})=0,
\qquad
\|\boldsymbol{m}(\mathcal{O})\|_2\leq\rho.
\tag{B.32}
$$

全チャネルの期待二乗誤差はこの条件付き平均で最小になる。実際、別の予測と条件付き平均の差を用いて誤差を展開すると、targetと条件付き平均の差の条件付き期待値が0なので交差項が消え、予測を条件付き平均からずらした距離の二乗だけ誤差が増える。また、

$$
\mathbb{E}\!\left[
\|\boldsymbol{x}-\boldsymbol{m}(\mathcal{O})\|_2^2
\mid\mathcal{O}
\right]
=\rho^2-\|\boldsymbol{m}(\mathcal{O})\|_2^2.
\tag{B.33}
$$

これは条件付き平均の周りのtargetのばらつきと、条件付き平均のnormの不足を結ぶ式である。異なる方向の候補を平均すると球面の内側へ入るため、targetが球面上にあっても、期待MSEに最適な予測まで球面上にある必要はない。平均を除く操作が全チャネル誤差を増やさないこと（B.5.6）と、normを強制的に $\rho$ へ戻す操作の効果は異なる。後者は条件付き平均から予測を動かし、期待誤差を増やす場合がある。

本モデルには単位潜在とアフィンdecoderの制約があり、MAE条件では不可視チャネルだけを評価する。固定maskに対する制約なしのmasked MSEでは、不可視成分の条件付き平均が最適となるが、可視成分の予測はその損失から定まらない。そのため、実際の復元値全体を式(B.32)の条件付き平均と同一視したり、実測のnorm不足を式(B.33)でそのまま不確実性へ換算したりしない。ここでのclean targetも測定ノイズのない真値ではなく、追加摂動前の観測である。

<a id="model-interpretation"></a>

### 学習済みモデルから読み取れることと解釈の限界

前節までの式を用いると、decoderの拡大率、データが使用する潜在座標、targetに対する誤差を別々に記述できる。以下では、それらを結びつけてモデルを読む際の範囲を整理する。学習済み重みのSVDや追加の残差診断を実施した結果を示す節ではない。

<a id="svd-interpretation"></a>

#### B.5.10 SVDで復元方向・拡大率・使用範囲を分けて読む

B.5.1の薄いSVDにおける $U_{\mathrm{dec}}$ の第 $i$ 列を $\boldsymbol{u}_{\mathrm{dec},i}$ とする。画素 $p$ の潜在を右特異ベクトルの座標系へ回転したものを $\boldsymbol{z}_{\mathrm{rot},p}=V_{\mathrm{dec}}^{\mathsf T}\boldsymbol{z}_p$、その第 $i$ 成分を $z_{\mathrm{rot},p,i}$ とすると、

$$
\widehat{\boldsymbol{x}}_p
=\boldsymbol{b}_{\mathrm{dec}}
+\sum_{i=1}^{d_z}
\sigma_{\mathrm{dec},i}\,
z_{\mathrm{rot},p,i}\,
\boldsymbol{u}_{\mathrm{dec},i}.
\tag{B.34}
$$

$d_z=16$ は潜在次元、$\sigma_{\mathrm{dec},i}$ は第 $i$ 特異値である。$V_{\mathrm{dec}}^{\mathsf T}$ は潜在を拡大・縮小する軸へ向け直し、$\Sigma_{\mathrm{dec}}$ は各軸を伸縮し、$U_{\mathrm{dec}}$ はその成分を256チャネルの復元方向へ配置する。Biasは共通の平行移動を与える。この見方では、SVDの三つの要素に異なる役割がある。

| 読む対象 | 調べられること |
| --- | --- |
| 特異値 $\sigma_{\mathrm{dec},i}$ | その潜在方向の変化を復元空間へ伝える拡大率 |
| 左特異ベクトル $\boldsymbol{u}_{\mathrm{dec},i}$ | その方向に動いたとき、どの波長がどの符号で変わるか |
| 回転後の潜在 $z_{\mathrm{rot},p,i}$ | 実際の画素がその方向をどの範囲で使用しているか、試料表面でどこに現れるか |

特異値だけでは、実際のデータの変動量は決まらない。同じ固定モデルで、明示した画素集合上の分散を $\operatorname{Var}$ と書くと、左特異ベクトルが直交するため、

$$
\operatorname{Var}\!\left(
\left[U_{\mathrm{dec}}^{\mathsf T}
(\widehat{\boldsymbol{x}}_p-\boldsymbol{b}_{\mathrm{dec}})\right]_i
\right)
=\sigma_{\mathrm{dec},i}^2\,
\operatorname{Var}(z_{\mathrm{rot},p,i}).
\tag{B.35}
$$

したがって、拡大率が大きくても、その座標がほぼ使われなければ復元の変動は小さい。逆に、小さな特異値に対応する潜在の変化は復元値に現れにくいが、後段のcosine距離ではその軸を特異値で弱めていない（B.5.5）。低い再構成誤差だけから、潜在の方向が摂動に対して安定していることは導けない。

左特異ベクトルは復元の変動方向であり、純粋な化学成分のスペクトルとは限らない。特異ベクトルの符号は任意であり、重複した特異値に対応する軸も一意には定まらない。近い特異値の個別軸や異なる反復間の軸を、整列や部分空間としての比較なしに同じ化学的意味へ対応づけない。特異値・波長方向・画素分布を併読することは探索的な解釈の候補であり、本稿では学習済み重みからこれらを計算した結果は報告していない。

<a id="small-reconstruction-loss"></a>

#### B.5.11 小さな再構成誤差と球面・楕円体の関係

SNVの制約集合は254次元の球面だが、観測されたスペクトルがその全域を埋めるわけではない。全チャネル再構成誤差が小さければ、使用された復元点で平均・norm・方向のずれが合計として小さいことが式(B.31)から分かる。これは、実際のtarget集合がdecoderの使用された領域でよく近似できることを示し得る。潜在球面全体で式(B.27)・(B.29)が近似的に成立することや、すべての特異値がそろうことまでは意味しない。入力データの内在次元や可逆性の証明でもない。

SNV後には $\|\boldsymbol{x}\|_2^2/C=(C-1)/C$ が成り立つが、これは画素内の二乗値の尺度であり、画素間のスペクトル変動の尺度ではない。共通の大きな形を復元しても、試料や状態を区別する小さな差を取りこぼす可能性がある。低いMSEを、化学的に重要な変動の保存率やクラスタリング品質へ直接読み替えない。

A0のtrain lossは全帯域、M11は追加摂動とmaskの抽選のもとで不可視帯域を評価するため、数値をそのまま同一課題の優劣として比較できない。また、学習履歴はparameterを更新しながら得たbatch lossのepoch内平均であり、一つの固定モデルによる再評価値ではない。

ここまでの解釈を確かめる候補には、未学習試料で入力条件をそろえた全チャネル診断、学習画素の平均スペクトルだけを返す参照との比較、波長別・空間別の残差確認がある。これらは現行の主評価へ追加したものではない。候補の位置づけは[解釈メモ](../../docs/interpretation_notes.md#decoder-residual-discussion)に、混合精度の影響はB.6と付録Cに整理する。

<a id="numerical-geometry"></a>

## B.6 理想式と数値実装の区別

### B.6.1 正規化の保護と制約の近似

本研究の摂動実装では、各操作後に画素内平均を除き、その操作の直前の入力normへ戻す。理想的なSNV入力ではこのnormが $\sqrt{255}$ であるが、実装では定数を上書きするのではなく、入力ごとのnormを参照する。

TGNの接方向normおよび再投影候補のnormに $10^{-8}$ の保護を用いる。接方向がこの閾値以下ならその方向への回転を行わず、再正規化候補が閾値以下なら参照入力を保持する。有限精度での再中心化・再正規化は理想式の制約を近似するもので、bit単位の厳密な等式は仮定しない。

Encoderの潜在正規化には別の微小値 $10^{-12}$ を用いる。表現抽出では正規化前の非有限値、ゼロnormおよび保護値を下回るnormを検査する。正常な非ゼロ領域について定義した本付録の式と、数値的に不適切な出力を許容する規則とを混同しない。

### B.6.2 混合精度とlossの尺度

学習はFP16 autocastを用いる混合精度であり、全演算をFP16で行う方式ではない。重み・target・追加摂動の計算と、復元MSEの演算経路は[付録C.3.1](implementation_details.md#training-precision)に記す。FP16の1付近での隣接する表現可能値の間隔は $2^{-10}\simeq9.77\times10^{-4}$ だが、これは値の間隔であってMSEの下限ではない。丸めに由来する値の誤差と、その二乗・平均、さらにforward・勾配・最適化を通した影響を区別する。小さなtrain lossだけから、丸め誤差が支配的とも無視できるとも断定しない。

### B.6.3 数値検証でそろえる条件

精度の影響を検証する場合は、二つの確認を分ける。第一に、同じ固定重み・同じ入力・同じ可視集合でAMPとFP32のforwardを比較し、出力差とtargetへの誤差を別々に集計する。AMP出力を計算後にFP32へcastするだけでは、FP32でforwardをやり直した結果にはならない。

第二に、式(B.31)の確認では、FP32またはFP64で平均・norm・内積を集計し、丸め後のtargetの平均・normも確認する。Targetが理想制約からずれる場合は、両者の実測平均の差と、中心化したtargetの実測normを用いた同じ直交分解で照合する。有限精度で計算された潜在・復元値が、理想的な球面・楕円体表面へ厳密に乗るとは仮定しない。これらの精度比較は未実施である。

---

## 執筆メモ・論点の出典

数学的な根拠は、第3章と[現行プロトコル](../../docs/design/experiment_protocol.md)に対応する定義、および本付録の導出に置く。使用中のChemoMAE v0.2.2のaugmentation・モデル・masked MSEと[呼出し側](../../src/wood_degradation_map/experiments/neural.py)は、既存の執筆時に読み取りで照合した。関連する説明は[ChemoMAEの位置づけ](../../docs/chemomae_positioning.md)を参照する。標準手法の原典と本文の引用は最終稿で整理する。

会話は論点の整理に用いた執筆補助資料である。導出や実験結果の根拠と区別し、取り込んだ内容を次に記録する。

| 執筆補助資料 | 整理した論点 | 原稿への反映 |
| --- | --- | --- |
| 2026-09-12の[会話「内容の意味解説」](https://chatgpt.com/c/6aa41285-9758-83ee-9d0f-8ca3daa23ff6) | SNVの方向情報、PCA score、decoderの距離・平均・norm | B.5の説明とB.5.4〜B.5.7。SNVのnormは標本標準偏差に合わせて $\sqrt{C-1}$ とし、PCAのfit平均中心化、decoderのbias、全チャネルMSEとmasked MSEの違いを反映 |
| 同日の議論「理論的背景もなかなか面白くなってきたね。君は何が特に面白いと思う？」以降 | SVD、SNV targetと復元集合、残差、train loss、FP16 | B.5.8〜B.5.11とB.6。Lossの概数と探索的な確認候補は[解釈メモ](../../docs/interpretation_notes.md#decoder-residual-discussion)に保持 |

これらの補足によって、新しい制約や評価実験は追加していない。球面化・化学的な分離・性能改善を実験結果として採用せず、学習済み重みの数値検証も実施していない。
