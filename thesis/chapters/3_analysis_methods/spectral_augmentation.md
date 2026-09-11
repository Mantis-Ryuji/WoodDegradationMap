# 3.3 SNV制約を保つスペクトル摂動

## 3.3.1 摂動の目的と定義域

Masked reconstructionで用いる入力にTGNおよびFractional Shiftを加え、追加摂動前の観測を復元する課題を構成する。狙いは、不可視帯域の補完と追加摂動からの復元を通じて、状態差の探索に有用なスペクトル表現を学ぶことである。摂動の設計には、前節で定義したSNVスペクトルの平均ゼロ・一定normという制約を用いる。以下では二つの操作を定義し、学習時の組合せと、その設計が保証する範囲を整理する。

以下では画素添字を省略し、SNV入力を $\boldsymbol{x}\in\mathbb{R}^{C}$、チャネル数を $C=256$、そのnormを $\rho=\sqrt{C-1}$ とする。全成分が1のベクトルを $\boldsymbol{1}$ とし、平均を除く射影行列を

$$
P=I_C-\frac{\boldsymbol{1}\boldsymbol{1}^{\mathsf T}}{C}
\tag{3.5}
$$

と定義する。$I_C$ は $C$ 次元単位行列である。SNV入力は $P\boldsymbol{x}=\boldsymbol{x}$ と $\|\boldsymbol{x}\|_2=\rho$ を満たす。本節の式は、これらの制約が成り立ち、正規化の分母が非ゼロである場合の理想演算を表す。浮動小数演算における再中心化・再正規化と退化ケースの扱いは付録Bで補足する。

## 3.3.2 Tangent Gaussian Noise

TGNでは、Gaussian乱数から平均方向と入力の半径方向の成分を除き、残る接方向へ入力を回転させる。入力の単位方向を $\boldsymbol{n}=\boldsymbol{x}/\rho$ とする。方向候補の標準正規乱数ベクトルを $\boldsymbol{\epsilon}$、それを接空間へ射影した未正規化のベクトルを $\boldsymbol{q}_{\mathrm{raw}}$、その単位方向を $\boldsymbol{q}$ として、

$$
\begin{aligned}
\boldsymbol{\epsilon}&\sim\mathcal{N}(\boldsymbol{0},I_C),\\
\boldsymbol{q}_{\mathrm{raw}}
&=P\boldsymbol{\epsilon}
-\boldsymbol{n}\left(\boldsymbol{n}^{\mathsf T}P\boldsymbol{\epsilon}\right),\\
\boldsymbol{q}
&=\frac{\boldsymbol{q}_{\mathrm{raw}}}{\|\boldsymbol{q}_{\mathrm{raw}}\|_2}.
\end{aligned}
\tag{3.6}
$$

とする。$\mathcal{N}(\boldsymbol{0},I_C)$ は平均がゼロベクトル、共分散が単位行列の $C$ 次元正規分布を表す。$\|\boldsymbol{q}_{\mathrm{raw}}\|_2>0$ のとき、$\boldsymbol{q}$ は平均ゼロ、単位normであり、$\boldsymbol{n}$ と直交する。TGNによる変換を

$$
T_{\mathrm{TGN}}(\boldsymbol{x})
=
\rho\left(\cos\alpha\,\boldsymbol{n}
+\sin\alpha\,\boldsymbol{q}\right),
\qquad
\alpha\sim\mathcal{U}(0,\pi/36)
\tag{3.7}
$$

と定義する。$T_{\mathrm{TGN}}$ はTGNの変換、$\mathcal{U}$ は括弧内の区間の連続一様分布を表す。$\pi$ は円周率、$\alpha$ はradianで表した回転角であり、採用範囲は $0$–$5^\circ$ である。

二つの直交単位方向を用いるため、変換後も平均ゼロ・norm $\rho$ が保たれる。また、入力との角度は $\alpha$ となり、摂動強度を角度によって指定できる。制約保存の証明と、角度・弧長・残差normの関係は[付録B.2](../../appendices/mathematical_details.md#tgn-geometry)に示す。

TGNのGaussianは方向候補 $\boldsymbol{\epsilon}$ の生成分布を指す。最終的に加わる残差や回転角の分布がGaussianであることを意味せず、各チャネルへ独立なGaussian雑音を加算する操作とも異なる。

## 3.3.3 Fractional Shift

Fractional Shiftでは、共通波長gridのチャネルindexに沿ってスペクトルを移動させる。移動量 $\delta$ はチャネル単位で定め、整数に限らない位置の値を線形補間する。チャネル範囲外は端点値を延長し、反対側の端へ回り込む循環shiftは用いない。

この補間操作を $S_\delta$、補間後に平均を除いたスペクトルを $\boldsymbol{y}$ とする。平均除去とnormの復元を含むFractional Shiftの変換 $T_{\mathrm{FS}}$ は

$$
\boldsymbol{y}=P S_\delta(\boldsymbol{x}),
\qquad
T_{\mathrm{FS}}(\boldsymbol{x})
=
\rho\frac{\boldsymbol{y}}{\|\boldsymbol{y}\|_2},
\qquad
\delta\sim\mathcal{U}(-2,2)
\tag{3.8}
$$

と表せる。ここでも $\mathcal{U}$ は連続一様分布である。正の $\delta$ では、出力チャネルの添字 $j=0,\ldots,C-1$ が元の位置 $j-\delta$ を参照するため、補間操作は特徴を大きいチャネルindex側へ移す向きに対応する。補間と端点処理の具体式は[付録B.3](../../appendices/mathematical_details.md#fractional-shift)に示す。

補間だけでは画素内の平均やnormが変化し得るため、各操作後に再中心化と再正規化を行う。これにより、$\|\boldsymbol{y}\|_2>0$ の場合にはSNVの制約集合へ戻る。一方、入力と出力の角度変化は元のスペクトル形状に依存し、同じ $\delta$ を与えても一定にはならない。

ここでのshift軸は等間隔の波長gridであり、波数軸上の等間隔移動ではない。また、スペクトル軸への操作であるため、元画像の画素座標は変化させない。復元targetの波長位置も移動させず、追加摂動前の観測をそのまま用いる。

## 3.3.4 学習時の組合せと適用条件

学習時には、有効とした各操作を画素ごとに確率0.5で適用する。TGNとshiftを併用する条件では二つの適用をそれぞれ抽選し、操作順をmini-batchごとにランダム化する。したがって、併用条件でも、すべての画素に必ず両操作を加えるわけではない。

| 条件 | TGNの適用確率 | Shiftの適用確率 | 位置づけ |
| --- | ---: | ---: | --- |
| M00 | 0 | 0 | 追加摂動なしのMAE |
| M10 | 0.5 | 0 | TGNを用いるMAE |
| M01 | 0 | 0.5 | Shiftを用いるMAE |
| M11 | 0.5 | 0.5 | 両操作を用いる提案条件 |

各操作後には平均を0へ戻し、normをその操作直前の入力のnormへ戻す。実装は微小値 $10^{-8}$ によって数値上の除算を保護し、接方向や再投影候補のnormが十分でない場合には元の入力を保持する。入力が理想的なSNV制約を満たすとき、この処理は式(3.7)・(3.8)の半径 $\rho$ の保持に対応する。

通常のマップ作成には追加摂動を用いず、固定モデルに対する摂動安定性の評価では、同じ強度分布で対象操作の適用確率を1とする。学習時の適用確率と、操作を必ず加えて応答を調べる評価時の確率は役割が異なる。評価時の反復と共有規約は第4章で述べる。

## 3.3.5 採用強度の根拠と解釈の限界

採用した強度は、SNVスペクトル上で変化の程度を確認して固定した学習条件である。測定装置の誤差分布から推定した値や、CV指標を最適化して得た値ではない。

本設計が保持するのは平均・normに関する幾何的制約である。吸収帯の帰属や化学状態が保存されること、あるいは化学的に重要な違いが残ることを、制約保存だけから保証することはできない。本研究では、これらの摂動を用いた復元学習によって得られる表現とマップを比較し、その有用性と限界を検討する。

---

## 執筆メモ（本文外）

- **参照資料・照合先：** 採用強度と条件の正は[実験プロトコル](../../../docs/design/experiment_protocol.md)、式の説明は[既存の位置づけ](../../../docs/chemomae_positioning.md#spectral-augmentation)。原稿作成時にChemoMAE v0.2.2のSpectraAugmenterと[呼出し側](../../../src/wood_degradation_map/experiments/neural.py)を読み取りで照合した。
- **残る整備：** 原典・関連手法の引用を最終稿で整備する。
- **記号の規約：** モデルparameterには $\theta$、TGN角度には $\alpha$ を用い、同じ記号の兼用を避ける。
