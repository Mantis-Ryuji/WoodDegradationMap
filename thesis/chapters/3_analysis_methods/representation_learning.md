# 3.4 Masked denoisingによる表現学習

本節では、前節の摂動を受けた入力から追加摂動前のSNVを復元するモデルを定義する。波長軸のpatch化とmask、単一の単位潜在を介する再構成、損失関数、学習後の全帯域可視抽出の順に述べる。学習で復元する対象と、クラスタリングへ渡す表現を区別する。

## 3.4.1 スペクトルpatchと可視帯域

出力チャネル数を $C=256$、画素添字を $p$、その画素の追加摂動前のSNVを $\boldsymbol{x}_p$ とする。この入力スペクトルを、連続16チャネルからなる16個のpatchへ分割する。Patch数を $N_{\mathrm{patch}}=16$、patch幅を $C_{\mathrm{patch}}=C/N_{\mathrm{patch}}=16$ とする。画素 $p$ の追加摂動後の入力を $\widetilde{\boldsymbol{x}}_p=\mathcal{A}_p(\boldsymbol{x}_p)$ とすると、第 $a$ patchは

$$
\widetilde{\boldsymbol{x}}_p^{(a)}
=
\left(
\widetilde{x}_{p,C_{\mathrm{patch}}a},\ldots,
\widetilde{x}_{p,C_{\mathrm{patch}}a+C_{\mathrm{patch}}-1}
\right)^{\mathsf T},
\qquad a=0,\ldots,N_{\mathrm{patch}}-1
\tag{3.9}
$$

である。$a$ はpatchの添字、$\widetilde{\boldsymbol{x}}_p^{(a)}$ はそのpatchのスペクトル、$\widetilde{x}_{p,j}$ は追加摂動後の全スペクトルの第 $j$ 成分を表す。$\mathcal{A}_p$ は、前節で定義した適用抽選・順序・強度を含む入力変換を表す。追加摂動なしの条件では恒等写像となる。

主比較のMAE条件では、各画素について16 patch中8個をランダムに選び、patch全体を不可視にする。同じpatch内で可視チャネルと不可視チャネルを混在させない。可視patch集合を $\mathcal{V}_p$、不可視patchに含まれるチャネル集合を $\mathcal{H}_p$ とする。Maskは画素ごと・学習stepごとに抽選するため、常に同じ波長帯を隠すものではない。

各可視patchは、共通の線形写像で幅256のtokenへ変換し、元のpatch位置に対応する学習可能な位置埋め込みを加える。埋め込み幅を $d_{\mathrm{emb}}=256$ とすると、

$$
\boldsymbol{t}_{p,a}
=
E_{\mathrm{patch}}\widetilde{\boldsymbol{x}}_p^{(a)}
+\boldsymbol{b}_{\mathrm{patch}}+\boldsymbol{\pi}_a,
\qquad
E_{\mathrm{patch}}\in\mathbb{R}^{d_{\mathrm{emb}}\times C_{\mathrm{patch}}},
\quad
\boldsymbol{b}_{\mathrm{patch}},\boldsymbol{\pi}_a\in\mathbb{R}^{d_{\mathrm{emb}}}
\tag{3.10}
$$

と表せる。$\boldsymbol{t}_{p,a}$ は画素 $p$ のpatch $a$ の入力token、$E_{\mathrm{patch}}$ と $\boldsymbol{b}_{\mathrm{patch}}$ は全patchで共有する線形写像の重み行列とbias、$\boldsymbol{\pi}_a$ はpatch位置 $a$ の学習可能な位置埋め込みである。これに、スペクトル全体の情報を集約する学習可能なCLS tokenと、その位置埋め込みを加えてencoderへ入力する。不可視patchのtokenはattentionの入力から除く。主比較の50% maskでは、可視8 tokenとCLSの計9 tokenを用いる。

ここでのpatchは波長軸上の連続チャネルであり、画像の空間領域ではない。可視tokenの抽出後も元の波長位置に対応する埋め込みを使うため、残ったtokenの順番だけで位置情報を付け直すことはしない。

## 3.4.2 単一の単位潜在と線形再構成

### Encoderとdecoderの構成

Encoderには、埋め込み幅256、8層、8 head、feed-forward幅1024のTransformerを用いる。ActivationはGELU、LayerNormは各blockの前に置くpre-norm構成、dropoutは0とする。最終CLS出力を線形射影して16次元へ変換し、L2正規化したベクトルを潜在表現とする。潜在次元を $d_z=16$ とする。CLSの16次元射影までを含む写像を $f_\theta$、正規化前の出力を $\boldsymbol{h}_p$、単位化後の潜在を $\boldsymbol{z}_p$ として、

$$
\boldsymbol{h}_p=f_\theta(\widetilde{\boldsymbol{x}}_p;\mathcal{V}_p),
\qquad
\boldsymbol{z}_p=
\frac{\boldsymbol{h}_p}{\|\boldsymbol{h}_p\|_2},
\qquad
\boldsymbol{z}_p\in\mathbb{R}^{d_z}
\tag{3.11}
$$

と書く。$\theta$ はencoderの学習parameterを表す。式(3.11)は非ゼロnormの通常の場合であり、実装の微小値による保護と表現抽出時の検査は付録Cに示す。

Decoderはbias付きの線形1層とし、単一の潜在ベクトルから全スペクトルを復元する。復元値を $\widehat{\boldsymbol{x}}_p$、decoderの学習可能な重み行列を $W_{\mathrm{dec}}$、biasを $\boldsymbol{b}_{\mathrm{dec}}$ として、

$$
\widehat{\boldsymbol{x}}_p
=
W_{\mathrm{dec}}\boldsymbol{z}_p+\boldsymbol{b}_{\mathrm{dec}},
\qquad
W_{\mathrm{dec}}\in\mathbb{R}^{C\times d_z},
\quad
\boldsymbol{b}_{\mathrm{dec}}\in\mathbb{R}^{C}.
\tag{3.12}
$$

とする。Decoderへは、patchごとのencoder出力や入力からのskip connectionを渡さない。また、decoder用のmask tokenを用いるTransformer decoderではない。入力由来の情報はすべて16次元のbottleneckを通じて復元に用いられる。

### 単位潜在と線形decoderの役割

SNV入力では、全帯域の平均とnormがすでに固定され、画素間の違いは平均ゼロの部分空間内での方向として表される。本モデルはその高次元のスペクトル形状を、16次元空間内の単位球面 $\mathbb{S}^{15}$ 上の潜在方向へ非線形に写す。潜在の単位化は、圧縮後の座標でもnormを独立の情報量として用いないという設計である。SNV入力の時点で全帯域のnormは固定されているため、潜在の単位化を入力の絶対的な大きさの除去と同一視しない。入力の角度関係や情報がすべて保存されることも仮定しない。

線形decoderを用いることで、encoderは定められた復元課題を共通のアフィン写像で解けるように潜在座標を推定する。潜在自体は単位球面上にあり、decoderの重みが列full rankの場合、その復元値は高々16次元のアフィン部分空間内にある15次元の楕円体表面に制約される。楕円体は復元空間に現れるものであり、潜在が楕円体上にあるという意味ではない。これが非線形encoderによる座標推定と、線形decoderによる復元を組み合わせる設計上の意味である。

SNVの球面とdecoderの楕円体は、それぞれtargetの制約集合とモデルの復元可能集合である。観測スペクトルや使用される潜在がそれぞれの集合全体を覆う必要はなく、両者の違いだけから再構成が困難だとはいえない。全帯域誤差を平均・norm・方向へ分けた説明は[付録B.5.8](../../appendices/mathematical_details.md#reconstruction-error-components)、SVDによる復元方向・拡大率・潜在座標の読み方は[付録B.5.10](../../appendices/mathematical_details.md#svd-interpretation)に示す。

この構成は、クラスタが必ず分離することや化学的な類似性が学習されることを保証しない。また、SNV targetを用いるだけでは、decoder出力の平均ゼロ・一定normや、重みの列の直交性は保証されない。潜在の自由度・復元範囲、PCAとの比較、decoderが定める距離、および出力制約の成立条件は[付録B.5](../../appendices/mathematical_details.md#latent-decoder)に示す。

## 3.4.3 復元targetと損失関数

### MAE条件の損失とmaskの役割

復元targetには、追加摂動前のSNVスペクトル $\boldsymbol{x}_p$ を用いる。Shiftを含む条件でもtarget自体の波長位置は変更しない。Mini-batch内の画素集合を $\mathcal{P}_{\mathrm{batch}}$、その画素数を $N_{\mathrm{batch}}$ とする。$j$ はチャネル添字、$|\mathcal{H}_p|$ は画素 $p$ の不可視チャネル数である。$\widehat{x}_{p,j}$ と $x_{p,j}$ はそれぞれ復元値とtargetの第 $j$ 成分を表す。損失 $\mathcal{L}_{\mathrm{masked}}$ は不可視チャネル上の平均二乗誤差

$$
\mathcal{L}_{\mathrm{masked}}
=
\frac{
\displaystyle\sum_{p\in\mathcal{P}_{\mathrm{batch}}}\sum_{j\in \mathcal{H}_p}
\left(\widehat{x}_{p,j}-x_{p,j}\right)^2
}{
\displaystyle\sum_{p\in\mathcal{P}_{\mathrm{batch}}}|\mathcal{H}_p|
}
=
\frac{1}{N_{\mathrm{batch}}}\sum_{p\in\mathcal{P}_{\mathrm{batch}}}
\frac{1}{128}\sum_{j\in \mathcal{H}_p}
\left(\widehat{x}_{p,j}-x_{p,j}\right)^2
\tag{3.13}
$$

とする。右辺は主比較の50% mask、すなわち全画素で $|\mathcal{H}_p|=8\times16=128$ の場合である。補助的なmask率の条件では不可視チャネル数を対応する値へ置き換える。

M00では可視帯域から不可視帯域を復元し、M10・M01・M11では追加摂動を受けた可視帯域から元の観測の不可視帯域を復元する。Decoderは全256チャネルを出力するが、そのstepで可視だったチャネルの誤差は式(3.13)に含めない。Targetにpatchごとの追加正規化は行わない。

各stepでの損失範囲は不可視帯域に限られる一方、maskの抽選を繰り返すことで全帯域が復元対象となる機会を持つ。この意味で、学習課題は特定の固定帯域だけに限定されない。ただし、予測値自体が可視集合に依存するため、この目的関数は毎回全帯域を可視にして全帯域の誤差を計算する目的関数とは異なる。Maskに関する期待値を用いた説明を[付録B.4](../../appendices/mathematical_details.md#masked-objective)に示す。

### Maskを用いない比較条件A0

比較条件A0では、追加摂動とmaskを用いず、全帯域をencoderへ入力し、全256チャネルの平均二乗再構成誤差 $\mathcal{L}_{\mathrm{AE}}$

$$
\mathcal{L}_{\mathrm{AE}}
=
\frac{1}{N_{\mathrm{batch}}C}\sum_{p\in\mathcal{P}_{\mathrm{batch}}}\sum_{j=0}^{C-1}
\left(\widehat{x}_{p,j}-x_{p,j}\right)^2
\tag{3.14}
$$

を用いる。A0とMAE条件では、maskの有無と損失対象の両方が異なる。

### 共通の学習条件

学習にはAdamWを用い、800 epoch、batch size 1024、40 epochのwarmupとその後のcosine型学習率減衰を共通条件とする。演算にはFP16 autocastによる混合精度を用い、parameterとtargetはFP32に保持する。詳細な学習設定は[付録C](../../appendices/implementation_details.md)、lossの演算精度と履歴の集計は[付録C.3.1](../../appendices/implementation_details.md#training-precision)に示す。再構成損失にクラスタラベル、空間座標、空間的一貫性の評価値を含めず、クラスタリングは表現学習後に行う。

## 3.4.4 全帯域可視での表現抽出

学習後はencoderのparameterを固定し、全patch集合 $\mathcal{V}_{\mathrm{all}}=\{0,\ldots,15\}$ を可視として、

$$
\boldsymbol{z}^{\mathrm{full}}_p
=
\frac{
f_{\theta^\ast}(\boldsymbol{x}_p;\mathcal{V}_{\mathrm{all}})
}{
\left\|f_{\theta^\ast}(\boldsymbol{x}_p;\mathcal{V}_{\mathrm{all}})\right\|_2
}
\tag{3.15}
$$

により画素 $p$ の表現を抽出する。$\boldsymbol{z}^{\mathrm{full}}_p$ の上付きfullは、全patchを可視にして得た潜在であることを表す。$\theta^\ast$ は固定した学習済みparameterである。利用時には16 patchとCLSの計17 tokenをencoderに渡し、通常のマップ作成では追加摂動を行わない。

抽出時は評価modeと全帯域可視のmaskを明示し、FP32で演算する。学習時のランダムmask生成は利用せず、decoderによる再構成値もクラスタリング入力に用いない。抽出されるのは、最終CLSを16次元へ射影し単位化した表現である。

CVでは $\theta^\ast$ をtrain試料だけから求め、test試料で再学習しない。評価用に追加摂動を与える場合も、同じ固定encoderを全帯域可視で用いる。全体fitによる解釈ではfit対象が異なるため、その範囲は第4章で別に定義する。

---

## 執筆メモ（本文外）

- **参照資料・照合先：** ChemoMAE v0.2.2のモデル・mask生成・masked MSE、[固定設定](../../../src/wood_degradation_map/experiments/config.py)、[学習](../../../src/wood_degradation_map/experiments/training.py)、[抽出](../../../src/wood_degradation_map/experiments/neural.py)。原稿作成時の読み取り照合に基づく方法の記述であり、全実験の完了報告ではない。
- **残る整備：** Transformer・MAE・denoisingの原典の引用を最終稿で整備する。
