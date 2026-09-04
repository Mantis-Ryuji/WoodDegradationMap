# 評価指標

## 1. ステータス

**Fixed**

本研究には画素単位の正解劣化ラベルがないため、教師あり分類精度やIoUによる順位付けは行わない。
CVの主評価は次の3系列とする。

1. cosine-silhouette
2. LLA-3、LLA-5、LLA-9
3. label flip rate

これらを単一scoreへ合成せず、表現分離、空間的一貫性、摂動安定性として個別に報告する。

## 2. 共通規約

### 2.1 ラベル

- 背景ラベルは0とする。
- クラスタラベルは1からKまでとする。
- Cosine-KMeansが返す0からK-1までのラベルは、ラベルマップへ格納するときに1を加える。
- 背景画素はすべてのクラスタ評価から除外する。

### 2.2 評価単位

独立な評価単位は画素ではなく `KYOw...` で識別される試料である。

1. test画素について必要なpixel-wise量を計算する。
2. 同一試料内で平均し、試料ごとの指標値を得る。
3. test試料を同じ重みでmacro平均する。
4. 5-foldのout-of-fold予測をまとめ、全試料の平均と試料間標準偏差を報告する。

画素数の多い試料が結果を支配するpixel-weighted平均を、条件比較の代表値にしない。
条件差は、同じ試料、fold、Kに対するpairedな値として扱う。

### 2.3 K

基準値$K_0$と共通range$\mathcal{K}$は
[experiment_protocol.md](experiment_protocol.md)に従う。
主表は$K_0$、補助図は$K\in\mathcal{K}$の曲線とする。

## 3. cosine-silhouette

### 3.1 定義

test埋め込み$z_i$と、train centroidsによって割り当てたラベル$c_i$を用いる。
cosine距離を

$$
d_{\cos}(z_i,z_j)
=1-\frac{z_i^\top z_j}{\lVert z_i\rVert_2\lVert z_j\rVert_2}
$$

とする。同一クラスタ内の平均距離を$a_i$、最も近い別クラスタへの平均距離を$b_i$として、

$$
s_i=\frac{b_i-a_i}{\max(a_i,b_i)}
$$

を計算する。高いほど、同一クラスタ内で近く別クラスタから離れている。

### 3.2 計算方法

- foldの全test木材画素をまとめてpixel-wise silhouetteを計算する。
- 得られた$s_i$を試料境界に従って分割し、試料内平均からmacro平均を求める。
- 背景0は入力前に除外する。
- singleton clusterの画素scoreは0とする。
- test集合全体に2クラスタ以上存在しない場合は未定義として記録し、cluster collapseとして扱う。

実装にはChemoMAE v0.2.1の
`silhouette_samples_cosine_gpu`を使用する。この関数はcosine距離の$N\times N$行列を作らず、
cluster統計とchunk処理を用いてexactなsilhouetteを計算する。
chunk sizeは計算結果を変えない実装パラメータとして扱う。

## 4. LLA: Local Label Agreement

### 4.1 定義

試料$m$の木材領域を$\Omega_m$、画素$p$のラベルを$y_p\in\{1,\ldots,K\}$とする。
幅$r\in\{3,5,9\}$の正方近傍から中心画素自身と背景を除いた集合を

$$
\mathcal{N}^{*}_{r}(p)
=\{q\in\mathcal{N}_{r}(p)\cap\Omega_m:q\ne p\}
$$

とする。試料$m$のLLA-rを

$$
\mathrm{LLA}_{r,m}
=
\frac{
\sum_{p\in\Omega_m}
\sum_{q\in\mathcal{N}^{*}_{r}(p)}
\mathbf{1}[y_p=y_q]
}{
\sum_{p\in\Omega_m}|\mathcal{N}^{*}_{r}(p)|
}
$$

と定義する。

| 指標 | 中心以外の最大近傍数 | 主に確認する構造 |
| --- | ---: | --- |
| LLA-3 | 8 | salt-and-pepper状の局所変動 |
| LLA-5 | 24 | 小から中スケールのまとまり |
| LLA-9 | 80 | より広い面状・縞状構造 |

LLAは高いほど局所ラベル一致が多い。ただし、クラスタ占有率が極端に偏る場合にも高くなり得るため、
単独では解釈せずcluster occupancyを併記する。
LLA-3/5/9は重み付き平均せず、個別に報告する。

## 5. label flip rate

### 5.1 定義

clean入力に対するtest画素$p$のラベルを$y_p$、同じ画素の摂動入力に対するラベルを
$y_p^{(a)}$とする。試料$m$、摂動$a$のlabel flip rateを

$$
\mathrm{LFR}_{m}^{(a)}
=\frac{1}{|\Omega_m|}
\sum_{p\in\Omega_m}
\mathbf{1}[y_p\ne y_p^{(a)}]
$$

と定義する。低いほど摂動に対して安定している。

### 5.2 固定事項

- cleanと摂動後で同一のtest画素を同じ順序で使用する。
- encoderとtrain centroidsを固定する。
- 摂動後に再学習または再クラスタリングしない。
- 評価時の対象augmentation適用確率は1とする。
- ChemoMAEの`SpectraAugmenter`を評価時だけ明示的にtraining modeへ切り替えて摂動を生成する。
- 摂動seedと反復数は評価開始前に固定し、全条件で同一の摂動系列を使用する。

次の3条件を混合せず、個別に報告する。

| 摂動 | 評価時設定 |
| --- | --- |
| noise | `noise_prob=1`, `shift_prob=0` |
| shift | `noise_prob=0`, `shift_prob=1` |
| noise + shift | `noise_prob=1`, `shift_prob=1` |

shift幅はChemoMAEの既定範囲を用いる。noise角度はCV開始前に固定し、強度sweepを行わない。
ChemoMAEのnoise角度は設定範囲から一様分布で抽出されるため、正規分布$N(0,\sigma)$とは記述しない。

## 6. 補助診断

次はCVにおける条件選択指標には使用せず、cluster collapseまたはラベルマップの異常を確認する。

| 診断 | 用途 |
| --- | --- |
| cluster occupancy | 空クラスタ、単一クラスタへの集中、極端な不均衡の確認 |
| cluster size distribution | 条件間および試料間のラベル分布の確認 |
| isolated label rate | 孤立画素の確認 |
| small component mass rate | 小連結成分への破片化の確認 |
| label map | 空間構造の目視確認 |

これらによって主条件またはbest条件を事後選択しない。

## 7. 結果の報告

- $K_0$について、条件ごとの試料macro平均と試料間標準偏差を表にする。
- $K\in\mathcal{K}$について、cosine-silhouette、LLA-3/5/9、3種類のLFRを曲線で示す。
- M00、M10、M01、M11は計画されたpaired contrastとして比較する。
- 指標間で結論が異なる場合は、総合順位に潰さずtrade-offとして報告する。
- LLA=1またはLFR=0だけを良好な結果とみなさず、cluster occupancyと合わせてcollapseを確認する。
- Hungarian matchingは全体可視化にだけ用い、CV指標の計算には使用しない。
