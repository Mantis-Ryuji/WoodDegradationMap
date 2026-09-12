# 表現による変動の強調・抑制：数理と補助診断の計画

作成日: 2026-09-12

## 1. 目的と合意した範囲

**クラスタ単位・画素単位の方針に合意し、付録B.5を強調・抑制の数理へ改訂。対象・抽出・数値規約はOpen、診断実装・実行・結果の原稿反映は未実施。**

中心の問いを次に置く。

> 入力のどのスペクトル差が潜在のcosine幾何で強調され、どの差が弱められるか。
> その対応がMAE群のcorruption条件によってどう変わり、領域分割や指定摂動への応答をどう説明するか。

- 対象の軸はM00・M10・M01・M11とする。Random maskもdenoisingのcorruptionであり、TGN・shiftは追加corruptionとして扱う。
- 強調と抑制の双方を調べる。画素間差をすべて有用な信号、人工摂動を実際の測定誤差と仮定しない。
- A1等の学習条件を追加しない。既存の学習済みモデルを使う補助診断として計画する。
- 主評価、学習条件、split、checkpoint選択、既定の代表・差スペクトルの定義を変更しない。

「学習による変化」は学習済み表現と入力幾何の対応、および同一入力に対する条件間差を指す。checkpointを追跡せず、学習途中の時間的な増減は主張しない。

[実験プロトコル](experiment_protocol.md)、[評価・集計規約](evaluation_metrics.md)、[全体fitと解釈](visualization_and_interpretation.md)を継承する。本計画はCV開始後に立てた探索的な補助診断であり、事前の主要仮説・指標として遡及記載しない。

## 2. 解析の二つの単位と比較対象

| 単位 | 主に答える問い | 用いる情報 |
| --- | --- | --- |
| クラスタ単位 | 分割された領域間のどの形状差が、潜在で大きく／小さく表れるか | 同じ所属画素の平均スペクトル・平均潜在・平均残差、内部の広がり |
| 画素単位 | 平均で消える局所的な差や摂動応答が、どう拡大／縮小するか | 同じ実画素対の入力・潜在距離、同じ実画素の摂動前後、残差、固定中心との位置関係 |

既存CVモデルのheld-out試料を全帯域可視で調べることを基本案とし、使用runは第8節で確定する。全体fitは別枠の記述的解析とする。既定対象にないM10・M01の全体学習は追加しない。

クラスタには次の二つの使い方がある。

1. **各モデル自身のクラスタによる記述**：そのモデルがどのスペクトル差で分けたかを説明する。
2. **固定した共通画素集合による条件間比較**：同じ所属画素を各モデルへ渡し、入力集団の違いと表現の違いを切り分ける。

比較用の分割基準はOpen。M00等のクラスタを使う場合は基準モデルの分割に条件づけた解析と明記し、クラスタに依存せず選ぶ共通画素対を併用する。共通集合でも基準モデルに適した群を選ぶ効果は残るため、自身のクラスタの中心間距離だけでモデルの優越性を判定しない。

同じfold・学習反復・試料の中で比較する。異なるモデルやfoldの16次元潜在を直接差し引いたり、OOF潜在を一つの座標系として平均したりしない。集約するのは、各固定モデル内で計算した距離・分散等のスカラー量である。

## 3. 共通記号と数理的な前提

本計画内では固定モデルの添字を省略する。画素 $p$ のSNV入力を $\boldsymbol{x}_p\in\mathbb{R}^{C}$、$C=256$、$\rho=\sqrt{C-1}$、入力の単位方向を $\boldsymbol{s}_p=\boldsymbol{x}_p/\rho$ とする。全帯域可視のencoderと単位化を合わせた写像を $F$ とし、

$$
\boldsymbol{z}_p=F(\boldsymbol{x}_p),\qquad
\widehat{\boldsymbol{x}}_p=W\boldsymbol{z}_p+\boldsymbol{b},\qquad
\boldsymbol{e}_p=\boldsymbol{x}_p-\widehat{\boldsymbol{x}}_p
$$

と書く。$\|\boldsymbol{s}_p\|_2=\|\boldsymbol{z}_p\|_2=1$ の理想式を用いる。実装時には実測normと数値保護を照合する。

数式は同一モデルのbias付き線形decoderを前提とする。SVDは $W=U\Sigma V^{\mathsf T}$ とし、正の特異値 $\sigma_i$ に対応する左右の特異ベクトルを $\boldsymbol{u}_i,\boldsymbol{v}_i$ とする。数値rank・小特異値の判定規則は実施前に確定する。

## 4. クラスタ単位で主張できることと診断

### 4.1 平均スペクトル・平均潜在・平均残差の対応

非空の画素集合 $\mathcal{P}_k$ に対して、非負で和が1の共通重み $w_{kp}$ を用い、入力・潜在・残差をそれぞれ平均する。

$$
\overline{\boldsymbol{x}}_k=\sum_{p\in\mathcal{P}_k}w_{kp}\boldsymbol{x}_p,\qquad
\overline{\boldsymbol{z}}_k=\sum_{p\in\mathcal{P}_k}w_{kp}\boldsymbol{z}_p,\qquad
\overline{\boldsymbol{e}}_k=\sum_{p\in\mathcal{P}_k}w_{kp}\boldsymbol{e}_p.
$$

線形decoderでは、

$$
\overline{\boldsymbol{x}}_k
=W\overline{\boldsymbol{z}}_k+\boldsymbol{b}+\overline{\boldsymbol{e}}_k,
\qquad
\Delta\overline{\boldsymbol{x}}
=W\Delta\overline{\boldsymbol{z}}+\Delta\overline{\boldsymbol{e}}
$$

が厳密に成立する。$\Delta$ は明示した二クラスタの差である。

**診断と解釈：** クラスタ間の実測平均差、復元平均差、符号付き残差差を同じ波長軸に示す。どの形状差が平均潜在から読み出され、どの差が取りこぼされているかを記述できる。平均残差は画素間で相殺され得るため、残差の二乗量や内部のばらつきも併記する。

平均スペクトルをencoderへ再入力しない。非線形写像では一般に $F(\overline{\boldsymbol{x}}_k)\neq\overline{\boldsymbol{z}}_k$ であり、SNVの平均スペクトルも一定normを満たすとは限らない。上の等式に用いる平均潜在は単位化しない。平均方向の計算は別に行う。

診断用の算術平均は[既定の中央値による代表線](visualization_and_interpretation.md)と区別する。試料をまたぐ場合も三つの量に共通重みを使い、試料内集計・試料macro比較を基本案として第8節で確定する。

### 4.2 中心方向の分離とクラスタ内の広がり

集合 $k,l$ から上記重みで画素を独立に選ぶ場合、平均cosine不類似度は全画素対を列挙せず、

$$
\mathbb{E}_{p\in k,q\in l}[d_{\mathrm{in}}(p,q)]
=1-\frac{\overline{\boldsymbol{x}}_k^{\mathsf T}\overline{\boldsymbol{x}}_l}{\rho^2},
\qquad
\mathbb{E}_{p\in k,q\in l}[d_{\mathrm{lat}}(p,q)]
=1-\overline{\boldsymbol{z}}_k^{\mathsf T}\overline{\boldsymbol{z}}_l
$$

と計算できる。$d_{\mathrm{in}},d_{\mathrm{lat}}$ の画素対の定義は第5.1節に示す。

$a_k=\|\overline{\boldsymbol{z}}_k\|_2>0$、$\boldsymbol{c}^{\mathrm{emp}}_k=\overline{\boldsymbol{z}}_k/a_k$ とすると、

$$
1-\overline{\boldsymbol{z}}_k^{\mathsf T}\overline{\boldsymbol{z}}_l
=(1-a_ka_l)+a_ka_l\left(1-(\boldsymbol{c}^{\mathrm{emp}}_k)^{\mathsf T}\boldsymbol{c}^{\mathrm{emp}}_l\right).
$$

また、クラスタ内の重み付き二乗偏差は、

$$
\sum_{p\in\mathcal{P}_k}w_{kp}\|\boldsymbol{z}_p-\overline{\boldsymbol{z}}_k\|_2^2
=1-a_k^2
$$

である。入力側も $\overline{\boldsymbol{x}}_k/\rho$ で同じ式が成り立つ。

**診断と解釈：** 入力・潜在それぞれについて、平均方向間のcosine不類似度、平均ベクトルのnorm、平均画素対不類似度を併記する。平均画素対距離の増加が、中心方向の分離と内部の拡散のどちらに対応するかを区別できる。どれか一つの増減を単独で良好と扱わない。

$\boldsymbol{c}^{\mathrm{emp}}_k$ は解析した画素集合の平均方向であり、trainでfitしたKMeans中心と同一とは限らない。平均がゼロなら方向は未定義とする。$k=l$ の独立抽出は同一画素を選ぶ場合も含む。等重みの異なる二画素だけを対象にする場合、平均不類似度は $n_k/(n_k-1)$ 倍となるため、計数方式を混在させない。

## 5. 画素単位で主張できることと診断

### 5.1 同じ入力差が潜在でどれだけ拡大・縮小するか

共通の実画素対 $(p,q)$ に対して、

$$
d_{\mathrm{in}}(p,q)=1-\boldsymbol{s}_p^{\mathsf T}\boldsymbol{s}_q,\qquad
d_{\mathrm{lat}}(p,q)=1-\boldsymbol{z}_p^{\mathsf T}\boldsymbol{z}_q
$$

を比較する。双方を単位方向で測ることで、SNVのnorm $\rho$ と潜在norm 1の尺度差を除く。$d_{\mathrm{in}}>0$ のとき、

$$
g_{pq}=\sqrt{\frac{d_{\mathrm{lat}}(p,q)}{d_{\mathrm{in}}(p,q)}}
=\frac{\|\boldsymbol{z}_p-\boldsymbol{z}_q\|_2}{\|\boldsymbol{s}_p-\boldsymbol{s}_q\|_2}
$$

はその画素対の弦長の比となる。1より大きければ拡大、小さければ縮小である。局所的な微分や全入力に対するLipschitz定数ではない。

**診断：** 入力距離対潜在距離の散布図と同じ対の条件間差を示す。クラスタ内・間の層別とクラスタ非依存の対を区別し、拡大・縮小の分布、入力距離依存性、例外を調べる。

比だけを代表値にしない。入力距離ゼロでは比を未定義とし、極小距離での比の不安定性は元の二距離と併読する。小分母の取扱い、対の抽出方法・数・距離階級はOpen。全組合せの列挙は計画しない。画素対の数を独立試料数とみなさない。

### 5.2 SVDで入力差・残差・潜在の拡大率を結ぶ

同一モデルでは、画素対にも第4.1節のクラスタ平均対にも、

$$
\Delta\boldsymbol{x}=W\Delta\boldsymbol{z}+\Delta\boldsymbol{e},\qquad
\boldsymbol{v}_i^{\mathsf T}\Delta\boldsymbol{z}
=\frac{\boldsymbol{u}_i^{\mathsf T}\Delta\boldsymbol{x}-\boldsymbol{u}_i^{\mathsf T}\Delta\boldsymbol{e}}{\sigma_i}
\quad(\sigma_i>0)
$$

が成立する。入力方向 $\boldsymbol{s}=\boldsymbol{x}/\rho$ の尺度で見る場合、対応する逆方向の倍率は $\rho/\sigma_i$ である。

**診断：** 入力差・残差差・潜在差の射影を対応づけ、小特異値に伴う大きな潜在差が観測差と対応するか、残差に敏感かを検討する。特異値はdecoderの読み出し倍率であり、実際の変動と合わせて読む。

残差はdecoderの列空間に直交するとは限らない。したがって、

$$
\|\Delta\boldsymbol{x}\|_2^2
=\|W\Delta\boldsymbol{z}\|_2^2+\|\Delta\boldsymbol{e}\|_2^2
+2(W\Delta\boldsymbol{z})^{\mathsf T}\Delta\boldsymbol{e}
$$

の交差項を無視し、「復元成分のエネルギー＋失われた情報量」と解釈しない。個々の小さなMSEだけで、画素間の弱い差の残差も小さいと仮定しない。

Rankが不足する場合はkernel方向の潜在変動を別に記述し、ゼロ特異値で除算しない。小特異値の逆数を結果に合わせて切り詰めない。符号任意性・近接／重複特異値を考慮し、異なるモデルの「第i軸」を同じ化学的意味へ対応づけない。波長方向の比較には符号整列または部分空間としての扱いが必要となる。

### 5.3 同じ画素への摂動：ばらつきとcleanからのずれ

追加摂動 $a$ を受けた入力を $\boldsymbol{x}_{p,a}$、全可視潜在を $\boldsymbol{z}_{p,a}=F(\boldsymbol{x}_{p,a})$ とし、cleanとの差を入力・潜在の両方で測る。入力の摂動は条件間で共通化し、既定のTGN・shift・併用の評価定義を継承する。

固定画素に対して $\boldsymbol{\mu}_p=\mathbb{E}_a[\boldsymbol{z}_{p,a}]$、$\Sigma_p=\operatorname{Cov}_a(\boldsymbol{z}_{p,a})$ とすると、

$$
\mathbb{E}_a\|\boldsymbol{z}_{p,a}-\boldsymbol{z}_p\|_2^2
=\operatorname{tr}(\Sigma_p)+\|\boldsymbol{\mu}_p-\boldsymbol{z}_p\|_2^2.
$$

左辺はcleanからのcosine不類似度の期待値の2倍である。摂動ごとのばらつきが小さくても、すべての摂動でcleanから同じ方向へずれる場合があるため、二項を分ける。$\operatorname{tr}$ は行列の対角和を表す。

有限の $D$ drawsについて恒等式を照合する際は、平均と分母 $D$ の共分散を使う。分母 $D-1$ の不偏共分散を用いる推定とは区別し、そのまま恒等式へ代入しない。

**診断と解釈：** 入力変化に対する潜在変化の分布、摂動ごとのばらつき、平均的なずれ、SVD方向ごとの変動を調べる。画素間の差の変化と併読し、一律の拡大・縮小と、変動の種類による違いを区別する。補助量を単一の総合scoreへ合成しない。

復元は常に同じclean targetを基準にする。$\boldsymbol{e}_{p,a}=\boldsymbol{x}_p-(W\boldsymbol{z}_{p,a}+\boldsymbol{b})$ とすると、

$$
W(\boldsymbol{z}_{p,a}-\boldsymbol{z}_p)=\boldsymbol{e}_p-\boldsymbol{e}_{p,a}.
$$

第5.2節の「異なるclean targetの差」と、この同一targetへの復元を混同しない。摂動入力そのものへの自己再構成誤差にtargetを変更しない。

### 5.4 Masked lossが潜在変動へ与える制約

これは学習課題を説明する数理的補足とする。固定モデル・固定画素・固定maskのもと、追加摂動だけに期待値を取る。不可視チャネル集合を $H$、その数を $h$、decoderの該当行を $W_H,\boldsymbol{b}_H$ とする。潜在の平均・共分散を $\boldsymbol{\mu}_{p,H},\Sigma_{p,H}$ とすると、

$$
\mathbb{E}_a[\mathcal{L}_{p,H,a}]
=\frac{\|W_H\boldsymbol{\mu}_{p,H}+\boldsymbol{b}_H-\boldsymbol{x}_{p,H}\|_2^2}{h}
+\frac{\operatorname{tr}(W_H\Sigma_{p,H}W_H^{\mathsf T})}{h}.
$$

第2項は $h^{-1}\sum_i\sigma_{H,i}^2\operatorname{Var}_a(\boldsymbol{v}_{H,i}^{\mathsf T}\boldsymbol{z}_{p,H,a})$ であり、方向ごとに異なる重みで潜在のばらつきが損失へ現れる。

以下の上界は必要性を判断する任意候補とし、現付録には掲載しない。$a,a'$ を独立な摂動抽出とすると、単位潜在と列full rankの $W_H$ について、

$$
\mathbb{E}_{a,a'}[1-\boldsymbol{z}_{p,H,a}^{\mathsf T}\boldsymbol{z}_{p,H,a'}]
=1-\|\boldsymbol{\mu}_{p,H}\|_2^2
\leq\frac{h}{\sigma_{\min}(W_H)^2}\mathbb{E}_a[\mathcal{L}_{p,H,a}]
\quad\bigl(\sigma_{\min}(W_H)>0\bigr)
$$

が得られる。

低いmasked lossがcosine変動を制限する条件を示すが、小特異値では緩くなる。Decoderも学習条件で変わるため、小さな予測変動と小さな潜在変動は同一視しない。

この期待値は摂動対間の変動であり、clean対摂動の変動とは第5.3節の平均のずれを介して区別する。Maskを固定した条件付きの式を、mask間の不変性や全可視利用へ直接拡張しない。M00で学習時の追加摂動がないことは、mask自身のdenoising作用がないことを意味しない。共通評価摂動をM00へ加えたリスクにこの式を適用する場合、そのリスクをM00の学習目的と呼ばない。

**追加診断の位置づけ：** まず全可視での診断を実施し、その結果の説明に必要なら固定maskでの再評価を行う。対象maskとdrawはOpen。全mask列挙や追加学習は行わない。期待値の式を有限drawへ照合する際は分母を統一し、有限抽出のばらつきを報告する。移動するparameterで集計されたepoch train lossを代入して保証値としない。

### 5.5 潜在の変化とlabel flipの接続

境界marginは必要性を判断する任意候補とし、現付録には掲載しない。

全可視clean潜在 $\boldsymbol{z}_p$ の割当先を $k$、固定した単位KMeans中心を $\boldsymbol{c}_j$ とする。異なる中心間で

$$
\eta_p=\min_{j\neq k}
\frac{\boldsymbol{z}_p^{\mathsf T}(\boldsymbol{c}_k-\boldsymbol{c}_j)}{\|\boldsymbol{c}_k-\boldsymbol{c}_j\|_2}
$$

を定義すると、$\|\boldsymbol{z}_{p,a}-\boldsymbol{z}_p\|_2<\eta_p$ はlabelが変わらない十分条件になる。これは各割当境界の超平面までの距離から得られる。等距離tieや同一中心は退化として扱う。

**診断と解釈：** 潜在変化の大きさ・向き、境界までの余裕、実際のflipを併読する。潜在が動いたのにlabelが変わらない場合と、小さな動きで変わる場合を説明できる。十分条件を外れたことをflipの予測としない。中心を再fitせず、既定のLFRそのものは変更しない。

## 6. 実測後に述べる範囲

第4〜5節の数理は測定量の関係を与える。補助診断では、評価した画素集合・対・摂動・maskの範囲で、
どの差が強調／抑制され、それが入力形状・残差・クラスタ内拡散・割当境界とどう対応したかを記述する。
全体の一律な拡大・縮小と、変動の種類による異なる応答を区別し、有用性は波長別の差、マップ、occupancy、主評価と併読する。

これらの恒等式だけから化学成分・重要度・単一因子への因果帰属は決められない。全入力の距離保存、
未知の実測誤差への頑健性、化学状態の不変性、劣化分類精度へ一般化せず、効果が見られない場合も報告する。

## 7. 実施順序・成果物・論文への反映

| 段階 | 作業 | 完成物・次へ進む条件 |
| --- | --- | --- |
| 1. 定義の確定 | 第8節の比較集合、重み、抽出、数値規約を決め、利用できる既存成果物を確認 | 診断configと記録項目の仕様。実行負荷・保存先も具体化 |
| 2. 数式の最小照合 | 実装時に小規模の合成配列と少数の実画素で、平均の可換性、距離平均、SVD対応、分散分解を照合 | 非線形encoderで平均を交換していないこと、零norm・rank不足・残差交差項の扱いを確認 |
| 3. クラスタ単位の診断 | 既存固定モデルの共通対象から入力・潜在・復元・残差を集計 | 平均／差スペクトル、平均方向距離と内部の広がり、対応するSVD説明 |
| 4. 画素単位の診断 | 共通画素対と共通評価摂動から拡大・縮小・平均のずれを調べる | 距離散布図、条件間対応差、摂動応答とLFRの対応。全画素対の列挙はしない |
| 5. 必要な機構確認 | 全可視結果を説明する問いが残る場合に固定mask診断を行う | $W_H$ と摂動間変動。上界を採用する場合は、その緩さも記録 |
| 6. 試料・反復での集約 | 試料macroとpaired比較を適用し、反例・未定義・対象数を併記 | 効果が見られた範囲と、見られなかった範囲を区別した記述 |
| 7. 実測結果を原稿へ反映 | 改訂済みの数理と、確定した診断手順・実測結果を対応づける | 評価・解釈手順と補足結果。数理上の関係を実測した効果と区別 |

図表は段階3のクラスタ形状差、段階4の実画素対・指定摂動応答の三組を基本候補とし、選定条件を結果を見る前に定める。

主評価の有効画素・K・反復を変更しない。診断用の部分抽出を行う場合は、主評価の母集団と区別して対象数・重みを示す。計算はbatch集計を基本案とし、全対距離行列や全maskの生成を避ける。既存表現の再利用可否、復元・摂動forwardの必要数、メモリ・GPU負荷は実装前に見積もる。

改訂した付録との対応は次のとおり。数理の記載は診断の実装・実行済みを意味しない。

| 付録の箇所 | 数理と診断の対応 |
| --- | --- |
| [B.5.1](../../thesis/appendices/mathematical_details.md#pairwise-gain) | 同じ画素対の入力・潜在距離と有限差分の拡大率（本計画§5.1） |
| [B.5.2](../../thesis/appendices/mathematical_details.md#svd-interpretation) | 入力差・潜在差・残差差のSVD対応（§5.2） |
| [B.5.3](../../thesis/appendices/mathematical_details.md#cluster-mean-geometry) | 共通重みのクラスタ平均、方向差と集中度（§4） |
| [B.5.4](../../thesis/appendices/mathematical_details.md#perturbation-response) | 固定画素への摂動応答をばらつきとcleanからの平均のずれに分ける（§5.3） |
| [B.5.5](../../thesis/appendices/mathematical_details.md#masked-loss-variation) | 固定maskのlossを平均予測誤差とdecoderで重み付けされた変動へ分ける（§5.4） |
| [B.6](../../thesis/appendices/mathematical_details.md#numerical-geometry) | 単位化・有限精度・rank等の照合条件。精度比較は自動追加しない |

本文に定義・前提・主要結果、付録Bに導出と成立条件、評価方法に診断手順、補足結果に実測図表を置く。数理的に計算可能なことと、実測が解釈を支持したことを区別する。

<a id="open-items"></a>

## 8. 実装・実行前に確定する事項

| Open事項 | 推奨する検討の起点 | 決定が必要な理由 |
| --- | --- | --- |
| 使用run・対象範囲 | 既存CVのheld-out試料、同じfold・反復のMAE4条件。全体fitは別解析 | 完了状況と利用可能な保存物を確認し、OOFと全体fitを混在させない |
| 比較用の共通分割 | 結果を見る前に基準分割を指定。各モデル自身の分割は記述用に併記 | 分割の選択と表現の変化を混同せず、基準条件への依存を明示する |
| クラスタ平均・対の重み | 試料内の算術平均と試料macroを起点とする | 画素数・クラスタ占有率・寄与試料の差で結論が変わり得る |
| 画素・画素対の抽出 | 条件共通、試料内の対を起点に、cluster非依存の対と層別の対を用意 | 数・seed・距離階級・重みを固定し、良い例の事後選択と二次コストを避ける |
| 診断に用いるKと例示範囲 | 既定の代表表示 $K_0=8$ と共通K集合の方針を継承 | 補助診断を全Kで行う範囲と、結論のK依存性を確かめる範囲を定める |
| 摂動drawと固定mask診断 | 全可視は既定評価の共通摂動・draw規約を起点とする | 学習時と評価時の適用確率を区別。固定mask再評価の要否・数・seedは別途決める |
| 数値規約 | 既定FP32抽出を継承。集計精度と恒等式照合を仕様化 | 小距離、零平均、数値rank、小特異値、重複中心、有限drawの分母・許容差を定める |
| 保存・負荷 | 既存成果物を読み、診断出力を別に保存 | 出力schema・出力先・再開規約・forward数・GPU負荷を具体化してから実行する |

上記は採用値ではなく検討の起点である。確定後の実行はリポジトリの実行規約に従う。

## 9. 関連文書

- [付録B.5以降](../../thesis/appendices/mathematical_details.md#latent-decoder)：画素対とクラスタ平均、入力差・残差・摂動応答、固定maskのlossと数値条件。
- [モデルとloss](../../thesis/chapters/3_analysis_methods/representation_learning.md)：clean target、不可視帯域loss、全可視での抽出。
- [クラスタリング](../../thesis/chapters/3_analysis_methods/clustering_mapping.md)：単位中心、fit対象とtestへの固定適用。
- [全体可視化のスペクトル集計](visualization_and_interpretation.md)：第4.1節の中央値による代表線とCVとの区別。
- [解釈メモ第8節](../interpretation_notes.md#decoder-residual-discussion)：既存の残差・SVD確認候補。今回、強調と抑制、クラスタ単位と画素単位の両方へ計画を具体化した。

計画時点では文書照合と代数的確認のみ実施し、コード・学習済み重み・実画素・追加forward・テストは未実行。
数式の新規性や先行研究に対する優先性を主張する計画ではない。
