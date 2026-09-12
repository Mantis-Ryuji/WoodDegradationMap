# 本文・付録の共通記号表

第3章、第4章のLLA・補正LLA、付録A〜Cの記号を対応づける。各節の初出でも意味を説明し、同じ量には同じ記号、異なる量には別の記号または対象を示す添字を用いる。

画素の添字を省略した $\boldsymbol{x},\mu,s,\boldsymbol{h},\boldsymbol{z},\mathcal{A},\mathcal{H}$ は、それぞれ一つの画素の $\boldsymbol{x}_p,\mu_p,s_p,\boldsymbol{h}_p,\boldsymbol{z}_p,\mathcal{A}_p,\mathcal{H}_p$ と同じ意味である。省略する節ではその旨を明記する。ベクトルの第 $j$ 成分は、例えば $\boldsymbol{x}_p$ に対して $x_{p,j}$ と書く。

## 添字・次元・集合

| 記号 | 意味・範囲 | 主な説明箇所 |
| --- | --- | --- |
| $p,q$ | 画素の添字。比較する二画素を $p,q$ とし、複数試料では試料との対応も保持する | 3.2、3.4、3.5、B.5 |
| $(u,v)$ | 元画像の行・列座標 | 3.2、A.1 |
| $b$ | 元の測定bandの添字。0始まり | 3.2、A.1〜A.3 |
| $j$ | 補間後の出力チャネルの添字。0〜255 | 3.2、3.4、B.3 |
| $a$ | スペクトルpatchの添字。0〜15 | 3.4 |
| $k,l,K$ | クラスタの添字とクラスタ数。比較する二群を $k,l$ とし、$k=1,\ldots,K$ | 3.5、B.5.3 |
| $i$ | Decoderの特異値・特異ベクトルの添字。1〜16 | B.5.2 |
| $C_{\mathrm{src}}$ | 元の測定band数。256 | A.1 |
| $C_{\mathrm{keep}}$ | 保持する測定band数。現行記録では222 | 3.2、A.3 |
| $C$ | 補間後のチャネル数。256 | 3.2 |
| $N_v$ | Referenceの検出器列数。320 | A.2 |
| $N_{\mathrm{patch}},C_{\mathrm{patch}}$ | Patch数と1 patchのチャネル数。ともに16 | 3.4 |
| $d_{\mathrm{emb}},d_z$ | Tokenの埋め込み幅256と潜在次元16 | 3.4 |
| $d_{\mathrm{feat}}$ | クラスタリング入力の次元。B0は256、PCA・NNは16 | 3.5 |
| $\Omega$ | 前処理の品質条件を通過した有効画素集合 | 3.2、3.5 |
| $\mathcal{P}_{\mathrm{batch}},N_{\mathrm{batch}}$ | Mini-batch内の画素集合とその画素数 | 3.4 |
| $\mathcal{V}_p,\mathcal{V}_{\mathrm{all}}$ | 画素ごとの可視patch集合と全patch集合 | 3.4 |
| $\mathcal{H}_p,N_{\mathrm{hide}}$ | 不可視チャネル集合と、mask数を固定したときの要素数。主比較の要素数は128 | 3.4、B.4 |
| $\mathcal{F},N_{\mathcal{F}},\mathcal{F}_k$ | 中心推定用のfit画素集合、その画素数、第 $k$ クラスタに属するfit画素集合 | 3.5 |

## 前処理・SNV

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $S_{\mathrm{raw}}(u,v,b)$ | 試料の生強度。$S$ は測定信号（signal）、rawは反射率変換前の値を表す | 3.2、A.1 |
| $W(v,b),D(v,b)$ | 同じ検出器列・測定bandのwhite/dark reference | 3.2、A.2 |
| $R(u,v,b),R_{p,b}$ | 元の測定band上の反射率。座標表記と画素添字表記 | 3.2、A.3 |
| $\lambda_b,\widetilde{\lambda}_j$ | 測定波長と補間先波長。単位nm | 3.2、A.3 |
| $\omega_j$ | 前処理の線形補間で右側の測定値に掛ける重み | A.3 |
| $\widetilde{\boldsymbol{R}}_p,\widetilde{R}_{p,j}$ | 共通grid上の補間反射率スペクトルとその成分 | 3.2、A.3、B.1 |
| $\mu_p,s_p$ | 補間反射率の画素内平均と標本標準偏差 | 3.2、B.1 |
| $\boldsymbol{x}_p,x_{p,j}$ | 追加摂動前の最終SNVスペクトルとその成分 | 3.2 |
| $\rho$ | 理想的なSNVのL2 norm。$\rho=\sqrt{C-1}$ | 3.3、B.1〜B.3 |
| $P$ | 画素内の平均を除く中心化射影行列 | 3.3、B.1 |
| $\mathcal{S}_{\mathrm{SNV}}$ | 平均ゼロ・norm $\rho$ の制約を満たすスペクトルの集合 | B.1 |
| $\kappa,\beta$ | SNVの不変性を説明するための正の共通scaleと共通offset | B.1.2 |
| $A(u,v)$ | 全測定bandsの生強度を足した、マスク用の明るさの指標 | A.1 |
| $M_{\mathrm{cand}},M_{\mathrm{morph}}$ | 強度による候補マスクと形態処理後のマスク | A.1 |
| $\mathcal{B}_{\mathrm{ero}}$ | 半径1画素の円盤状erosion構造要素 | A.1 |
| $Q(v,b),\overline{Q}_b,s^{(0)}_{Q,b}$ | White−darkの差、その列方向平均、母標準偏差。上付き $(0)$ は自由度補正0 | A.2 |
| $\gamma_b$ | Reference由来SNR proxy | A.2 |
| $\operatorname{Norm}_R(u,v)$ | 有効画素の補間反射率のL2 normを元座標へ配置した確認量 | A.5.2 |

図A.1の診断用SNVは、波長除外・補間前の全測定bandsで計算した値である。学習入力の $\boldsymbol{x}_p$ と混同しないよう、本文中では「診断用SNV」と呼び、同じ記号を割り当てない。

## スペクトル摂動

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $\boldsymbol{n},\boldsymbol{n}_p$ | SNV入力の単位方向 $\boldsymbol{x}/\rho$ と、画素を明示した表記 | 3.3、B.2、B.5.1 |
| $\boldsymbol{\epsilon}$ | 接方向を作るための標準正規乱数ベクトル | 3.3、B.2 |
| $\boldsymbol{q}_{\mathrm{raw}},\boldsymbol{q}$ | 射影直後の接方向ベクトルと、その単位化後の方向 | 3.3、B.2 |
| $\mathcal{T}_{\boldsymbol{n}}$ | 入力方向 $\boldsymbol{n}$ における接空間 | B.2 |
| $\alpha$ | TGNの回転角。radian | 3.3、B.2、C.2 |
| $\delta$ | Fractional Shiftの移動量。チャネル単位 | 3.3、B.3、C.2 |
| $T_{\mathrm{TGN}},T_{\mathrm{FS}}$ | TGNと、再中心化・再正規化を含むFractional Shiftの変換 | 3.3 |
| $S_\delta$ | 再中心化・再正規化前のshift補間操作 | 3.3、B.3 |
| $\boldsymbol{y}$ | Shift補間後に平均を除いたスペクトル | 3.3、B.3 |
| $\boldsymbol{x}_{\mathrm{TGN}},\boldsymbol{x}_{\mathrm{FS}}$ | 各摂動の最終出力。各々 $T_{\mathrm{TGN}}(\boldsymbol{x}),T_{\mathrm{FS}}(\boldsymbol{x})$ に対応 | B.2、B.3 |
| $\zeta_j$ | Shiftで出力チャネル $j$ が参照する連続位置 | B.3 |
| $\iota_j^-,\iota_j^+$ | 連続参照位置の左側整数indexと、そのindexに1を加えた右側index | B.3 |
| $\bar{\iota}_j^-,\bar{\iota}_j^+$ | 上の参照indexをチャネル範囲の端点で切り詰めた値 | B.3 |
| $\omega_j^{\mathrm{FS}}$ | Shift補間で右側の値に掛ける重み | B.3 |
| $\Delta\lambda$ | 共通gridの隣接チャネル間の波長差 | B.3 |
| $\mathcal{A}_p$ | 学習時の適用抽選・順序・強度を含む追加摂動の入力変換 | 3.4、B.4 |
| $\widetilde{\boldsymbol{x}}_p$ | 追加摂動後のSNV入力 | 3.4 |

## 表現学習・クラスタリング

### Encoderと学習課題

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $\widetilde{\boldsymbol{x}}_p^{(a)}$ | 画素 $p$ の追加摂動後スペクトルから取り出したpatch $a$ | 3.4 |
| $E_{\mathrm{patch}},\boldsymbol{b}_{\mathrm{patch}}$ | Patchの線形埋め込みの重み行列とbias | 3.4 |
| $\boldsymbol{\pi}_a,\boldsymbol{t}_{p,a}$ | Patch位置の埋め込みと、埋め込み後の入力token | 3.4 |
| $f_\theta,\theta,\theta^\ast$ | 潜在への射影まで含むencoder写像、その学習parameter、固定した学習済みparameter | 3.4 |
| $\boldsymbol{h}_p,\boldsymbol{z}_p$ | Encoderの正規化前出力と、単位化後の潜在 | 3.4、B.5 |
| $\boldsymbol{z}^{\mathrm{full}}_p$ | 全patch可視で抽出した単位潜在 | 3.4、3.5 |
| $\mathcal{L}_{\mathrm{masked}},\mathcal{L}_{\mathrm{AE}}$ | Mini-batchについて集計した不可視チャネルMSEと全チャネルMSE | 3.4 |
| $\mathcal{L}_{\mathrm{one}}$ | 一画素の不可視チャネルMSE | B.4 |

### 入力差・潜在差・残差の対応

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $W_{\mathrm{dec}},\boldsymbol{b}_{\mathrm{dec}}$ | 線形decoderの重み行列とbias | 3.4、B.5 |
| $\widehat{\boldsymbol{x}}_p$ | Decoderが出力する復元スペクトル | 3.4、B.4、B.5 |
| $F$ | 固定モデルを全帯域可視で用い、単位潜在を返す写像。B.5では $\boldsymbol{z}_p=F(\boldsymbol{x}_p)=\boldsymbol{z}^{\mathrm{full}}_p$ と省略する | B.5.1 |
| $\boldsymbol{e}_p$ | 入力から全帯域可視の復元値を引いた残差 $\boldsymbol{x}_p-\widehat{\boldsymbol{x}}_p$ | B.5.2 |
| $d_{\mathrm{in}}(p,q),d_{\mathrm{lat}}(p,q),g_{pq}$ | 入力・潜在のcosine不類似度と、潜在／入力の弦距離比 | B.5.1 |
| $\Delta\boldsymbol{x},\Delta\boldsymbol{z},\Delta\boldsymbol{e}$ | 同じ画素対の入力差・潜在差・残差差 | B.5.2 |
| $U_{\mathrm{dec}},\Sigma_{\mathrm{dec}},V_{\mathrm{dec}},\sigma_{\mathrm{dec},i}$ | Decoder重みの左特異ベクトル行列、対角特異値行列、右特異ベクトル行列、第 $i$ 特異値 | B.5.2 |
| $\boldsymbol{u}_{\mathrm{dec},i},\boldsymbol{v}_{\mathrm{dec},i}$ | 左・右特異ベクトル行列の第 $i$ 列 | B.5.2 |

### クラスタ平均と摂動応答

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $\mathcal{P}_k,n_k,w_{kp}$ | 診断対象の第 $k$ 群の画素集合、その画素数、群内で和が1の非負重み | B.5.3 |
| $\overline{\boldsymbol{x}}_k,\overline{\boldsymbol{z}}_k,\overline{\boldsymbol{e}}_k$ | 同じ画素と重みによる入力・潜在・残差の算術平均。潜在平均は再単位化しない | B.5.3 |
| $a_k,\boldsymbol{c}^{\mathrm{emp}}_k$ | 潜在平均のnormと、そのnormが正の場合の単位平均方向 | B.5.3 |
| $\boldsymbol{z}_{p,\mathcal{A}}$ | 同一画素に追加摂動 $\mathcal{A}$ を与え、全帯域可視で得た単位潜在 | B.5.4 |
| $\boldsymbol{\mu}_p,\Gamma_p$ | 指定した摂動分布に関する潜在の平均と共分散。SNV前のスカラー平均 $\mu_p$ とは異なる | B.5.4 |
| $D$ | 摂動応答を推定するdraw数 | B.5.4 |
| $W_{\mathcal{H}},\boldsymbol{b}_{\mathcal{H}}$ | 固定した不可視チャネル集合 $\mathcal{H}$ に対応するdecoderの行・bias成分 | B.5.5 |
| $\boldsymbol{z}_{p,\mathcal{H},\mathcal{A}},\mathcal{L}_{p,\mathcal{H},\mathcal{A}}$ | 固定画素・固定maskに追加摂動を与えたときの単位潜在と、一画素のmasked MSE | B.5.5 |
| $\boldsymbol{\mu}_{p,\mathcal{H}},\Gamma_{p,\mathcal{H}}$ | 固定画素・固定maskのもとで、追加摂動に関して求めた潜在の平均と共分散 | B.5.5 |

### PCAとクラスタリング

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $\overline{\boldsymbol{x}}_{\mathrm{fit}},U_{\mathrm{PCA}}$ | PCAのfit対象画素の平均と、主成分方向を列に持つ行列 | 3.5 |
| $\boldsymbol{\psi}_p,\boldsymbol{\xi}_p$ | クラスタリングに渡す前の表現と、その単位化後の表現。B1の $\boldsymbol{\psi}_p$ はPCA score | 3.5 |
| $\boldsymbol{c}_k,\boldsymbol{g}_k$ | 単位クラスタ中心と、更新に使う所属fit画素の表現の和 | 3.5 |
| $d_{\cos},J$ | 単位表現間のcosine不類似度と、fit画素で平均した目的関数 | 3.5 |
| $\ell_p,\operatorname{Label}(u,v)$ | 画素のクラスタラベルと、元座標へ配置したラベルマップ | 3.5 |

## 評価指標：LLA・補正LLA

以下は[第4.4節](chapters/evaluation_protocol/metrics_aggregation.md#lla)で用いる記号である。一つの条件・クラスタ数・学習反復を固定した式とし、画素 $p,q$ を元画像上の整数座標と同一視する。設計文書のラベル $y_p$ は、本文では第3章と同じ $\ell_p$ と表す。

| 記号 | 意味・範囲 | 主な説明箇所 |
| --- | --- | --- |
| $m,\Omega_m$ | 試料の添字と、その試料の有効画素集合 | 4.4.1 |
| $B_{m,k},M_m$ | クラスタ $k$ の指示関数と有効画素マスク。背景・除外画素・画像外で0 | 4.4.1 |
| $r,h_r$ | 正方近傍の幅 $r\in\{3,5,9\}$ と、中心を除いた近傍カーネル | 4.4.1 |
| $\boldsymbol{u}$ | 二次元整数格子上の画素座標の差。成分は $u_1,u_2$ | 4.4.1 |
| $N_m,n_{m,k}$ | 試料の有効画素数と、そのうちクラスタ $k$ に属する画素数 | 4.4.2 |
| $P_m$ | マスク・クラスタ画素数を保つ帰無配置での異なる二画素の偶然一致確率。中心化射影行列 $P$ とは異なる | 4.4.2 |
| $\mathrm{LLA}_{r,m},\mathrm{LLA}^{\mathrm{adj}}_{r,m}$ | 有効近傍対のラベル一致率と、その偶然一致による補正値 | 4.4.1〜4.4.2 |

## 学習の進行・共通演算

### 学習の進行

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $e,t_{\mathrm{batch}},N_{\mathrm{step},e}$ | Epoch index、epoch内のbatch index、端数を除く1 epochのstep数 | C.3 |
| $\tau$ | Epoch単位の連続的な学習進行度 | C.3 |
| $\eta(\tau),\eta_{\max}$ | 学習率とそのpeak値。Referenceのproxy $\gamma_b$ とは区別する | C.3 |

### 線形代数・確率・その他の演算

| 記号 | 意味 | 主な説明箇所 |
| --- | --- | --- |
| $\mathbb{R}^{C}$ | 実数の $C$ 次元ベクトルの空間。行列は行数×列数で示す | 3.2 |
| $\boldsymbol{1},\boldsymbol{0}$ | 全成分1の $C$ 次元ベクトル、対象空間のゼロベクトル。ゼロベクトルの次元は各式の説明に従う | 3.2、B.1 |
| $I_C$ | 入力チャネル数 $C$ に対応する単位行列 | 3.3、B.1 |
| $\mathsf T,\lVert\cdot\rVert_2$ | 転置、L2 norm | 3.2、B.1 |
| $\pi$ | 円周率。位置埋め込みは対象の添字を付けたベクトル $\boldsymbol{\pi}_a$ | 3.3、B.2 |
| $\mathcal{N},\mathcal{U}$ | 正規分布と、指定区間の連続一様分布 | 3.3、B.2、C.2 |
| $\sim$ | 指定した分布に従って抽選することを表す | 3.3 |
| $\operatorname{Ind},\Pr,\mathbb{E}$ | 指示関数、確率、期待値。期待値の添字は抽選対象を表す | 4.4.1、B.4、B.5 |
| $\mathbb{Z}^{2},\lVert\cdot\rVert_\infty$ | 二次元整数格子と、成分の絶対値の最大値を取るnorm | 4.4.1 |
| $*,\langle f,g\rangle$ | 二次元離散畳み込みと、画素上の内積。対象関数は有限範囲の外で0とする | 4.4.1 |
| $\operatorname{tr}$ | 行列の対角成分の和 | B.5.4〜B.5.5 |
| $\lfloor\cdot\rfloor$ | その値以下の最大の整数を返す床関数 | B.3 |
| $\operatorname{arg\,max}$ | 対象の値が最大になる添字を選ぶ操作 | 3.5 |
| $\ominus,\mathrm{RemoveSmall}$ | 二値erosion、小連結領域の除去 | A.1 |

## 図中の表記と今後の追記

図3.1の旧ラベルH・W等の修正事項は[作図メモ](figures/analysis_workflow_notes.md)を参照する。第4章の残りと[幾何診断の実施計画](../docs/design/representation_geometry_diagnostics.md)で追加する記号は、定義の確定・原稿への導入時に追記する。
