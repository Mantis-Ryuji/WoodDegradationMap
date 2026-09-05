# 実験プロトコル

## 1. ステータス

**Fixed（第11.2節は実行前の確認・記録事項）**

前処理は200 Hzのみを使用する固定仕様とし、詳細は
[preprocessing.md](preprocessing.md)で管理する。
本書のraw SNVは、本番前処理済みの256次元SNVを表現変換せず直接使用するbaselineを指し、
センサのraw強度を意味しない。

## 2. 研究目的

近赤外ハイパースペクトル画像から得られるSNV後スペクトルについて、
ChemoMAEによる表現変換が、教師なし領域分割の空間的一貫性と、指定したスペクトル摂動に対する
安定性をどのように変えるかを検証する。各表現空間のクラスタ分離性は、その挙動を説明する
幾何学的診断として扱う。

外部の正解劣化ラベルや独立した劣化測定による定量評価は行わない。したがって、劣化領域の検出精度、
劣化度の推定精度、化学成分量の定量性能は本研究の検証対象に含めない。劣化との対応は、実験条件を
確定した後のラベルマップ、代表スペクトル、差スペクトルおよび試料情報に基づく探索的解釈とする。
教師なし指標が良好であることを、劣化との対応の証明に置き換えない。

中心となる問いは次の3点である。

1. 提案Aug-MAEはraw SNV、PCAおよび標準MAEと比べて、領域分割の空間的一貫性と摂動安定性を
   どのように変えるか。指標間にtrade-offがあるか。
2. 標準MAEのmasked reconstruction方式は、maskなし・全領域再構成のAEと比べて、どのような違いを与えるか。
3. Gaussian noiseと波長方向shiftは、MAEに対してそれぞれ単独または併用でどのような効果を与えるか。

mask率の最適化は主題にせず、提案条件の感度を確認する補助実験として扱う。

## 3. 実験単位とcross-validation

### 3.1 試料単位

- 分割・集計単位は、ファイル名の `KYOw...` で識別される試料とする。
- 同一 `KYOw` に属する画像、測定条件、画素は、必ず同じfoldへ割り当てる。
- 画素を独立なsplit単位として使用しない。
- metadataの参照元は `data/metadata/古材メタデータ.csv` とする。

`KYOw`が異なることだけを、生物学的・由来的な独立性の証明にしない。同一原材からの採取関係など、
より上位の依存関係が確認された場合は、split作成前に扱いを決定する。未確認の関係を推測で補完しない。
樹種、由来、測定条件について確認できる試料構成を報告し、未知樹種・未知産地・別装置への汎化を
このランダム5-foldだけから主張しない。

### 3.2 5-fold CV

- 試料をランダムな5-foldに分割する。
- 各foldは概ねtrain:test = 8:2となり、各試料は一度だけtestに現れる。
- 樹種による層化抽出は行わない。
- split生成時のseedを明示的に固定し、split manifestとともに記録する。
- すべての比較条件で同一のfoldを使用する。

各foldでは、表現学習、PCAのfit、Cosine-KMeansのfitをtrain試料だけで行う。
test試料は、trainで得た変換器、encoder、クラスタ中心を固定した状態で評価する。

### 3.3 3反復と乱数の管理

- 同一の5-fold splitを固定し、主比較とmask率補助実験をそれぞれ3反復する。split自体は作り直さない。
- 反復IDを1、2、3とし、対応するseed一覧をCV開始前に固定する。良いseedを選択する探索は行わない。
- 同じ反復IDを全条件で対応付け、モデル初期化、画素順序、mask、学習augmentation、PCAの乱数、
  KMeans初期化および評価摂動の乱数を区別して記録する。
- 条件ごとの乱数消費順の違いで比較用の入力や評価摂動が変わらないよう、乱数系列を分離する。
- trainに使う画素集合とtest画素集合は反復間で固定する。反復は、固定したsplitと画素集合に対する
  学習・クラスタリングの確率的変動を確認するものであり、データ分割の不確実性は評価しない。
- B0は表現学習を行わない。B1のPCAが決定的な設定ならfoldごとのfit結果を再利用し、確率的な設定なら
  反復ごとにseedを記録してfitする。B0/B1を含めKMeansは各反復でfitする。

主比較の学習5条件は$5\times5\times3=75$学習、追加mask率2条件は$2\times5\times3=30$学習で、
CVのニューラルネット学習は合計105回となる。Kの数だけ表現学習を繰り返さず、同一fold・条件・反復の
表現を全Kで共有する。PCA、KMeansおよび全体可視化用の学習はこの105回に含めない。

### 3.4 train画素の試料間均等化

- 各foldの各train試料から同数$q=8192$の有効画素を、試料内の一様ランダム・非復元抽出で選ぶ。
- $q$と抽出方法は全fold共通の固定条件とする。
- 同じfoldでは、全条件・全K・全反復で同じ画素座標集合を共有する。
- ChemoMAEの学習、PCAのfitおよびCosine-KMeansのfitは、この同じtrain画素集合を使用する。
- 反復内の画素順序と学習augmentationは確率的でよいが、試料ごとの元画素数による寄与の差は作らない。
- 有効画素数が$q$未満のtrain試料がある場合は、重複抽出や試料別の画素数変更で暗黙に補わず、
  実行前に$q$の設定を見直す。
- test評価は抽出せず、各test試料の全有効画素を用いる。背景・品質条件による除外は本番前処理に従う。

2026-09-05にユーザーが実行・共有した確認結果では、
`data/processed/preprocessing/200hz_snr10_linear256/sample_quality.parquet`の
49試料・3,902,746有効画素に対し、49/49試料で$q=8192$の非復元抽出が可能だった。
試料ごとの有効画素数は最小26,249、中央値73,259、最大161,735で、抽出率は5.07–31.21%となる。
この49試料を採用する場合、trainは39または40試料となり、
319,488または327,680画素/fold、312または320 batch/epoch、
800 epochで249,600または256,000予定更新/runとなる。
1024の倍数なので、この$q$では`drop_last=True`でも端数除外が発生しない。

この$q$は800 epoch・105学習の計算量と試料ごとの画素数を考慮して事前固定した値であり、
精度や希少領域の捕捉率を検証して最適化した値ではない。画素を増やしても独立な試料数は増えない。
試料集合や前処理データを変更した場合は、`sample_quality.parquet`の`saved_pixel_count`で
抽出可能性を再確認する。test評価には引き続き全有効画素を使う。

確認には`scripts/preprocess/check_sampling_pixels.py`を使用する。リポジトリrootからの実行例は次のとおり。

```powershell
.\.venv\Scripts\python.exe scripts/preprocess/check_sampling_pixels.py --q 8192
```

既存の`sample_quality.parquet`の試料ID・有効画素数だけを読み、全試料の画素数、要求抽出率、
不足画素数、最小・中央値・最大画素数を標準出力へ表示する。全試料で抽出可能なら終了code 0、
不足があれば終了code 1とする。これは画素数の確認であり、実際の画素抽出・split作成・
スペクトル網羅性の評価は行わない。生成物や前処理データを変更しない。

抽出seed、$q$、試料IDおよび`pixel_row_col`に対応する抽出座標をmanifestへ保存する。
空間近傍を使うLLAは、抽出したtrain画素上では計算しない。各試料から同数を使うことは試料の重みを
揃える操作であり、樹種や由来の構成比を均等にする操作ではない。

## 4. 主比較条件

主比較ではmask率を50%に固定する。通常の推論入力にはaugmentationを適用せず、
学習時の各augmentationの適用確率は、有効な条件においてそれぞれ0.5とする。
LFR評価では別途、固定モデルに対する評価摂動を明示的に生成する。

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
本書の「標準MAE」M00は、本研究で共通のChemoMAE構成を用いたaugmentationなしの対照を指す。
原MAE論文の画像用architectureを再現した条件という意味ではない。

### 4.1 条件間で固定するもの

- ChemoMAE系条件のencoder、decoder、latent次元および学習budget
- 学習データ、fold、前処理および画素mask
- Cosine-KMeansの実装、初期化方針および共通クラスタ数
- 評価対象の試料および画素
- augmentationの強度とseed方針

augmentation強度は結果を見て条件ごとに変更せず、CV開始前に固定する。
強度sweepは行わない。

#### 4.1.1 表現次元とL2正規化（Fixed）

B1のPCA成分数と、A0・すべてのMAE条件の`latent_dim`を16に固定する。
mask率補助実験および全体学習でも同じ16次元を使用する。B0は256次元の入力空間baselineとする。

| 条件 | クラスタリング・評価へ渡す表現 |
| --- | --- |
| B0 | SNV 256次元 → 行ごとのL2正規化 |
| B1 | trainでfitしたPCAによる16次元得点 → 行ごとのL2正規化 |
| A0・MAE系 | 全可視encoderのCLS出力 → `to_latent`で16次元 → L2正規化 |

正規化前の表現を$h$として、各画素で$z=h/\lVert h\rVert_2$とする。
PCAでは**次元削減後**に正規化し、PCA入力の正規化だけで代用しない。
PCAの平均・基底などのfit対象は第3.4節のtrain画素集合に限定する。L2正規化自体は画素ごとの操作で、
test集合から統計量を推定しない。clean・摂動後・train・testに同じ変換順序を適用する。
PCAは`sklearn.decomposition.PCA(n_components=16)`を使用し、成分数以外は既定設定を採用する。
[scikit-learnのPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)に従い、
train画素の波長ごとの平均で中心化し、`whiten=False`、`svd_solver="auto"`、`copy=True`とする。
SNVに加えて波長ごとの標準偏差で割るautoscalingは行わない。SNVの画素内中心化と、
PCAのtrain画素集合に基づく波長ごとの中心化は別の操作である。
whiteningは成分ごとの分散を揃える操作であり、後段のL2正規化とは区別する。

中心化したスペクトルを共分散に基づいてPCAへ渡す設定を、通常の線形baselineとして用いる。
スペクトルデータでは同じ単位の変数を扱うことが多く、autoscalingで小さな変動やnoiseを拡大しない
平均中心化が一般的な選択肢となる（[Eigenvectorのケモメトリクス解説](https://eigenvector.com/wp-content/uploads/2022/12/Chemometrics_Crash_Course_Part_2.pdf)）。
PCA得点のL2正規化は、本研究のcosineクラスタリングに接続するための後処理と位置づける。

その他の引数も既定値（`tol=0.0`、`iterated_power="auto"`、`n_oversamples=10`、
`power_iteration_normalizer="auto"`、`random_state=None`）とする。使用versionと実際に選ばれたsolverを保存する。
`auto`は入力shapeとlibrary versionで決まる計算方式の選択であり、CV指標による手法選択ではない。
乱数を使うsolverになった場合は、第3.3節に従いPCA用seedでNumPy乱数状態を制御してfitし、
他の乱数系列と分離する。`random_state=None`を乱数管理不要という意味では扱わない。

ChemoMAEでは`latent_normalize=True`を明示し、学習時も正規化した16次元潜在をdecoderへ入力する。
抽出するのは最終CLS出力を`to_latent`へ通した潜在であり、256次元CLSやpatch平均を代用しない。
全可視抽出には`Extractor`を使用するか、`model.eval()`と全要素Trueの`visible_mask`を明示して
`model.encoder(x, visible_mask)`を呼ぶ。`model.eval()`だけでは`ChemoMAE.forward()`の
ランダムmask生成は無効にならない。通常の抽出では`augmenter=None`とし、LFR用の共通摂動は別途生成する。

Cosine-KMeansはfit・predictの両方で入力をL2正規化し、更新した中心も正規化する。
したがって、正常な非ゼロ表現ではPCA後の明示的なL2正規化は既存のcosine幾何と整合する。
libraryのepsilonによる除算保護はゼロnormを有効にする規則ではない。正規化前後の非有限値、ゼロnorm、
数値的に単位化できない表現は評価規約第2.4節に従って記録し、画素を無言で除外しない。
表現抽出・評価の精度と正規化の数値設定は第4.1.4節に従う。

16次元と単位normを揃えても、PCAとChemoMAEが作る画素間の角度や分布は異なる。
silhouetteを各表現空間の診断とする方針は維持する。SNV入力・再構成target・保存済み前処理データは、
この後段の単位化とは区別する。B0の単位化も、元のSNVで定義されるcosine距離を変えるものではない。

#### 4.1.2 ChemoMAEの構成と初期化（Fixed）

ユーザー提示のencoder構成と確認済みの線形1層decoderを共通構成とする。
16次元出力・全可視抽出は前節に従う。dropoutは0.0、初期化はChemoMAE v0.2.1の既定動作を採用する。

| 設定 | 採用値 |
| --- | --- |
| `seq_len` | 256（本番前処理と共通） |
| `d_model` | 256 |
| `nhead` | 8（headあたり32次元） |
| `num_layers` | 8 |
| `dim_feedforward` | 1024 |
| `dropout` | 0.0。全学習条件で共通 |
| `decoder_num_layers` | 1（線形写像、Fixed） |
| `latent_dim` / `latent_normalize` | 16 / True（Fixed） |
| `n_patches` | 16（1 patchは連続16チャネル） |

この構成ではA0、mask率25%・50%・75%の`n_mask`はそれぞれ0、4、8、12となる。
全可視ではCLSを含め17 token、主比較の50% maskでは9 tokenをencoderへ入力する。

[ChemoMAE v0.2.1のモデル実装](https://github.com/Mantis-Ryuji/ChemoMAE/blob/942804a176750e4f79ee530ca650e0e317efbf90/src/chemomae/models/chemo_mae.py)
では`decoder_num_layers=1`は`Linear(16, 256)`であり、Transformer decoderの1層ではない。
この場合、再構成は$\hat{x}=Wz+b$で、出力集合は高々16次元のアフィン部分空間に含まれる。
さらに潜在の単位norm制約がある。非線形encoderから線形に復元可能な表現を学ぶ設計として扱い、
decoderまで非線形なAEや原論文のViT-MAEと同一とは記述しない。
単純な線形復元headを用いる方針は[SimMIM](https://arxiv.org/abs/2111.09886)と共通する。
ただし[SimMIMの実装](https://github.com/microsoft/SimMIM/blob/main/models/simmim.py)は空間位置ごとの
encoder特徴から画素を復元する。本研究はCLS由来の単一の16次元球面潜在から全スペクトルを復元し、
このbottleneckを表現学習の制約として明示する。SimMIMの検証結果を本研究の性能保証には用いない。

この構成のパラメータ数は、実装の各層からの計算でencoder 6,331,152、decoder 4,352、合計6,335,504である。
モデルを実行して得た計測値ではない。dropout=0.0の最適性を検証したという意味ではない。

初期化は、導入済みChemoMAE v0.2.1とPyTorch 2.13.0のソースで次の動作を確認し、そのまま採用する。

- CLS tokenと学習可能な位置埋め込みはChemoMAEが`nn.init.trunc_normal_(..., std=0.02)`で初期化する。
- patch projection、`to_latent`、線形decoderなどは`nn.Linear`の既定初期化を使う。
  重みとbiasは入力次元$d_{\mathrm{in}}$に対する$U(-1/\sqrt{d_{\mathrm{in}}},1/\sqrt{d_{\mathrm{in}}})$となる。
- attentionの結合QKV重みはPyTorchのXavier uniform、attentionのbiasは0、LayerNormはweight=1・bias=0とする。
- `nn.TransformerEncoder`は初期化された1つのencoder layerをdeepcopyして8層を作る。
  対応するパラメータは層間で同じ初期値から始まるが、別のパラメータであり重み共有ではない。
  ChemoMAEは複製後に層ごとの再初期化を行わない。利用側でも追加の再初期化を行わない。

用途別seedを設定してからモデルを構築する。同じfold・反復では同じ構築seedを全学習条件に対応付ける。
原MAEの画像用モデルに対する初期化を上書きして混ぜない。

#### 4.1.3 参照実装とaugmentation・クラスタリングの確認

参照版はプロジェクトが固定するChemoMAE v0.2.1、commit
[`942804a176750e4f79ee530ca650e0e317efbf90`](https://github.com/Mantis-Ryuji/ChemoMAE/commit/942804a176750e4f79ee530ca650e0e317efbf90)
とする。導入済みpackageと同commitのモデル・augmentation・抽出・optimizer・Trainer・
Cosine-KMeans・正規化helperのソース内容が一致することを、読み取りによって確認した。
これは実験pipelineの動作検証を意味しない。

[SpectraAugmenterの実装](https://github.com/Mantis-Ryuji/ChemoMAE/blob/942804a176750e4f79ee530ca650e0e317efbf90/src/chemomae/training/augmenter.py)
を使用し、noise角度はユーザー指定の$U(0,2.5^\circ)$、shiftおよびその他の操作設定は既定値で固定する。
以下は参照用の候補ではなく、主比較・mask率補助実験・全体学習に共通の採用設定である。

| 項目 | 採用設定（Fixed） |
| --- | --- |
| shift | $\delta\sim U(-2,2)$、チャネルindex単位のfractional shift。`shift_delta_range=(-2.0, 2.0)`（既定値） |
| shiftの補間・端点 | 線形補間。参照indexを端点へclampし、範囲外は端点値を延長する。循環shiftではない |
| noise | Gaussian乱数から平均ゼロ・入力に直交する方向を作り、その方向へ指定角度だけ回転 |
| noise角度 | $\theta\sim U(0,2.5^\circ)$。`noise_angle_deg_range=(0.0, 2.5)`。各チャネルへ独立なGaussian雑音を加算する方式とは異なる |
| 適用確率 | 学習時は有効な操作ごとに0.5、無効な操作は0。画素ごとに適用を抽選する |
| 操作順序 | `shuffle_order_per_batch=True`。2操作の順序をbatchごとにランダム化（既定値） |
| 再中心化・norm | `recenter_after_each_op=True`、`renorm_to_input_norm=True`。各操作後に画素内平均を0、normを操作前の値へ戻す（既定値） |
| 数値安定化 | `eps=1e-8`（既定値） |

noise強度の最適化や追加の強度ablationは行わない。noiseとshiftの有無による第4節の2×2 ablationは維持する。
LFRにも同じ角度分布・shift幅・操作設定を使用し、評価対象の操作の適用確率だけを1にする。
これらは実行可能な共通条件として事前固定した値であり、測定装置の誤差分布を同定した値や、
予備実験で最適化した値とは記述しない。

本研究の入力軸は等間隔の波長gridなので、shiftの単位は256点grid上のチャネルindexと記す。
nmへの換算には保存済み波長間隔を使用し、波数軸の等間隔shiftとは記述しない。
noiseのGaussianは方向の生成方法を指し、回転角の分布や最終的な加算残差がGaussianという意味ではない。

[Cosine-KMeansの実装](https://github.com/Mantis-Ryuji/ChemoMAE/blob/942804a176750e4f79ee530ca650e0e317efbf90/src/chemomae/clustering/cosine_kmeans.py)
の既定アルゴリズム・停止設定を採用する（Fixed）。実装から確認した値は次のとおりである。

| 設定 | 採用値・動作 |
| --- | --- |
| 初期化 | cosine不類似度を用いたk-means++型の初期化を1 fitあたり1回。追加のrestartや最良run選択なし |
| 最大反復 | `max_iter=500` |
| 許容値 | `tol=1e-4` |
| 停止規則 | 平均cosine不類似度の前回との差について、相対変化$<10^{-4}$または絶対変化$<10^{-7}$で停止。相対変化の分母は前回値の絶対値$+10^{-12}$ |
| 空クラスタ | 現在の最近中心から遠いtrain画素で中心を置き換える |
| 正規化・精度 | 入力・更新中心を行ごとにL2正規化。内部FP32、正規化の`eps=1e-6` |
| device | `device="cuda"`（既定値） |

`n_components`は第6節の各K、`random_state`は第3.3節の用途別seedを指定する。
これらは既に決めたK依存性と3反復のための指定であり、既定値のK=8・seed=42で全runを上書きしない。
libraryに`n_init`引数はない。3反復と、1 fit内の初期化1回を区別する。

#### 4.1.4 表現抽出・評価の数値設定（Fixed）

PCAのfit・transform、ChemoMAEの全可視抽出、評価摂動の生成、L2正規化、Cosine-KMeansおよび
評価指標の連続値の計算はFP32とする。入力・モデルパラメータ・抽出表現はFP32に揃え、
特徴抽出には`ExtractorConfig(amp=False)`を明示する。FP16/BF16で計算した結果を最後にFP32へcastする
方式で代用しない。評価ではTF32も無効にする。学習は第4.2節のFP16 AMPを維持する。

`silhouette_samples_cosine_gpu`は`dtype=torch.float32`を明示し、数値保護はlibrary既定値を採用する。
ChemoMAEの潜在正規化とsilhouetteは`eps=1e-12`、Cosine-KMeansの正規化helperは`eps=1e-6`、
SpectraAugmenterは`eps=1e-8`である。B0/PCA後の明示的な正規化には同じKMeans helperを用いる。
各epsilonは除算保護であり、ゼロnormを有効なcosine入力と認める閾値ではない。
非有限値・ゼロnormなどの扱いは評価文書第2.4節の既存規約に従う。
単位normの誤差は診断値として保存し、結果に合わせた新しい画素除外閾値を作らない。

離散ラベル・画素数・一致件数・contingencyなどは整数で保持し、FP32指定を理由に整数の計数を
浮動小数へ置き換えない。seed、実際の演算設定とlibrary versionは第11.2節に記録する。
GPUの並列reductionによる微小な非決定性まで消えたとは主張しない。

### 4.2 学習設定（Fixed）

学習設定は、[MAE論文](https://arxiv.org/pdf/2111.06377)のTable 8およびTable 1のablationに対応する
事前学習recipeを採用する。公式PyTorch実装はcommit
[`efb2a8062c206524e35e47d04501ed4f544c0ae8`](https://github.com/facebookresearch/mae/commit/efb2a8062c206524e35e47d04501ed4f544c0ae8)
を参照する。batch sizeはユーザー指定の1024、勾配蓄積はなしとする。

| 設定 | 採用値 |
| --- | --- |
| optimizer | AdamW |
| betas | $(\beta_1,\beta_2)=(0.9,0.95)$ |
| epsilon / AMSGrad | $10^{-8}$ / 無効 |
| weight decay | 0.05。biasと正規化層のパラメータは0 |
| batch size | 1024スペクトル / GPU |
| GPU数 | 単一GPUを前提とする（`world_size=1`） |
| gradient accumulation | `accum_iter=1`。各batchで更新し、勾配を蓄積しない |
| effective batch size | 1024 |
| base learning rate | $1.5\times10^{-4}$。256のeffective batchに対する基準値 |
| peak learning rate | $6.0\times10^{-4}$。下記の線形スケーリングによる |
| training epochs | 800 |
| warmup | 最初の40 epochで0からpeak learning rateまで線形増加 |
| warmup後のschedule | 残り760 epochでhalf-cycle cosine decay。restartなし |
| minimum learning rate | 0 |
| learning rate更新 | 各batchの処理前に、epoch内の進捗を含めて更新 |
| gradient clipping | 使用しない |
| drop path | 0 |
| 学習精度 | CUDA AMP（FP16 autocast）と動的GradScaler。モデルパラメータはFP32 |
| train DataLoader | epochごとのshuffle、`drop_last=True` |
| EMA | 使用しない（`use_ema=False`） |
| checkpoint選択 | 800 epoch完了時のraw weights（`last_model.pt`）。EMA weightsを使用しない |

この設定をA0、M00、M10、M01、M11およびM11のmask率25%・75%へ共通適用する。
全体可視化用のM00・M11にも同じ800 epochのrecipeを使用する。
参照は[公式PRETRAIN.mdの明示設定](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/PRETRAIN.md)
を優先する。`main_pretrain.py`の引数既定値には400 epoch・base lr $10^{-3}$が含まれるが、
それらを本研究の学習設定へ転記しない。

effective batch sizeを$B_{\mathrm{eff}}$として、学習率は

$$
B_{\mathrm{eff}}=1024\times1\times1=1024,\qquad
\eta_{\max}=1.5\times10^{-4}\frac{B_{\mathrm{eff}}}{256}=6.0\times10^{-4}
$$

とする。原論文のeffective batch size 4096を勾配蓄積で再現する設定ではない。
複数GPUへ移行する場合は、GPU数・effective batch size・学習率を一体として再指定する。
batch size 1024でGPUメモリに収まることは未検証であり、メモリ不足を理由に実装がbatch sizeや
accumulationを暗黙に変更しない。

epoch進捗を$u=e+j/S_f$（$e$は0始まりのepoch、$j$は0始まりのbatch index）として、
[公式scheduler](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_sched.py)
と同じく

$$
\eta(u)=
\begin{cases}
\eta_{\max}\,u/40, & 0\le u<40 \\
\dfrac{\eta_{\max}}{2}\left[1+\cos\left(\pi\dfrac{u-40}{800-40}\right)\right], & 40\le u\le800
\end{cases}
$$

を用いる。epoch単位の段階的なschedulerに置き換えない。

fold $f$のtrain試料数を$M_f$とすると、抽出集合の画素数と1 epochのbatch数、予定更新回数は

$$
N_f=M_fq,\qquad S_f=\left\lfloor\frac{N_f}{1024}\right\rfloor,\qquad U_f=800S_f
$$

となる。$S_f\ge1$を実行前に確認する。同じfold内では全条件・全反復で同じ$S_f$と$U_f$を用いる。
train試料数が異なるfold間や全体学習では更新回数が異なるため、実際の$N_f$、$S_f$、$U_f$を保存する。
AMPによる更新skipが生じた場合は、skip回数と実更新回数も記録する。

`drop_last=True`では、shuffle後の端数batchをそのepochのニューラルネット学習から除く。
同じfold・反復・epochでは全学習条件で同じ順序と端数の除外を共有する。試料ごとの同数保証は
抽出集合に対するものであり、端数除外後に毎epoch厳密に同数の画素が寄与するとは記述しない。
PCAとKMeansのfitは端数を除かず、抽出集合全体を使う。test評価でも端数を落とさない。

optimizer・weight decay groupingは
[main_pretrain.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py)、
AMPとbatchごとのscheduler呼び出しは
[engine_pretrain.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/engine_pretrain.py)
および[util/misc.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py)
に基づく。epsilonとAMSGradは[PyTorch AdamWの既定値](https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html)
を明示したものである。

本節は事前学習のoptimizer・schedule・batch処理・精度設定を固定する。スペクトルのSNV、
第4節のmask率・augmentation条件およびA0/MAEのloss対象は本研究の仕様に従う。
ImageNetの画像augmentation、RGB正規化、追加のpatch内target正規化（`norm_pix_loss`）は導入しない。
ViT-Lなどのモデル構成・初期化を一括して転記せず、ChemoMAEの構成は第4.1.2節に従う。
dropoutと初期化も第4.1.2節でFixedとした。16次元出力と全可視抽出は第4.1.1節に従う。

ChemoMAE v0.2.1の
[Trainer](https://github.com/Mantis-Ryuji/ChemoMAE/blob/942804a176750e4f79ee530ca650e0e317efbf90/src/chemomae/training/trainer.py)
を使う場合は`amp_dtype="fp16"`、`grad_clip=None`、`use_ema=False`を明示する。
既定のBF16・clipping 1.0・EMA有効では本節と一致しない。独立runは`resume_from=None`で開始し、
再開時は同じrunのcheckpointを明示する。別条件・別反復からの自動resumeを行わない。

[optimizer・scheduler helper](https://github.com/Mantis-Ryuji/ChemoMAE/blob/942804a176750e4f79ee530ca650e0e317efbf90/src/chemomae/training/optim.py)
にも差がある。`build_optimizer`はbias・LayerNormに加えてCLS tokenと学習可能な位置埋め込みを
weight decayから除外する。既存の`build_scheduler`はwarmupに`step + 1`を使い、
Trainerはbatch更新後にschedulerを進めるため、本節の0から始まる学習率列と完全一致しない。
本節のrecipeに合わせるには利用側のparameter groupingと学習率列の調整が必要となる。
既定helperをそのまま使用して原MAEと完全に同じ設定だと記述しない。本文書の更新では学習コードを変更しない。

### 4.3 計画比較

| 比較 | 検証する内容 |
| --- | --- |
| M11 vs B0 | 提案Aug-MAEと入力空間baselineの直接比較 |
| M11 vs B1 | 提案Aug-MAEと線形次元削減baselineの直接比較 |
| M11 vs M00 | 標準MAEに2種類のaugmentationを併用する効果 |
| M00 vs B0 | 標準MAEと入力空間baselineの比較 |
| M00 vs B1 | 標準MAEとPCAの比較 |
| M00 vs A0 | maskなし・全領域再構成からmasked reconstruction方式へ変更する効果 |
| M10 vs M00 | Gaussian noiseの単独効果 |
| M01 vs M00 | shiftの単独効果 |
| M11 vs M10 | shiftの追加効果 |
| M11 vs M01 | Gaussian noiseの追加効果 |

M11 vs B0、M11 vs B1、M11 vs M00を主要な計画比較とし、残りは構成要素を説明するablationとする。
M00 vs B1では出力次元を16、normを1に揃えるが、学習目的と変換器が異なるため、非線形性だけの効果とは解釈しない。
M00 vs A0ではmaskの有無とloss対象が同時に異なるため、mask率だけの効果とは解釈しない。
augmentation付きAEは本計画に含めず、augmentationとmask modelingの組合せに固有の優位性は主張しない。

M00、M10、M01、M11は、Gaussian noiseの有無とshiftの有無による2×2要因計画として扱う。
各主評価指標について、同じ試料・K・反復における次の交互作用contrastを報告する。

$$
(M11-M10)-(M01-M00)
$$

LLAとLFRは改善方向が逆である。contrastは指標の元の尺度で示し、正負の意味を明記する。
交互作用はこの指標・この強度設定に対する記述であり、augmentationの一般的な相乗効果とは断定しない。

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
fold、共通K集合、学習budget、3反復のseed一覧および評価指標は主比較と同一にする。

Gaussian noiseのみ、shiftのみの条件ではmask率sweepを行わない。

## 6. 共通クラスタ数とK依存性

### 6.1 結果に依存しない事前固定

全fold・全条件・全反復で使用する共通K集合を、次の7点に固定する。

$$
\mathcal{K}=\{2,4,6,8,10,12,14\}
$$

全試料raw SNVのelbowによるK校正は行わない。test試料のスペクトル分布、CV指標、ラベルマップ、
または劣化の解釈を用いて集合を変更しない。

観察上の組織・表面状態は互いに重なり得るため、列挙した要素数を正解クラスタ数とは仮定しない。
粗い分割から細かい分割までのK依存性を、2から14まで2刻みの等間隔で確認するための事前設定とする。
この範囲が実際の組織や異常をすべて分離できることを保証するものではない。

### 6.2 代表表示と全Kの報告

主表および全体可視化に使用する代表表示値を、共通集合の中央の値である8に固定する。

$$
K_0=8
$$

この$K_0$は表示上の代表値であり、データから推定した真のクラスタ数や最適Kを意味しない。
条件ごと、foldごと、評価指標ごとに都合のよいKを選ばない。

- $K_0$について主評価、診断および計画比較を表形式で報告する。
- $\mathcal{K}$全体の曲線と計画比較の差を併記し、代表Kだけで結論を決めない。
- 改善の方向がKによって変わる場合は、どのKで変わるかを明示する。
- K方向の平均、最大値などを手法の総合scoreにしない。

K依存性は、主実験で得た同じ表現を使う補助解析とする。追加の表現学習は行わず、各Kの
クラスタリングと評価を実施する。Kの集合内で傾向が保たれることを、集合外への保証とは扱わない。

## 7. fold内の処理

各foldについて、固定したtrain/test試料と画素集合を準備する。各条件・各反復について次を行う。

1. train試料だけを用いて、必要な表現変換器またはChemoMAEを学習する。
2. train/testの木材領域画素から、第4.1.1節の次元・全可視抽出・L2正規化に従って表現を抽出する。
3. 共通集合内の各Kについて、train表現だけでCosine-KMeansをfitする。
4. 固定したtrain centroidsを用いて、cleanなtest表現と同一画素の摂動後表現へラベルを割り当てる。
5. test試料に対して主評価指標と補助診断を計算する。
6. 試料・K・反復単位の値、foldおよび全out-of-fold試料の結果を保存する。

全反復の完了後に、同一fold・条件・Kの反復間でラベル分割のARIを計算する。
K間・反復間・条件間でクラスタ番号の意味が共通であるとは仮定しない。

B1のPCAはfoldごとにtrain試料だけでfitし、同じ変換をtest試料へ適用する。
PCA次元は16に固定し、寄与率やCV結果を見て変更しない。射影後にL2正規化する。

checkpointは、第4.2節の800 epoch完了時の最終時点を使用する。test再構成loss、test指標、
可視化で停止時点を選ばない。early stoppingまたは別のcheckpoint選択を導入する場合は、
outer train内の試料単位validationと選択規則を含む設計変更が必要となる。

## 8. 条件選択の方針

- CV結果から単一の「best条件」を選定しない。
- 主評価はLLAとlabel flip rateとし、cosine-silhouetteは各表現空間の幾何学的診断とする。
- 補正LLA、occupancyおよび反復間ARIを補助診断として報告する。どの指標も劣化精度の代用にしない。
- 各指標を恣意的に合成した総合scoreを作らない。
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

3反復は確率的変動の確認であり、seedの最適化または良好なrunの選別ではない。

## 10. 実行順序

1. 固定済みの200 Hz本番前処理datasetと診断図を生成・確認する。
2. 第11.2節の確認・記録を行う。固定済みの$q$、共通$\mathcal{K}$と代表表示$K_0$を含む実行configを作成する。
3. 試料単位の共通5-fold splitとtrain画素のmanifestを作成する。
4. 主比較7条件を3反復で評価する。
5. M11のmask率補助実験を同じ3反復で行う。
6. pairedな条件差、K依存性、mask率依存性および反復間安定性を集計する。
7. B0、B1、M00、M11を全試料でfitまたは学習し、事前指定した$K_0$でクラスタリングする。
8. ラベルを整列したうえで、スペクトル、ラベルマップ、潜在空間を解釈する。

## 11. 実験条件の確定状況と実行前の記録

### 11.1 実験条件の確定状況（Fixed）

PCA、モデル構成・初期化、augmentation、Cosine-KMeans、抽出・評価のFP32、LFRの$R=5$はFixedとした。
共通抽出数$q=8192$画素/試料と一様ランダム・非復元抽出もFixedとし、現行49試料すべてで
抽出可能なことを確認した（第3.4節）。これまでOpenとしていた主実験の条件選択は解消した。
既に決まった条件を未決定事項として再掲しない。

### 11.2 データ確認と実行記録

次は新しい実験系列や研究仮説の選択ではなく、確定した設計を再現可能に実行するための確認・記録事項である。
設定上のOpenから分けて管理するが、必要な確認・記録を省略してよいという意味ではない。

| 項目 | 実行前に確認・固定・保存する内容 |
| --- | --- |
| 試料集合と分割 | production manifestとmetadataの対応、採用試料数、上位の採取関係の確認状況。新たな依存関係が判明した場合はsplit前に扱いを決める |
| 乱数 | split・画素抽出・3学習反復・KMeans・評価摂動などの用途別seed一覧。数値は実装時に事前固定し、結果を見て選び直さない |
| manifest | fold割当、抽出座標、反復IDとseedの対応、全体学習用の画素集合 |
| 実行環境 | GPU機種、実際に使用したlibrary version、学習AMP・抽出評価FP32・評価TF32無効の設定、seedと決定性の実設定、PCAの実solver、library既定epsilonと単位norm誤差 |
| 学習処理 | 第4.2節のrecipeとの整合、run別のcheckpoint、予定・実更新回数、AMP skip回数、実行時間 |

本文用の表示例の基準・試料IDは可視化文書、任意の形状診断を採用する場合の定義は評価文書で管理する。
これらはPCA・augmentation設定の確定とは別に残る。現時点ではCVを実行できる完成configではない。

学習設定は第4.2節でFixedとした。固定済み$q=8192$と確定したsplitから各foldの予定更新回数$U_f$を計算し、
同一fold内の条件間で共通にする。800 epochはfold間でも共通とし、異なるfoldの更新回数を揃えるために
epoch数を変えない。mask率により実際の計算量や実行時間が異なるため、等計算量の比較とは記述せず、
実行時間を併記する。設定変更が必要になった場合は、結果を見た事後調整と事前の設計変更を区別し、
異なるrunを同一条件として混合しない。
