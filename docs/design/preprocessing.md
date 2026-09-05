# 前処理仕様

## 1. ステータス

**Fixed**

2026-09-06ユーザー決定: SNV前の補間反射率が1帯域でも負の画素は、train・test共通で背景とする。
`production_v1` として再生成し、ユーザーが旧生成ディレクトリを削除した。raw原本は変更していない。

本書は、後段の解析と可視化に渡す200 Hz NIR-HSIの本番前処理を定義する。
前処理は全試料および全実験で共通とし、cross-validationやモデルの結果を見て変更しない。

本書で「raw SNV」と呼ぶ場合、それはセンサのraw強度ではなく、本前処理で生成した256次元SNVを
表現学習せず直接用いるbaselineを意味する。

## 2. 確定仕様

| 項目 | 仕様 |
| --- | --- |
| 入力 | 200 Hz HSIおよび対応する200 Hz white/dark reference |
| 入力shape | 高さは試料ごとに可変、幅320 px、256 bands |
| mask score | 200 Hz生強度の全256 band和 |
| 二値化 | 試料別3-class Multi-Otsuのclass 1とclass 2を結合 |
| mask後処理 | erosion radius 1、その後にmin object size 1、connectivity 2 |
| 反射率 | 列ごとのwhite/dark referenceを用いた変換 |
| 波長cutoff | reference由来SNR proxyが10以下となる終端連続区間を自動除外 |
| 補間 | 保持波長範囲内の等間隔256点への線形補間 |
| 正規化 | 補間後に画素単位のSNV、標本標準偏差、`ddof=1` |
| 負の反射率 | 補間後・SNV前に1帯域でも0未満なら、その画素全体を解析上の背景とする |
| 数値型 | 保存時`float32` |
| 追加加工 | clip、平滑化、微分、baseline補正、外挿を行わない |
| スペクトルsource | 200 Hzのみ。030/200 Hz融合は行わない |

`min_object_size=1`は、現行実装では追加の小領域除去を実質的に行わない設定である。
ただし処理順序を固定し、将来の仕様変更時にも形態処理の意味が曖昧にならないよう記録する。

## 3. 入力契約

本番CLIが使用する入力は`data/raw/`内の次のファイルである。

- `200hz_<sample_id>.hdr`および`200hz_<sample_id>.raw`
- `200hz_white.hdr`および`200hz_white.raw`
- `200hz_dark.hdr`および`200hz_dark.raw`

`<sample_id>`は`KYOw...`形式とする。本番前処理は030 Hz HSIおよびRGB画像を必要としない。
rawデータは原本として扱い、変更、改名、移動または上書きしない。

処理開始前に次を検証する。

- headerとraw本体の対応およびraw byte数
- HSIが3次元であり、高さが正、幅が320、band数が256であること
- sample IDが重複していないこと
- white/dark referenceが$1 \times 320 \times 256$であること
- white、darkおよび全試料で波長軸と`x start`が一致すること
- 波長が有限かつ単調増加であること

試料間で画像高さが異なることは許容する。200 Hzだけを使用するため、030 Hzとの共通高さへの切り詰めや
画像registrationは行わない。

## 4. 実行方法

リポジトリrootから次を実行する。

```powershell
uv run python scripts/preprocess/run_production_preprocessing.py
```

既定の出力先は次のとおりである。

- 解析データ: `data/processed/production_v1/`
- 前処理レポート: `outputs/preprocessing/production_v1/`

両出力先は新規または空でなければならない。既存runを暗黙に上書きしない。
CLIで変更できるのは入出力先とmemory用chunk sizeだけであり、SNR閾値、mask条件、補間点数などの
研究上の固定値はoptionとして公開しない。

## 5. 処理手順

### 5.1 200 Hz referenceによる自動cutoff

whiteとdarkの差を、列$x$、band$b$について

$$
Q_b(x) = W_{200}(x,b) - D_{200}(x,b)
$$

とする。band別のreference由来SNR proxyは

$$
\operatorname{SNR}^{\mathrm{ref}}_{200}(b)
=
\frac{\operatorname{mean}_x Q_b(x)}
{\operatorname{std}_{x,\mathrm{ddof}=0} Q_b(x)+\varepsilon},
\qquad \varepsilon=10^{-12}
$$

で定義する。これはreferenceの列方向のばらつきに対する分母安定性のproxyであり、反復撮像から求める
時間方向SNRではない。

低SNR flagを

$$
L_b =
\begin{cases}
1, & \operatorname{SNR}^{\mathrm{ref}}_{200}(b) \le 10
     \text{ or non-finite} \\
0, & \text{otherwise}
\end{cases}
$$

とする。最終bandまで連続する$L_b=1$の区間を除外し、その直前までを保持する。
終端区間より前にも$L_b=1$のbandがある場合は、単一の上限cutoffでは表現できないためfail fastする。
試料スペクトル、劣化情報、CV結果またはモデル結果はcutoff決定に使用しない。

現在のreferenceでは、次が自動選択される。

- 保持: band 0–221、913.10–2305.59 nm、222 bands
- 除外: band 222–255、2311.85–2518.47 nm、34 bands
- 図示する境界: band 221と222の中点、2308.72 nm

これらのband番号や波長を処理コードへ直接固定せず、実行時にreferenceから再計算して保存する。

### 5.2 木材領域mask

200 Hz生強度cubeを$I_{200}$として、各画素のscoreを

$$
A(y,x) = \sum_{b=0}^{255} I_{200}(y,x,b)
$$

とする。反射率、SNV、030 HzまたはRGBはmask生成に使用しない。

試料ごとに$A(y,x)$へ3-class Multi-Otsuを適用し、強度順のclass 0、1、2へ分ける。
class 0を背景候補、class 1とclass 2の和集合を木材候補とする。class名は強度順を表すだけで、
材組織や劣化classを意味しない。

木材候補へ次の順序で形態処理を適用する。

1. disk radius 1によるbinary erosion
2. `min_object_size=1`、8近傍に相当する`connectivity=2`で小領域除去

穴埋め、closing、最大連結成分だけの選択など、未指定の形状変更は行わない。
形態処理後のmaskと、mask内の元画像座標を保持する。

### 5.3 反射率変換

試料強度$I_{200}$を、同じ列$x$のwhiteとdarkを用いて

$$
R(y,x,b)
=
\frac{I_{200}(y,x,b)-D_{200}(x,b)}
{W_{200}(x,b)-D_{200}(x,b)}
$$

へ変換する。referenceは列方向の情報を保ったまま試料の行方向へbroadcastし、空間平均した単一の
referenceスペクトルへ置き換えない。

分母が非正または非有限の場合は対応する反射率を非有限値とする。`eps`の加算や反射率の`[0,1]`への
clipは行わない。補間後の負値は第5.5節の規則で背景化する。1を超える画素は品質統計へ記録するが、
それだけでは除外しない。

### 5.4 有効画素の選択

形態学的mask内から、cutoff後の保持bandがすべて有限な画素を選ぶ。
非有限値を1つでも含む画素は解析用スペクトルから除外し、元座標と理由code 1を保存する。
形態学的mask自体は変更せず、品質条件適用後の範囲を`valid_spectrum_mask`として別に保存する。

### 5.5 256点線形補間

cutoff後の実測波長を$\lambda_0,\ldots,\lambda_{C_s-1}$とし、出力波長を

$$
\tilde{\lambda}_j
=
\lambda_0
+ \frac{j}{255}\left(\lambda_{C_s-1}-\lambda_0\right),
\qquad j=0,\ldots,255
$$

とする。各$\tilde{\lambda}_j$を挟む2つの実測band間で線形補間する。
補間区間は保持波長範囲の閉区間内に限定し、外挿、高次補間および平滑化を行わない。

現在の入力では、222点の実測反射率を913.10–2305.59 nmの範囲内で256点へ補間する。
256次元とするのは、後段で$16 \times 16$ patchとして扱えるようにするためである。

補間後の標本標準偏差が非有限または0以下の画素はSNVを定義できないため除外し、理由code 2を保存する。

残った画素の補間反射率256帯域に1つでも厳密に0未満の値があれば、画素全体を除外し、理由code 3を保存する。
除外理由が重なる場合はcode 1、2、3の順に優先する。0のband自体は除外理由にせず、負値を0へ置換しない。
SNV後に生じる負値にはこの規則を適用しない。

この画素は`valid_spectrum_mask=0`とし、疎な`reflectance`・`snv`・`pixel_row_col`には保存しない。
後段では背景ラベル0となり、trainの抽出候補、testの全有効画素、評価指標の分母から共通に外す。
空間診断図では透過表示する。元の形態学的`mask`は診断記録として保持し、
`excluded_pixel_row_col`と理由codeから背景化した位置を追跡できるようにする。

### 5.6 SNV

補間後の有効な256次元反射率を$r_p(b)$とする。画素$p$ごとに

$$
\mu_p = \frac{1}{C}\sum_{b=1}^{C} r_p(b),
\qquad C=256
$$

$$
s_p
=
\sqrt{
\frac{1}{C-1}
\sum_{b=1}^{C}\left(r_p(b)-\mu_p\right)^2
}
$$

を計算し、SNVを

$$
z_p(b) = \frac{r_p(b)-\mu_p}{s_p}
$$

と定義する。標本標準偏差を使用するため、数値誤差を除けば

$$
\frac{1}{C}\sum_{b=1}^{C}z_p(b)=0,
\qquad
\operatorname{std}_{\mathrm{ddof}=1}\!\left(z_p\right)=1,
\qquad
\lVert z_p\rVert_2=\sqrt{255}
$$

となる。実行時にこれらの不変量からの最大数値誤差を記録する。

### 5.7 反射率L2 norm map

空間確認用mapは、SNV前・256点補間後の反射率から

$$
M_{L2}(y,x)
=
\sqrt{\sum_{b=1}^{256}\widetilde{R}(y,x,b)^2}
$$

として作成する。SNV後のL2 normは全有効画素でほぼ$\sqrt{255}$となるため、mapには使用しない。
品質条件で除外した画素と背景は非有限値として透過表示する。

## 6. 保存形式

解析データと確認用可視化を分離する。

```text
data/processed/production_v1/
  config.json
  cutoff_decision.json
  preprocessing_summary.json
  manifest.parquet
  reference_band_quality.parquet
  wavelength_grid.parquet
  mask_quality.parquet
  mask_components.parquet
  sample_quality.parquet
  source_band_statistics.parquet
  source_band_summary.parquet
  output_band_statistics.parquet
  output_band_summary.parquet
  ranked_final_snv_spectra.parquet
  samples/
    <sample_id>.h5

outputs/preprocessing/production_v1/
  report_config.json
  cutoff_decision.png
  interpolated_reflectance_band_distribution.png
  interpolated_snv_band_distribution.png
  final_snv_anomaly_candidates.png
  reflectance_l2_norm/
    <sample_id>.png
```

`data/processed/`には下流解析へ渡すデータ、品質表、座標および固定configを置く。
`outputs/preprocessing/`には人が確認する図とその描画設定を置く。
探索runの保存物を本番前処理の入力として使用しない。

### 6.1 試料HDF5 schema

各`<sample_id>.h5`は、少なくとも次を持つ。

| dataset / attribute | 内容 |
| --- | --- |
| `wavelength_nm` | 補間後256点の波長vector |
| `source_wavelength_nm` | cutoff後、補間前の実測波長vector |
| `retained_source_band_index` | 保持した元band index |
| `mask` | 形態処理後の2次元木材mask |
| `valid_spectrum_mask` | 品質条件適用後の2次元有効画素mask |
| `pixel_row_col` | `reflectance`および`snv`の各行に対応する元画像座標 |
| `excluded_pixel_row_col` | 品質条件で除外したmask画素の元画像座標 |
| `excluded_reason_code` | 除外理由。1は非有限反射率、2は非正または非有限のSNV入力標準偏差、3は補間反射率の負値 |
| `reflectance` | SNV前・256点補間後の有効画素反射率 |
| `snv` | 256次元SNV |
| `reflectance_l2_norm` | 背景と無効画素を非有限値とした2次元map |
| attributes | schema version、試料ID、元ファイル、元shape、画素数、SNR閾値、SNV定義 |

HDF5は`float32`、gzip level 4、shuffleおよび画素方向chunkを使用し、1試料を1ファイルにまとめる。
目的は圧縮率だけではなく、可変長の画素スペクトル、座標、mask、波長およびmetadataを対応付け、
必要なdatasetや画素chunkだけを読めるようにすることである。
完全な3次元反射率cubeは複製せず、mask内の有効画素スペクトルだけを保存する。
現在のproduction storage schemaはversion 2とする。
負値除外版も既存datasetのshape・dtypeを保ち、理由code 3と品質集計列を追加する。
入力の区別には新しいpreprocessing IDとconfigの`negative_reflectance_policy`を使用する。
`negative_interpolated_reflectance_excluded_pixel_count`はcode 3で除外した画素数を表す。
既存の`pixels_with_any_negative_interpolated_reflectance`は保存対象についての件数を維持し、新版では0になる。

圧縮率は反射率分布に依存する。`manifest.parquet`へ試料別の実ファイル容量と、反射率・SNVを非圧縮
`float32`で保持した場合の容量を保存する。`preprocessing_summary.json`には全試料合計とその比を保存する。

## 7. 品質確認

### 7.1 機械可読な確認値

- `cutoff_decision.json`: 自動決定した保持・除外band、波長および境界
- `reference_band_quality.parquet`: 256 bandのSNR proxyと保持flag
- `mask_quality.parquet`: Multi-Otsu閾値、class画素数および形態処理前後の画素数
- `sample_quality.parquet`: 除外数、範囲外反射率、SNV形状指標および数値不変量
- `output_band_summary.parquet`: 補間後反射率とSNVのband別分布
- `preprocessing_summary.json`: 全体画素数、除外率、HDF5容量およびSNV数値誤差

### 7.2 確認用可視化

- `cutoff_decision.png`: reference SNR、反射率分布、異常値率およびSNV分布を同じcutoff境界で表示
- `interpolated_reflectance_band_distribution.png`: 補間後反射率のband別分布
- `interpolated_snv_band_distribution.png`: 最終SNVのband別分布
- `final_snv_anomaly_candidates.png`: 最大二次差分が大きい上位20画素の反射率とSNV
- `reflectance_l2_norm/<sample_id>.png`: 木材領域の空間的な反射率強度分布

`final_snv_anomaly_candidates.png`の橙線は、最終SNVで最大二次差分が大きい上位20画素である。
上段は同じ画素のSNV前反射率、下段はSNV、黒線は各bandの試料別中央値を全試料で中央値化した
代表線を示す。この図は最悪例におけるspikeの増幅を確認するための順位図であり、異常の発生率を
表すものではない。また、順位だけを根拠に画素を追加除外しない。

## 8. 可視化規約

本リポジトリのすべての図にfigure titleおよびaxes titleを付けない。図の意味、試料ID、条件名および
panelの説明はcaptionまたはファイル名で管理する。軸ラベル、目盛およびlegendは解釈に必要な場合だけ使う。

反射率L2 norm mapは次で統一する。

- colormapは`plasma`
- 背景は透過
- 全試料のmask内有限値をまとめた1%点と99%点を共通の`vmin`、`vmax`に使用
- 試料別の自動scaleを行わない
- colorbarを付けない
- 表示上のclipによって保存値を変更しない

## 9. 採用根拠の要約

- 030 Hzは長波長側の信号を補える一方、短波長側が飽和する。source接続による人工的な段差や曲率を
  導入せずに扱える200 Hz単独構成を採用した。
- 200 Hzの長波長側不安定性は、sampleやモデル結果に依存しないreference SNR終端cutoffで除去できる。
- 反射率L2 normによる二値化では一部の晩材が欠落したため、maskには200 Hz生強度和の
  3-class Multi-Otsuを採用した。

以上は確定済みの前処理仕様である。別方式を検討する場合は既存runを上書きせず、新しい研究判断と
preprocessing IDを持つ別仕様として扱う。
