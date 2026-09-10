# 関連研究・参考文献

本書は[ChemoMAEの位置づけ](chemomae_positioning.md)と[研究の目的](research_overview.md)を支える文献調査をまとめる。
調査日は2026-09-06〜07、追加確認は2026-09-10。下記の確認範囲は当時の記録であり、文書整理時に再調査したものではない。
各文献の構成・前処理・評価と、本研究で引用できる範囲を区別する。網羅的な新規性調査は未完了である。

## 1. 関連研究の対応

### 1.1 前例の対応表

| 文献 | 本構成とつながる点 | 相違点・引用の範囲 |
| --- | --- | --- |
| [Kramer (1991), 非線形PCA](https://doi.org/10.1002/aic.690370209) | 再構成とbottleneckを使い、化学工学データの次元削減・可視化を行う発想 | maskを使うTransformerではなく、decoder側も非線形 |
| [Hinton & Salakhutdinov (2006)](https://www.cs.toronto.edu/~hinton/absps/science.pdf) | autoencoder自体を次元削減・可視化・検索に使う | 深い非線形decoderを用いる。文中のfine-tuningは再構成を最適化する全体学習も指す |
| [Vincent et al. (2008), Denoising AE](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)・[同 (2010)](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) | corruptionからの復元を、有用な表現を学ぶための課題にする。noise・shiftを追加した動機に対応 | corruption・構造・評価は異なる。古材の化学状態に対する妥当性や、他のSSLより仮定が弱いことを示すものではない |
| [He et al. (2022), MAE](https://arxiv.org/html/2111.06377v3) | 可視patchだけをencoderへ渡し、masked MSEで学習 | decoderは可視tokenとmask tokenを使うTransformer。全再構成を単一16次元CLSに制限しない |
| [Xie et al. (2022), SimMIM](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.html) | masked modelingで単純な線形予測headを使う | 各位置のencoder特徴から復元する構成で、単一の低次元潜在ではない。原論文の損失はL1 |
| [Georgiev et al. (2024), Raman unmixing AE](https://arxiv.org/html/2403.04526v1) | 非線形encoderと線形decoderの組合せ。Transformer encoderの例もある | 物理制約を用いた成分分離が目的で、MAEではない |
| [Ren et al. (2025), Raman SMAE](https://arxiv.org/html/2504.16130v1) | スペクトルのmask再構成と、得た表現によるPCA等とのクラスタリング比較 | Transformer decoderを用い、教師ありfine-tuningも別途評価する |
| [Jensen et al. (2024), Ramanのdenoising VAE](https://www.nature.com/articles/s41598-024-56788-7) | noise・軸方向変動などを加えたスペクトルから元の観測を復元し、表現を学ぶ | 強度と周波数軸を別に扱うVAEが基礎で、SNV制約を保つTGNとmaskを用いる本構成とは異なる |
| [Ji et al. (2026), LeafVAE](https://doi.org/10.1016/j.compag.2026.111971) | 画素スペクトルの教師なし圧縮、潜在のクラスタリング、空間マップ化。学習済みモデルと固定クラスタ中心を新規データへ適用する | 2次元VAEと非線形decoderを使い、mask再構成ではない。葉の診断にはクラスタ構成比を入力とする教師ありrandom forestとSHAPを使う |

### 1.2 構成として特に近い例: Raman unmixing AE

Georgievらは、Ramanスペクトルの非線形encoderに、biasなしの線形decoderを組み合わせる。
encoderの候補にはDense、Convolutional、Transformerなどを含む。
したがって「encoderは複雑でも復元は線形に制約する」という設計には、分光解析で具体的な前例がある。
([Georgiev et al., Methods: Autoencoder architectures / Decoder choice](https://arxiv.org/html/2403.04526v1))

同研究はdecoder重みの非負性と、潜在の非負性・必要に応じた総和1の制約を使い、
重みをendmember、潜在を存在比として扱う。一方、本ChemoMAEにはその物理制約がなく、
SNV入力、bias、符号を許す単位潜在を使う。**この前例は構成の合理性を説明する根拠になっても、
本モデルの潜在を化学成分濃度と呼ぶ根拠にはならない。**

調査した前例のうち、**スペクトルからglobalな潜在を求め、非線形encoderと線形decoderで
再構成するという構成上の比較対象としては、Georgievらが特に近い**。
一方、本研究は化学成分への分解を課す代わりに、mask再構成から状態に関連するまとまりが
生じるかを問う。この目的の違いを含めて引用する。MAEという学習課題と、得た特徴による
クラスタリングの前例としては、次節のRenらを併せて位置づける。

### 1.3 使い方として特に近い例: Raman SMAE

RenらのSMAEは、スペクトルpatchを隠して再構成するTransformer型モデルである。
§3.2ではラベルを学習に使わず、得られた表現をKMeansへ渡し、PCAなどと比較している。
「スペクトルMAEを教師なしの特徴抽出器として利用する」という目的に近い。
([Ren et al., §2.2 / §3.2](https://arxiv.org/html/2504.16130v1))

ただし、同節のPCA等との比較はreference subsetを対象とし、別節の教師ありfine-tuning評価と
区別される。本研究の試料単位OOF評価と同一条件ではない。
また、decoderへtoken列を渡すため、現在の単一16次元潜在とは圧縮の制約が異なる。
論文中の優劣や精度を、そのまま古材NIRでの性能予測に用いることはできない。

### 1.4 「弱いdecoderなら固定特徴に有利」とまでは言えない

本構成ではdecoderが非線形な処理を担えないため、encoder側に「線形に復元できる座標」を
作る役割がある。これは構造の説明として妥当である。しかし、その座標が分類・クラスタリングに
有用かどうかは別の問いである。

原MAEのdecoder深さの比較では、encoderを固定するlinear probingに十分なdecoder深さが
重要だったと報告されている。著者らは、decoderに再構成への特化を担わせることで、
潜在をより抽象的に保てると解釈している。([He et al., §4.1 Decoder design](https://arxiv.org/html/2111.06377v3))

これは画像認識での結果であり、本ChemoMAEを否定するものでも、decoderを深くすべきという
結論でもない。**decoderの単純さは本研究の採用制約であって、固定表現の品質を保証する法則ではない。**
同様に、SimMIMの線形headの成功も、globalな16次元bottleneckの最適性までは検証していない。

### 1.5 解析の流れとして特に近い例: LeafVAE

JiらのLeafVAEは、葉のハイパースペクトル画像の各画素スペクトルを、教師なしの再構成学習で
2次元潜在へ圧縮し、KMeansによって代表的なspectral signatureへ分ける。
その割当を画像上の空間分布へ戻し、葉ごとのクラスタ構成比としても集約する。
トウモロコシの実験では、学習済みモデルと固定したクラスタ中心を用い、別の年・遺伝子型・
撮像条件を含むデータを共通の潜在空間で解析している。
**画素スペクトルの表現学習からクラスタリング、空間マップ化、新規データへの適用までの
解析の流れが本研究と近い先行研究**として位置づけられる。
([Ji et al., §2.3 / §3, Figs. 1–4](https://doi.org/10.1016/j.compag.2026.111971))

構成は全結合の非線形encoder・decoderを持つVAEで、再構成損失とKL正則化を用いる。
本研究のmask学習、アフィンdecoder、16次元単位潜在とは異なる。
また、同論文の教師なし学習はスペクトル表現とクラスタの獲得を指し、窒素量や病害などの
診断段階では、クラスタ構成比を入力とする教師ありrandom forestとSHAPを用いる。
SHAPによる画素への寄与の割当を、画素単位の化学量の直接測定とは扱わない。
本研究の教師なし領域分割とLLA・LFRによる評価とは目的・評価条件が異なるため、
同論文の診断性能を古材の劣化状態の同定や本モデルの優位性の根拠にはしない。
([Ji et al., §2.3 / §3–4](https://doi.org/10.1016/j.compag.2026.111971))

### 1.6 学習課題として近い例: Ramanのdenoising VAE

Jensenらは、RamanスペクトルへGaussian noise、波長校正の変動、clippingを加え、元のスペクトルを
復元する自己教師あり学習を行う。VAEを基礎とし、強度と周波数軸を分けて扱い、下流では潜在を
教師あり分類器へ渡す。**スペクトルのnoise・軸方向変動からの復元を表現学習に使う前例**である。
([Jensen et al., 2024, Introduction](https://www.nature.com/articles/s41598-024-56788-7))

したがって、noise・shiftから元の観測を復元する学習全般について「初」とは主張しない。
本研究との差分は、SNV制約を保つ具体的なcorruption、帯域maskとの組合せ、単一単位潜在への圧縮、
および古材の教師なしマッピングという用途に分けて説明する。Jensenらと直接比較する実験は行っていない。

## 2. 参考文献と確認範囲

論文本文または著者公開原稿を優先して確認した。以下の関連研究が本ChemoMAEと完全に同じ
構成・前処理・評価条件を検証した、という意味ではない。網羅的な新規性調査でもない。

1. Baldi, P. & Hornik, K. (1989).
   [Neural networks and principal component analysis: Learning from examples without local minima](https://doi.org/10.1016/0893-6080(89)90014-2).
   *Neural Networks*, 2(1), 53–58。
   [著者公開PDF](https://www.igb.uci.edu/~pfbaldi/publications/journals/1989/NN_and_PCA.pdf)。
2. Kramer, M. A. (1991).
   [Nonlinear principal component analysis using autoassociative neural networks](https://doi.org/10.1002/aic.690370209).
   *AIChE Journal*, 37(2), 233–243。
   [大学公開PDF](https://people.engr.tamu.edu/rgutier/web_courses/cpsc636_s10/kramer1991nonlinearPCA.pdf)。
3. Hinton, G. E. & Salakhutdinov, R. R. (2006).
   [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647).
   *Science*, 313, 504–507。
   [著者公開PDF](https://www.cs.toronto.edu/~hinton/absps/science.pdf)。
4. Vincent, P., Larochelle, H., Bengio, Y. & Manzagol, P.-A. (2008).
   [Extracting and Composing Robust Features with Denoising Autoencoders](https://doi.org/10.1145/1390156.1390294).
   *ICML*。
   [著者公開PDF](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)。
5. He, K. et al. (2022).
   [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
   *CVPR*。構成とdecoder比較は[公開本文 §3–4](https://arxiv.org/html/2111.06377v3)を参照。
6. Xie, Z. et al. (2022).
   [SimMIM: A Simple Framework for Masked Image Modeling](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.html).
   *CVPR*, 9653–9663。
7. Georgiev, D. et al. (2024).
   [Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders](https://doi.org/10.1073/pnas.2407439121).
   *PNAS*。構成の詳細は[著者公開原稿のMethods](https://arxiv.org/html/2403.04526v1)を確認。
8. Ren, P., Zhou, R.-G. & Li, Y. (2025).
   [A Self-supervised Learning Method for Raman Spectroscopy based on Masked Autoencoders](https://arxiv.org/abs/2504.16130).
   本書の構成・実験の説明は[公開原稿v1](https://arxiv.org/html/2504.16130v1)に基づく。
   [刊行版](https://doi.org/10.1016/j.eswa.2025.128576)の全文との差分は未照合。
9. Banerjee, A., Dhillon, I. S., Ghosh, J. & Sra, S. (2005).
   [Clustering on the Unit Hypersphere using von Mises-Fisher Distributions](https://jmlr.org/papers/v6/banerjee05a.html).
   *JMLR*, 6, 1345–1382。
10. Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020).
    [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html).
    *ICML*, PMLR 119, 1597–1607。SimCLRのaugmentationの役割を参照。
11. Grill, J.-B. et al. (2020).
    [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733).
    *NeurIPS*。異なるview間の予測を行うBYOLの課題定義を参照。
12. Caron, M. et al. (2021).
    [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).
    *ICCV*。[公開本文 §3.1](https://arxiv.org/html/2104.14294v2)のDINOのview間対応を参照。
13. Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y. & Manzagol, P.-A. (2010).
    [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://jmlr.org/papers/v11/vincent10a.html).
    *JMLR*, 11, 3371–3408。[公開本文 §3](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)の
    denoisingによる表現学習とcorruptionの選択を参照。原論文のmanifold解釈を、本研究のSNV制約や
    化学状態のmanifoldを同定した結果とは扱わない。
14. Hamamatsu Photonics.
    [Image sensors product selection](https://hub.hamamatsu.com/us/en/technical-notes/image-sensors/image-sensors-product-selection.html).
    メーカー技術資料。§1.2–1.3のnoise源とSNRの定義を参照。
    本研究の装置のnoise分布を測定した資料ではない。
15. Yamamoto, S., Tsuchida, S., Urai, M., Mizuochi, H., Iwao, K. & Iwasaki, A. (2022).
    [Initial Analysis of Spectral Smile Calibration of Hyperspectral Imager Suite (HISUI) Using Atmospheric Absorption Bands](https://doi.org/10.1109/TGRS.2022.3190486).
    *IEEE Transactions on Geoscience and Remote Sensing*, 60, 5534215, 1–15。
    [公開本文](https://www.researchgate.net/publication/362018335_Initial_Analysis_of_Spectral_Smile_Calibration_of_Hyperspectral_Imager_Suite_HISUI_using_Atmospheric_Absorption_Bands)の
    §IとVNIR・SWIRの解析結果を参照。衛星搭載HSIの事例であり、本研究の装置への発生頻度・大きさの外挿はしない。
16. Cui, X., Cai, W. & Shao, X. (2016).
    [Glucose induced variation of water structure from temperature dependent near infrared spectra](https://pubs.rsc.org/en/content/articlehtml/2016/ra/c6ra18912a).
    *RSC Advances*, 6, 105729–105736。§3.1の温度依存の見かけのピーク移動と、重なった帯域の
    相対強度による解釈を参照。古材の劣化や一様な波長shiftの物理モデルを検証した研究ではない。
17. Kruse, F. A. et al. (1993).
    [The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data](https://doi.org/10.1016/0034-4257(93)90013-N).
    *Remote Sensing of Environment*, 44(2–3), 145–163。SAMに関してENVI公式資料が挙げる文献。
    書誌とabstractを確認し、SAMの具体的な定義・説明は次項の公式資料で確認した。
18. NV5 Geospatial Software / Exelis Visual Information Solutions.
    [Spectral Angle Mapper](https://www.nv5geospatialsoftware.com/docs/spectralanglemapper.html)および
    [ENVI Classic Tutorial: Mapping Methods、pp. 8–9](https://www.nv5geospatialsoftware.com/portals/0/pdfs/envi/Mapping_Methods.pdf).
    SAMの角度による比較、反射率データの前提、未知のgainに対する不変性を参照。
    本研究の潜在正規化やSNV後の角度の妥当性を検証した資料ではない。
19. Ji, K. et al. (2026).
    [Variational autoencoder enables unsupervised leaf diagnosis via hyperspectral imaging](https://doi.org/10.1016/j.compag.2026.111971).
    *Computers and Electronics in Agriculture*, 251, 111971。
    ユーザー提供の刊行版PDFの§2.3、§3–4、Figs. 1–4を確認。
    画素スペクトルのVAE表現、クラスタリング、固定中心による新規データへの適用、空間マップ化を参照。
    教師なし表現学習と、ラベルを用いる下流の診断・SHAPによる説明を区別する。
20. Fearn, T., Riccioli, C., Garrido-Varo, A. & Guerrero-Ginel, J. E. (2009).
    [On the geometry of SNV and MSC](https://doi.org/10.1016/j.chemolab.2008.11.006).
    *Chemometrics and Intelligent Laboratory Systems*, 96(1), 22–26。
    出版社公開abstract・Introductionを確認。SNVの幾何とscore plotの曲線状構造の先行研究として参照し、
    TGNの優先性を示す文献とはしない。[入力球面の次元・半径](chemomae_positioning.md#snv-geometry)は本研究のSNV定義から導いた。
21. Jensen, M. N. et al. (2024).
    [Identification of extracellular vesicles from their Raman spectra via self-supervised learning](https://www.nature.com/articles/s41598-024-56788-7).
    *Scientific Reports*, 14, 6791。公開本文Introductionのcorruption・復元targetとVAEの説明を確認。
    noise・軸方向変動を用いるdenoising表現学習の前例として参照する。
22. Mantis-Ryuji.
    [分光データ拡張の著者解説、§3.2 Tangent Gaussian Noise・§3.3 Fractional Shift](https://zenn.dev/mantis_ryuji/articles/e17b4d223cd7da).
    2026-09-10確認。提案者自身の操作説明であり、第三者による優先性・有用性の検証ではない。
    SNVの分母、noise強度の抽選、入力軸の違いは[ChemoMAEの位置づけ第1.4節](chemomae_positioning.md#spectral-augmentation)に記載した。

2026-09-06〜10の調査では、固定config、利用側コード、導入済みChemoMAEのソースを読み取って実装を確認した。
2026-09-10には追加文献・記事とaugmentation実装を照合し、入力だけを摂動して元の観測をtargetにする点と
masked lossの関係を確認した。文献調査のための学習・評価・ベンチマークは実行していない。
FT-IRのデータ・結果と網羅的な新規性評価は確認範囲に含めない。
