# 関連研究・参考文献

本書は[ChemoMAEの採用理由](chemomae_positioning.md)に対応する先行研究と参考文献をまとめる。
各文献の参照版と対象範囲を示し、古材NIR-HSIにおける本研究の評価と区別する。

## 1. 関連研究の対応

構成、学習課題、表現の用途に分けて先行研究と対応づける。
いずれも本研究と同じ前処理・構成・評価を検証したものではなく、既報の性能を古材NIRへ外挿しない。

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

Georgievらのunmixingでは、非負性や総和制約によって潜在とdecoder重みを存在比・endmemberに対応づける。
本研究の符号を許す単位潜在とアフィンdecoderには、この物理的制約はない。
SMAEのRamanクラスタリング比較についても、データ分割・学習条件が本研究と同一とは限らない。
補助実験での比較条件は[Bacteria-IDプロトコル](design/bacteria_id_experiment_plan.md)を参照する。

単純なdecoderの採用は表現品質の保証ではない。原MAEのlinear probingにおけるdecoder深さの結果や、
SimMIMの位置別の線形headから、単一16次元潜在の最適性は導けない。
本構成の有用性は[試料単位の評価](design/evaluation_metrics.md)と観測スペクトルの解釈から検討する。

<a id="references"></a>

## 2. 参考文献

### 2.1 再構成による次元削減とdenoising

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
5. Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y. & Manzagol, P.-A. (2010).
    [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://jmlr.org/papers/v11/vincent10a.html).
    *JMLR*, 11, 3371–3408。[公開本文 §3](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)の
    denoisingによる表現学習とcorruptionの選択を参照。原論文のmanifold解釈を、本研究のSNV制約や
    化学状態のmanifoldを同定した結果とは扱わない。

### 2.2 Masked modelingとview間対応

6. He, K. et al. (2022).
    [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
    *CVPR*。構成とdecoder比較は[公開本文 §3–4](https://arxiv.org/html/2111.06377v3)を参照。
7. Xie, Z. et al. (2022).
    [SimMIM: A Simple Framework for Masked Image Modeling](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.html).
    *CVPR*, 9653–9663。
8. Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020).
    [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html).
    *ICML*, PMLR 119, 1597–1607。SimCLRのaugmentationの役割を参照。
9. Grill, J.-B. et al. (2020).
    [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733).
    *NeurIPS*。異なるview間の予測を行うBYOLの課題定義を参照。
10. Caron, M. et al. (2021).
    [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).
    *ICCV*。[公開本文 §3.1](https://arxiv.org/html/2104.14294v2)のDINOのview間対応を参照。

### 2.3 分光表現学習と空間マッピング

11. Georgiev, D. et al. (2024).
    [Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders](https://doi.org/10.1073/pnas.2407439121).
    *PNAS*。構成の説明は[著者公開原稿のMethods](https://arxiv.org/html/2403.04526v1)に基づく。
12. Ren, P., Zhou, R.-G. & Li, Y. (2025).
    [A Self-supervised Learning Method for Raman Spectroscopy based on Masked Autoencoders](https://arxiv.org/abs/2504.16130).
    本書の構成・実験の説明は[公開原稿v1](https://arxiv.org/html/2504.16130v1)に基づく。
    補助実験で用いる[刊行版](https://doi.org/10.1016/j.eswa.2025.128576)の報告値とは参照版を区別する。
13. Jensen, M. N. et al. (2024).
    [Identification of extracellular vesicles from their Raman spectra via self-supervised learning](https://www.nature.com/articles/s41598-024-56788-7).
    *Scientific Reports*, 14, 6791。Introductionのcorruption・復元targetとVAEの説明を参照。
    noise・軸方向変動を用いるdenoising表現学習の前例として参照する。
14. Ji, K. et al. (2026).
    [Variational autoencoder enables unsupervised leaf diagnosis via hyperspectral imaging](https://doi.org/10.1016/j.compag.2026.111971).
    *Computers and Electronics in Agriculture*, 251, 111971。
    §2.3、§3–4、Figs. 1–4の画素スペクトルのVAE表現、クラスタリング、
    固定中心による新規データへの適用、空間マップ化を参照。
    教師なし表現学習と、ラベルを用いる下流の診断・SHAPによる説明を区別する。

### 2.4 SNV幾何・角度比較・augmentation

15. Fearn, T., Riccioli, C., Garrido-Varo, A. & Guerrero-Ginel, J. E. (2009).
    [On the geometry of SNV and MSC](https://doi.org/10.1016/j.chemolab.2008.11.006).
    *Chemometrics and Intelligent Laboratory Systems*, 96(1), 22–26。
    出版社公開abstract・Introductionに示されたSNVの幾何とscore plotの曲線状構造を参照。
    [入力球面の次元・半径](chemomae_positioning.md#snv-geometry)は本研究のSNV定義から導いた。
16. Banerjee, A., Dhillon, I. S., Ghosh, J. & Sra, S. (2005).
    [Clustering on the Unit Hypersphere using von Mises-Fisher Distributions](https://jmlr.org/papers/v6/banerjee05a.html).
    *JMLR*, 6, 1345–1382。
17. Kruse, F. A. et al. (1993).
    [The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data](https://doi.org/10.1016/0034-4257(93)90013-N).
    *Remote Sensing of Environment*, 44(2–3), 145–163。SAMに関してENVI公式資料が挙げる文献。
    SAMの定義・説明は次項の公式資料を参照。
18. NV5 Geospatial Software / Exelis Visual Information Solutions.
    [Spectral Angle Mapper](https://www.nv5geospatialsoftware.com/docs/spectralanglemapper.html)および
    [ENVI Classic Tutorial: Mapping Methods、pp. 8–9](https://www.nv5geospatialsoftware.com/portals/0/pdfs/envi/Mapping_Methods.pdf).
    SAMの角度による比較、反射率データの前提、未知のgainに対する不変性を参照。
    本研究の潜在正規化やSNV後の角度の妥当性を検証した資料ではない。

### 2.5 測定noiseと波長方向の変動

19. Hamamatsu Photonics.
    [Image sensors product selection](https://hub.hamamatsu.com/us/en/technical-notes/image-sensors/image-sensors-product-selection.html).
    メーカー技術資料。§1.2–1.3のnoise源とSNRの定義を参照。
    本研究の装置のnoise分布を測定した資料ではない。
20. Yamamoto, S., Tsuchida, S., Urai, M., Mizuochi, H., Iwao, K. & Iwasaki, A. (2022).
    [Initial Analysis of Spectral Smile Calibration of Hyperspectral Imager Suite (HISUI) Using Atmospheric Absorption Bands](https://doi.org/10.1109/TGRS.2022.3190486).
    *IEEE Transactions on Geoscience and Remote Sensing*, 60, 5534215, 1–15。
    [公開本文](https://www.researchgate.net/publication/362018335_Initial_Analysis_of_Spectral_Smile_Calibration_of_Hyperspectral_Imager_Suite_HISUI_using_Atmospheric_Absorption_Bands)の
    §IとVNIR・SWIRの解析結果を参照。衛星搭載HSIの事例であり、本研究の装置への発生頻度・大きさの外挿はしない。
21. Cui, X., Cai, W. & Shao, X. (2016).
    [Glucose induced variation of water structure from temperature dependent near infrared spectra](https://pubs.rsc.org/en/content/articlehtml/2016/ra/c6ra18912a).
    *RSC Advances*, 6, 105729–105736。§3.1の温度依存の見かけのピーク移動と、重なった帯域の
    相対強度による解釈を参照。古材の劣化や一様な波長shiftの物理モデルを検証した研究ではない。
