# 研究の目的と説明文

本書は研究全体の問い、手法と証拠の役割、論文・発表向けの説明文をまとめる。
固定した実験条件は[研究設計](design/README.md)、数理的な位置づけと採用理由は
[ChemoMAEの位置づけ](chemomae_positioning.md)、進捗は[ToDo](../ToDo.md)を参照する。
説明文は研究計画に対応し、未完了の比較やFT-IRの結果を述べるものではない。

<a id="research-focus"></a>

## 1. 研究の問い

主軸は、**状態を事前に定義しにくい材料の化学的な違いを捉える、自己教師ありスペクトル表現学習**とする。
研究の中心となる問いは次のように表せる。

> 化学状態の正解ラベルを与えずに、スペクトルの帯域間関係を学ぶことで、
> 状態の違いを探索・解釈できる表現を得られるか。

ここで「定義しにくい」とは、状態の区分や正解ラベルを事前に定める根拠が十分にないことを指す。
中心に置くのは、どの違いを状態として区別するか自体が探索課題である場合である。
化学状態が連続的・複合的である可能性も考え、クラスタを初めから存在する離散的な化学classとは仮定しない。
「化学状態を捉える」とは、組成や構造などに関わる違いが表現上でどう現れるかを調べ、その対応を
別の観測から解釈することであり、潜在の各軸を成分や濃度として同定することを前提としない。

本研究では、この課題の実証対象として古材を扱う。劣化、樹種、組織、含水、表面状態などに関わる変動が
重なりうるNIR-HSIから低次元の座標系を学び、スペクトル群の空間分布を調べる。
古材で得られる知見は、この条件での手法の有用性と限界を示すものである。他材料への適用可能性を論じる際は、
SNVで保持・除去する情報、corruptionの妥当性、化学的対応の根拠を対象ごとに検討する必要がある。

| 範囲 | 本研究での位置づけ |
| --- | --- |
| 研究課題 | 状態の区分や正解を事前に定めにくい材料から、化学状態の違いを捉える表現を学ぶ |
| 提案する方法 | SNV制約を保つ摂動をmasked denoisingへ組み込む自己教師あり学習 |
| 実証対象 | 古材NIR-HSIの教師なし空間マッピングと、位置対応FT-IRによる解釈の計画 |
| 一般化の範囲 | 現行実験で比較する範囲は古材の未知試料。他材料・別測定条件への有効性は未検証 |

手法上の中心は、**TGN・Fractional Shiftという摂動の設計と、それをマスク再構成へ組み込む学習課題**である。
追加摂動前の観測を復元する過程で、化学状態をより安定して反映する表現の獲得を期待する。
この期待を、得られたマップの比較と化学的解釈から検討する構成にする。

| 研究内の役割 | 方法・観測 | 答える問い |
| --- | --- | --- |
| 表現を学ぶための提案 | SNV制約を保つTGN・shiftとmasked denoising | 帯域補完と追加摂動からの復元を学ぶことで、利用時の表現とマップがどう変わるか |
| 表現を領域へ対応づける | CLS由来の単一単位潜在、cosineクラスタリング、空間マップ | 学習したスペクトル群が試料表面のどこに分布するか。vMFは分割手法への依存性も調べる |
| マップの性質を比較する | 試料単位CV、baseline・ablation、LLA・LFR・ARI・occupancy等 | 未知試料でどの程度空間的にまとまり、指定摂動や学習反復でどう変わるか。退化やtrade-offはあるか |
| 化学的な意味を検討する | NIRの代表・差スペクトル、試料情報、位置対応FT-IRの計画 | 分割された領域の差を化学状態とどう対応づけられるか。劣化以外の説明は何か |

CVで比較できるのはマップの性質であり、化学的な意味は別の観測から検討する。
例えば、LLAが高くてもクラスタが一つへ集中していれば状態差を捉えやすくなったとは言えず、
LFRが低くても化学的に重要な差を保持したとは限らない。逆に、一部領域のFT-IRで差が見つかっても、
全画素の劣化分類精度や未知試料での性能が示されたことにはならない。

結果は「どの条件で、どの性質が変わったか」と「その領域差をどこまで化学的に解釈できたか」を対応づけて報告する。
改善が限定的な場合や化学的対応が不明な場合も、その範囲を研究の結論に含める。

## 2. 短い説明

> 状態の区分や正解ラベルを事前に定めにくい材料から、化学的な違いを捉えるスペクトル表現の学習を目指す。
> SNVの幾何的制約を保つTangent Gaussian NoiseとFractional Shiftをマスク再構成へ組み込み、
> 帯域補完とdenoisingを通じて、状態差を捉えるのに有用な座標系の獲得を目指す。
> 古材NIR-HSIを実証対象に、得られた表現を教師なしでマッピングし、試料単位CVで空間的一貫性・摂動安定性・反復間再現性を比較する。
> 化学的な意味はNIRスペクトルと位置対応FT-IRから検討する計画であり、他材料への有効性は今後の検証課題とする。

## 3. 論文・発表向けの説明

> 本研究は、状態の区分や正解ラベルを事前に定めにくい材料に対し、化学状態の違いを探索・解釈するための
> 自己教師ありスペクトル表現学習を検討する。可視帯域から隠した帯域を予測するマスク再構成を基礎とし、
> SNV後の平均ゼロ・一定norm制約を保つTangent Gaussian NoiseとFractional Shiftを入力へ加える。
> 復元targetには追加摂動前の観測を用い、帯域補完とdenoisingを通じて、化学状態を反映する帯域間関係の
> 学習を促すことを期待する。Transformer encoderは画素ごとのスペクトルをCLS由来の単一16次元単位ベクトルへ
> 集約し、アフィンdecoderによる再構成を学ぶ。実証対象には、画素単位の正解劣化ラベルがない古材NIR-HSIを用いる。
> 学習後は全帯域を可視としてencoderを固定し、表現のクラスタリングから試料表面の領域マップを得る。
> 試料単位のheld-out評価では、raw SNV、PCA、AE、MAEとの比較とaugmentationのablationにより、
> マップの空間的一貫性、指定摂動への安定性、反復間再現性を、占有率やクラスタ分離の診断と併せて調べる。
> さらに、NIRの代表・差スペクトルと、クラスタの位置に対応するFT-IR測定によって、
> 領域間の差を化学的に解釈する計画である。教師なし指標による比較と局所的な化学的観測を区別し、
> 学習した表現がどのような状態差の探索に有用か、その根拠と限界を明らかにする。
> 他材料への有効性は、この古材での実証とは分けて検証すべき課題として扱う。

上記は研究目的と方法を説明する文案であり、FT-IRの実施・結果を確認するまでは計画形を維持する。
MAEの選択とdenoisingの仮定、潜在の単位norm制約、主張できる範囲は
[ChemoMAEの位置づけ](chemomae_positioning.md)に詳述する。

この位置づけの背景としては、非線形PCA、Denoising AE、Raman unmixing AE、Raman SMAEをそれぞれ
「利用目的」「denoisingによる表現学習」「非線形encoderと線形decoder」「mask学習とクラスタリング」
の文脈で引用できる。
([Kramer, 1991](https://doi.org/10.1002/aic.690370209);
[Vincent et al., 2010](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf);
[Georgiev et al., 2024](https://doi.org/10.1073/pnas.2407439121);
[Ren et al., 2025](https://arxiv.org/html/2504.16130v1))

さらに、LeafVAEは「画素スペクトルの教師なし圧縮からクラスタリングと空間マップ化へ進み、
固定した表現・クラスタ中心を新規データへ適用する解析」の前例として引用できる。
([Ji et al., 2026](https://doi.org/10.1016/j.compag.2026.111971))

## 4. 英語での短い説明

> We investigate self-supervised spectral representation learning to capture chemical-state variation
> in materials for which state categories or ground-truth labels are difficult to define in advance.
> We combine masked reconstruction with Tangent Gaussian Noise and Fractional Shift,
> which preserve the zero-mean and constant-norm constraints of SNV spectra. The model predicts the
> original observed spectrum at masked channels from corrupted visible bands through a single
> unit-norm latent vector and an affine decoder. We hypothesize that this task encourages
> representations useful for exploring chemical-state differences. Aged-wood near-infrared hyperspectral
> imaging serves as the empirical case study. The frozen encoder is applied to fully visible spectra
> for clustering and spatial mapping. Sample-level held-out comparisons
> assess spatial coherence, stability under specified perturbations, and repeatability, alongside
> occupancy and geometric diagnostics. Spatially matched FT-IR measurements are planned to support
> chemical interpretation of selected regions. These complementary observations will be used to
> examine the utility and limits of the learned representation; the unsupervised metrics alone
> do not establish chemical validity or degradation-detection accuracy. Transfer to other materials
> remains to be evaluated.

## 5. 論文での説明順序と数式の配置

論文は、材料側の課題、表現学習の提案、CVでの比較、化学的解釈という順に組み立てる。
木材・文化財の読者には領域差と化学的解釈を、分光・ケモメトリクスの読者には摂動の設計と表現学習を
詳しく説明できるが、研究全体の問いと証拠の役割は共通とする。投稿先と最終的な強調点は未決定である。
既定の主要比較、反例、trade-offを保持し、結果を見て強調点を決めた場合も、後から作った仮説を事前仮説として記載しない。

本文の数式は、各式がどの設計判断を支えるかを一緒に説明する。SNVの二つの制約、
提案するTGN・shift、masked denoisingのtargetとlossを、手法の違いを理解するための中心に置く。
クラスタリングとLLA・LFRには意味と必要な定義を示し、標準的なTransformer内部式、導出の逐次展開、
vMFの数値処理、seedの生成詳細などは補足資料へ配置できる。vMFを主に論じる場合は密度・割当の定義も本文に置く。
この区分は論文編集上の案であり、リポジトリの再現用仕様を省略する方針ではない。
