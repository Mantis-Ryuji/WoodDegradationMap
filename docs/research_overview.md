# 研究の目的と説明文

本書は研究の問い・方法・証拠の対応と、日本語・英語の説明文をまとめる。
固定条件は[研究設計](design/README.md)、採用理由は[ChemoMAEの位置づけ](chemomae_positioning.md)、
進捗は[ToDo](../ToDo.md)を参照する。説明文は計画に対応し、未完了の比較・FT-IRの結果は含めない。

<a id="research-focus"></a>

## 1. 研究の問い

### 1.1 対象とする課題と実証の範囲

主軸は、**状態を事前に定義しにくい材料の化学的な違いを捉える、自己教師ありスペクトル表現学習**とする。
研究の中心となる問いは次のように表せる。

> 化学状態の正解ラベルを与えずに、スペクトルの帯域間関係を学ぶことで、
> 状態の違いを探索・解釈できる表現を得られるか。

状態区分の根拠が十分でなく、何を区別するか自体が探索課題となる場合を対象とする。
化学状態は連続的・複合的かもしれず、クラスタを既存の離散classとは仮定しない。
「化学状態を捉える」とは組成・構造に関わる違いが表現にどう現れるかを別の観測から解釈することで、
潜在軸を成分・濃度として同定することではない。

古材NIR-HSIでは劣化・樹種・組織・含水・表面状態などの変動が重なり得る。
ここから低次元座標を学び、スペクトル群の表面分布を調べる。現行比較の一般化範囲は古材の未知試料である。
他材料・別測定条件への適用には、SNVで残す情報、corruptionの妥当性、化学的対応の根拠を改めて検討する。

### 1.2 提案する学習課題と解析の流れ

手法上の中心は、**SNV制約を保つTGN・Fractional Shiftをmasked reconstructionへ組み込むこと**である。
Random maskもdenoisingのcorruptionと捉え、欠落に変形を追加する効果をMAE群で検討する。
追加摂動前の観測を復元する課題が、状態差の探索に有用な表現を促すことを期待する。

| 研究内の役割 | 方法・観測 | 答える問い |
| --- | --- | --- |
| 表現を学ぶための提案 | SNV制約を保つTGN・shiftとmasked denoising | 帯域補完と追加摂動からの復元を学ぶことで、利用時の表現とマップがどう変わるか |
| 表現を領域へ対応づける | CLS由来の単一単位潜在、cosineクラスタリング、空間マップ | 学習したスペクトル群が試料表面のどこに分布するか。vMFは分割手法への依存性も調べる |
| マップの性質を比較する | 試料単位CV、baseline・ablation、LLA・LFR・ARI・occupancy等 | 未知試料でどの程度空間的にまとまり、指定摂動や学習反復でどう変わるか。退化やtrade-offはあるか |
| 化学的な意味を検討する | NIRの代表・差スペクトル、試料情報、位置対応FT-IRの計画 | 分割された領域の差を化学状態とどう対応づけられるか。劣化以外の説明は何か |

TGN・shiftの効果は、主要比較とMAE4条件の2×2比較・交互作用を既定指標で評価する。
付録B.5に対応する表現幾何の補助診断一式は現行の実施計画から外し、数理的補足と未採用候補に位置づける
（[必要性の見直し](design/representation_geometry_diagnostics.md)）。

### 1.3 証拠から答えられること

CVで比較できるのはマップの性質であり、化学的な意味は別の観測から検討する。
例えば、LLAが高くてもクラスタが一つへ集中していれば状態差を捉えやすくなったとは言えず、
LFRが低くても化学的に重要な差を保持したとは限らない。逆に、一部領域のFT-IRで差が見つかっても、
全画素の劣化分類精度や未知試料での性能が示されたことにはならない。

結果は「どの条件で、どの性質が変わったか」と「その領域差をどこまで化学的に解釈できたか」を対応づけて報告する。
改善が限定的な場合や化学的対応が不明な場合も、その範囲を研究の結論に含める。
現行の指標比較だけから、特定の入力差の選択的な強調・抑制や、それが指標改善を引き起こした機構を主張しない。

## 2. 論文・発表向けの説明文

以下の3文案は第1節と同じ問い・提案・証拠の範囲を、用途に応じて書き分けたものである。
方法と期待を説明する文案として用い、比較結果やFT-IRの実施を確認するまでは計画形を維持する。

### 2.1 日本語の短い説明

> 状態の区分や正解ラベルを事前に定めにくい材料から、化学的な違いを捉えるスペクトル表現の学習を目指す。
> SNVの幾何的制約を保つTangent Gaussian NoiseとFractional Shiftをマスク再構成へ組み込み、
> 欠落・変形を与えた観測からの復元を通じて、状態差の探索に有用な座標系の獲得を目指す。
> 古材NIR-HSIを実証対象に、得られた表現を教師なしでマッピングし、試料単位CVで空間的一貫性・摂動安定性・反復間再現性を比較する。
> 化学的な意味はNIRスペクトルと位置対応FT-IRから検討する計画であり、他材料への有効性は今後の検証課題とする。

### 2.2 日本語の詳しい説明

> 本研究は、状態の区分や正解ラベルを事前に定めにくい材料に対し、化学状態の違いを探索・解釈するための
> 自己教師ありスペクトル表現学習を検討する。可視帯域から隠した帯域を予測するマスク再構成を基礎とし、
> SNV後の平均ゼロ・一定norm制約を保つTangent Gaussian NoiseとFractional Shiftを入力へ加える。
> 復元targetには追加摂動前の観測を用い、欠落・変形からの復元を通じて、化学状態を反映する帯域間関係の
> 学習を促すことを期待する。Transformer encoderは画素ごとのスペクトルをCLS由来の単一16次元単位ベクトルへ
> 集約し、アフィンdecoderによる再構成を学ぶ。実証対象には、画素単位の正解劣化ラベルがない古材NIR-HSIを用いる。
> 学習後は全帯域を可視としてencoderを固定し、表現のクラスタリングから試料表面の領域マップを得る。
> 試料単位のheld-out評価では、raw SNV、PCA、AE、MAEとの比較とaugmentationのablationにより、
> マップの空間的一貫性、指定摂動への安定性、反復間再現性を、占有率やクラスタ分離の診断と併せて調べる。
> さらに、NIRの代表・差スペクトルと、クラスタの位置に対応するFT-IR測定によって、
> 領域間の差を化学的に解釈する計画である。教師なし指標による比較と局所的な化学的観測を区別し、
> 学習した表現がどのような状態差の探索に有用か、その根拠と限界を明らかにする。
> 他材料への有効性は、この古材での実証とは分けて検証すべき課題として扱う。

### 2.3 英語での短い説明

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

## 3. 説明を支える根拠と論文への展開

### 3.1 採用理由と関連研究の役割

MAE・追加corruption・単位潜在の採用理由は[ChemoMAEの位置づけ](chemomae_positioning.md)で説明する。
利用目的（非線形PCA）、学習課題（denoising・MAE）、構成（Raman unmixing AE）、解析手順（LeafVAE）に分けた
文献との対応と確認範囲は[関連研究](related_work.md)に集約する。

### 3.2 説明順序と本文・付録の分担

材料側の課題、表現学習、CV比較、化学的解釈の順を基本とする。投稿先と最終的な強調点はOpenであり、
木材・文化財向けの領域差、分光・ケモメトリクス向けの摂動設計のどちらを厚くしても、主要比較・反例・trade-offは保持する。
事後的な補助診断を事前仮説へ遡及させない。

本文はSNV・TGN・shift・targetとloss、クラスタリング・LLA・LFRの意味を説明し、導出・数値処理・再現詳細を付録で支える。
具体的な掲載位置と執筆状況は[構成案](../thesis/outline.md)・[執筆計画](../thesis/writing_plan.md)で管理する。
