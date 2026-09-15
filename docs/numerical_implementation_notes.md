# 数値実装の記録

2026-09-16に旧論文草稿から保存した、ChemoMAE 0.2.2と呼出し側の読み取り照合に基づく記録。
設定の正は[実験プロトコル](design/experiment_protocol.md)、実行済みの事実は[検証履歴](verification_history.md)と各runの記録に置く。
今回の文書整理ではプログラムを実行しておらず、実行時tensor dtypeの実測結果を追加したものではない。

<a id="training-precision"></a>

## 1. 混合精度と学習lossの記録

**演算精度。**

学習時はCUDA AMPのFP16 autocastと動的GradScalerを用いる。モデルparameterとclean targetはFP32に保持し、追加摂動はautocastを無効にしたFP32で計算する。Encoder・decoderのforwardとloss呼出しはautocastの範囲にあるが、すべての演算が一律にFP16になるわけではない。

導入済みChemoMAE 0.2.2のlossは、復元値とtargetの差を二乗し、対象maskで選んで平均する。TargetをFP16へ落とす処理はなく、FP16の復元値とFP32のtargetの減算はFP32へ型昇格するため、差分・二乗・平均はFP32の演算経路となる。この記述は呼出し側とlibraryソースの読み取りに基づき、実行時の各tensorのdtypeを計測した結果ではない。LossをFP32で計算しても、forward中のFP16演算や逆伝播・学習軌道への数値精度の影響が除かれるわけではない。GradScalerは逆伝播用にlossをscaleするもので、履歴へ保存するlossはscale前の値である。

**履歴の集計と解釈。**

学習履歴のtrain lossは、更新中の各batchで計算したMSEを、epoch内のbatch数で割った平均である。端数batchを除き、同一条件ではbatch sizeと対象チャネル数が固定される。これは固定した最終重みを用いて全train画素を再評価したlossでも、未学習試料のlossでもない。AMP scaleや更新が省略されたstep数は別の履歴項目として保存する。

FP16の表現間隔をMSEの下限へ読み替えない。幾何に関する恒等式の数値照合では[数理的補足B.6](mathematical_notes.md#numerical-geometry)に従い、既定のFP32での表現抽出・評価と学習時の演算条件を区別する。

## 2. クラスタ中心とinertiaの記録

中心更新で非空クラスタの表現和がゼロとなる場合、通常の単位中心の更新式は定義できない。
Libraryの除算保護だけで有効な中心とみなさず、プロジェクト側で最終中心の非有限値・ゼロ・極小normを検査し、不適切ならfitの失敗として記録する。

参照実装の`reference_inertia`は最終中心更新前の割当で計算した記録値であり、保存する最終中心から再計算した`final_center_inertia`と区別する。
目的関数である平均cosine不類似度と、反復中の特定時点の記録値を混同しない。
初期化・停止条件・epsilonは[クラスタリング仕様](design/experiment_protocol.md#augmentation-clustering)を参照する。

## 3. 照合先と残る確認

- [モデル・抽出・schedule](../src/wood_degradation_map/experiments/neural.py)、[学習](../src/wood_degradation_map/experiments/training.py)、[クラスタリング](../src/wood_degradation_map/experiments/clustering.py)。
- [設定](../src/wood_degradation_map/experiments/config.py)、[実行記録の規約](experiment_runbook.md#artifact-records)。
- 採用runのGPU機種、driver・CUDA・packageの実際の版、source hash、完了状況は各runの記録から確認する。依存関係指定だけで全runの環境が一致したとは判断しない。
