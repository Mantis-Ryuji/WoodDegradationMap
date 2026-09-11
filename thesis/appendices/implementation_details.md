# 付録C モデル・学習・数値設定

本付録は、第3章で用いる共通設定を記録する。条件は現行の研究設計に従い、全runが完了したことや最適な設定であることを意味しない。試料分割・比較・評価の規約は第4章で記述する。

## C.1 モデル構成

| 項目 | 採用値 |
| --- | --- |
| 実装 | ChemoMAE 0.2.2 |
| 入力長 | 256チャネル |
| Patch | 16 patch、1 patchは連続16チャネル |
| 埋め込み幅 | 256 |
| Encoder | Transformer 8層、8 head、headあたり32次元 |
| Feed-forward幅・activation | 1024、GELU |
| LayerNormの配置 | Pre-norm |
| Dropout | 0 |
| 位置情報 | CLSと各patchに学習可能な位置埋め込み |
| 集約 | 最終CLSから線形射影 |
| 潜在 | 16次元、学習中・抽出時ともL2正規化 |
| Decoder | Bias付きLinear、16次元から256チャネルへの1層 |
| 全可視時のtoken数 | 16 patch＋CLSの17 |
| 主MAE条件のtoken数 | 可視8 patch＋CLSの9 |

CLSと位置埋め込みは使用版の既定の切断正規分布初期化を用い、その他の層も参照実装の初期化を用いる。追加の再初期化は行わない。Fold・反復ごとに用途別seedを設定してから構築する。

## C.2 Mask・target・追加摂動

| 条件 | 不可視patch数 | Loss対象チャネル数 | TGN確率 | Shift確率 |
| --- | ---: | ---: | ---: | ---: |
| A0 | 0 | 256（全帯域） | 0 | 0 |
| M00 | 8 | 128 | 0 | 0 |
| M10 | 8 | 128 | 0.5 | 0 |
| M01 | 8 | 128 | 0 | 0.5 |
| M11 | 8 | 128 | 0.5 | 0.5 |
| M11-25 | 4 | 64 | 0.5 | 0.5 |
| M11-75 | 12 | 192 | 0.5 | 0.5 |

M11-25・M11-75はmask率の補助実験の設定である。すべての条件で復元targetは追加摂動前のSNVとし、patch単位の追加target正規化は行わない。

TGNの回転角を $\alpha$（radian）、Fractional Shiftの移動量を $\delta$（チャネル単位）とする。$\mathcal{U}$ を指定区間の連続一様分布として、$\alpha\sim\mathcal{U}(0,\pi/36)$、$\delta\sim\mathcal{U}(-2,2)$ とする。$\pi$ は円周率であり、角度上限 $\pi/36$ radは5度に対応する。操作順はmini-batchごとにランダム化し、各操作の適用は画素単位で抽選する。摂動後にmaskを用意し、入力・可視maskを明示してモデルへ渡す。

## C.3 学習設定

| 項目 | 採用値 |
| --- | --- |
| Epoch | 800 |
| Batch size | 1024 |
| Device構成 | 単一GPU |
| 勾配蓄積 | なし（1 step） |
| Optimizer | AdamW |
| Betas・epsilon | $(0.9,0.95)$、$10^{-8}$ |
| Weight decay | 0.05。BiasとLayerNormのparameterは0 |
| Peak learning rate | $6\times10^{-4}$ |
| Warmup | 40 epochの線形増加 |
| 以後のschedule | 最小値0の半周期cosine |
| データ順 | Epochごとにshuffle |
| 端数batch | Drop last |
| AMP | FP16、GradScalerあり。学習重みはFP32 |
| EMA・gradient clipping | 用いない |
| 重み採用 | 800 epoch終了時の最終の重み |
| Early stopping | 用いない |

Epochの0始まりindexを $e$、そのepoch内の0始まりbatch indexを $t_{\mathrm{batch}}$、端数を除いた1 epochのstep数を $N_{\mathrm{step},e}$ とする。Epoch単位の連続的な学習進行度を $\tau=e+t_{\mathrm{batch}}/N_{\mathrm{step},e}$ とし、$0\leq t_{\mathrm{batch}}<N_{\mathrm{step},e}$ とする。学習率を $\eta(\tau)$、そのpeak値を $\eta_{\max}$ として、batchのforward前に

$$
\eta(\tau)=
\begin{cases}
\eta_{\max}\tau/40, & 0\leq\tau<40,\\
\dfrac{\eta_{\max}}{2}
\left[
1+\cos\left(\pi\dfrac{\tau-40}{800-40}\right)
\right],
& 40\leq\tau<800
\end{cases},
\qquad
\eta_{\max}=6\times10^{-4}
\tag{C.1}
$$

とする。離散stepで評価するため、最後の実行batchの学習率が厳密に0になることを意味しない。

## C.4 表現抽出とCosine-KMeans

| 対象 | 規約 |
| --- | --- |
| Encoder抽出 | 評価mode、全可視maskを明示、augmentationなし |
| 抽出・クラスタリング・評価 | FP32、抽出AMPなし、TF32なし |
| 潜在正規化 | 微小値 $10^{-12}$、正規化前の非有限・ゼロ・極小normを検査 |
| B0・PCAとクラスタリングの単位化 | 微小値 $10^{-6}$。数値的に単位化できない表現は記録して停止 |
| PCA | 16成分、train平均中心化、追加autoscalingなし、whiten=False、solver=auto |
| クラスタ数 | $\{2,4,6,8,10,12,14\}$、代表表示8 |
| 中心初期化 | Cosine不類似度によるk-means++型、1 fitにつき1回 |
| 最大反復 | 500 |
| 主な停止条件 | 平均cosine不類似度の相対変化 $<10^{-4}$ または絶対変化 $<10^{-7}$ |
| 空クラスタ | 現在の最近中心から遠いfit画素で中心を置き換える |
| Testへの適用 | PCA・encoder・中心を固定、再fitしない |

停止判定の相対変化の分母は前回の平均cosine不類似度の絶対値に $10^{-12}$ を加えた値とする。1 fit内の追加restartや最良の初期化の選択は行わず、実験としての3反復と区別する。

中心更新において非空クラスタの和がゼロとなる退化ケースは、本文の通常の場合の更新式では定義できない。Libraryの除算保護だけで有効な中心とみなさず、プロジェクト側で最終中心の非有限値・ゼロ・極小normを検査し、不適切ならfitの失敗として記録する。

実装のinertia記録には最終中心更新前の割当で計算された値が含まれるため、保存後の中心で再計算する平均cosine不類似度と区別して保持する。本文の目的関数を、実装のある時点の記録値と混同しない。

## C.5 再現情報と残る記載項目

Seedの基準値は20260905とし、split、画素抽出、モデル初期化、画素順、mask、学習augmentation、PCA、KMeansおよび評価摂動の用途を区別する。基準値・用途・文脈を規定のJSON文字列にし、SHA-256の先頭4 byteからseedを生成する。同じfold・反復で対応する用途を条件間でそろえ、共通画素と評価摂動の規約を保持する。

Runごとのconfig、manifest、環境、source hash、seed、重みと完了記録を保存する。学習の再開では完了epoch境界の状態を用い、途中のepochはその開始状態から再実行する。採用runのGPU機種、driver・CUDA・packageの実際の版と完了状況は、最終稿で各runの記録から確認して記載する。リポジトリの依存関係指定だけから、すべてのrunの実行環境が一致したとは判断しない。

---

執筆メモ：設定は[config.py](../../src/wood_degradation_map/experiments/config.py)、[モデル構築・抽出・schedule](../../src/wood_degradation_map/experiments/neural.py)、[学習](../../src/wood_degradation_map/experiments/training.py)、[クラスタリング](../../src/wood_degradation_map/experiments/clustering.py)と照合した。再現情報の定義先は[実験プロトコル](../../docs/design/experiment_protocol.md)と[runbook](../../docs/experiment_runbook.md)。本ドラフト作成時に学習・検証コードは実行していない。
