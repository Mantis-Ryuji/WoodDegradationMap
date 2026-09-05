# 全体学習後の可視化と解釈

## 1. ステータス

**Fixed**

CV結果からbest条件を選択せず、事前に定めた次の4条件を全試料でfitまたは学習する。
本書のraw SNVは、本番前処理済みの256次元SNVを表現変換せず直接使用するbaselineを指す。

| ID | 条件 | 役割 |
| --- | --- | --- |
| B0 | raw SNV | 入力空間baseline |
| B1 | PCA | 線形baseline |
| M00 | MAE 50% | 標準MAE |
| M11 | Aug-MAE 50% | 提案条件 |

すべての条件を共通の基準クラスタ数$K_0$でクラスタリングする。
この段階の結果は全データを用いた記述的解析であり、汎化性能の根拠は5-fold CVに置く。

## 2. ラベル規約

- 背景は0とする。
- クラスタは1から$K_0$までとする。
- クラスタ番号に順序または劣化度の意味を持たせない。
- ラベル整列前後でクラスタ所属そのものは変更しない。

## 3. Hungarian matching

### 3.1 基準条件

B0のraw SNV partitionを基準とする。
B1、M00、M11は、それぞれ独立にB0へ直接matchingする。

試料ごとのmatching、およびB0→B1→M00→M11のような連鎖matchingは行わない。

### 3.2 contingency matrix

全試料で共通して有効な木材画素だけを使用する。
B0のラベルを$i$、比較条件のラベルを$j$として、

$$
C_{ij}
=\sum_p\mathbf{1}[y_p^{(B0)}=i\land y_p^{(c)}=j]
$$

を作る。背景0はmatrixおよび割当から除外する。

Hungarian algorithmにより

$$
\max_{\pi}\sum_{i=1}^{K_0}C_{i,\pi(i)}
$$

となる1対1対応$\pi$を求め、比較条件のラベルをB0のラベル番号へ置換する。

### 3.3 適用範囲

得られた対応は全試料で共通とし、次へ一貫して適用する。

- label mapの色と番号
- cluster sizeおよびoccupancy
- 代表スペクトルと差スペクトル
- cluster別の補助統計

matchingは表示と対応関係の確認を目的とし、異なる条件のクラスタが同一の意味を持つことを保証しない。
contingency matrixまたはmatching後のoverlapを併記し、対応の弱いクラスタを可視化上の同一色だけで解釈しない。

条件間の比較には、本番前処理HDF5の`pixel_row_col`で同一と確認できる画素だけを使用する。
条件ごとに画素集合を変更したり、label mapを位置補正したりしない。

## 4. 可視化・解釈項目

本節の規約は、前処理診断を含むリポジトリ内のすべての可視化へ適用する。

- figure titleおよびaxes titleを付けない。
- 図の意味、試料ID、条件名およびpanelの説明はcaptionまたはファイル名で管理する。
- 軸ラベル、目盛、legendおよびcolorbar labelは、値の解釈に必要な場合は表示する。

| 項目 | 目的 |
| --- | --- |
| label map | クラスタの空間分布を比較する |
| contingency/overlap matrix | 条件間のクラスタ対応の強さを確認する |
| representative spectra | 各クラスタのNIRスペクトル形状を確認する |
| difference spectra | クラスタ差が大きい波長帯を確認する |
| cluster size distribution | collapse、過小クラスタ、不均衡を確認する |
| latent PCA/UMAP | 潜在空間の構造を補助的に観察する |

raw SNV、PCA、M00、M11で同じ試料順、同じlabel palette、同じ表示範囲を使用する。

## 5. 解釈上の制約

- 正解劣化ラベルがないため、クラスタを直ちに劣化classと断定しない。
- label map、代表スペクトル、差スペクトルおよび試料情報を合わせて解釈する。
- UMAPなどの2次元投影だけで表現の優劣を決めない。
- 可視化結果を用いて主条件を入れ替えたり、best条件を事後定義したりしない。
