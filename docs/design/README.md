# 研究設計文書

## ステータス

このディレクトリは、本リポジトリにおける現行の研究・実装設計を管理する。
各文書で `Open` と明示した事項を除き、記載内容を **Fixed** とする。

`docs/修士研究実験計画.md` と `docs/修士研究評価指標.md` は初期検討時の draft であり、
内容が衝突する場合は本ディレクトリの文書を優先する。

## 文書一覧

| 文書 | 状態 | 内容 |
| --- | --- | --- |
| [experiment_protocol.md](experiment_protocol.md) | Fixed | 比較条件、5-fold CV、共通クラスタ数、実行順序 |
| [evaluation_metrics.md](evaluation_metrics.md) | Fixed | cosine-silhouette、LLA-3/5/9、label flip rate |
| [visualization_and_interpretation.md](visualization_and_interpretation.md) | Fixed | 全体学習後の4条件比較とHungarian matching |
| [preprocessing.md](preprocessing.md) | Open | 前処理および030 Hz/200 Hzの扱い |

## 設計の要約

- 分割単位は `KYOw...` で識別される試料とする。
- 試料単位のランダム5-fold CVを行い、同一試料をfold間で分割しない。
- 主比較はraw SNV、PCA、AE、MAE、およびaugmentationの2×2 ablationとする。
- mask率は主比較では50%に固定し、25/50/75%のsweepは補助実験とする。
- CVの主評価はcosine-silhouette、LLA-3/5/9、label flip rateとする。
- raw SNV全試料から一度だけ決めた共通のクラスタ数を全条件に適用する。
- CV結果から「best条件」を事後選択しない。
- 全体学習後はraw SNV、PCA、標準MAE、提案Aug-MAEの4条件を解釈・可視化する。
- A0はChemoMAE v0.2.1の全領域再構成lossを使用する。
