# Bacteria-ID Table 5: five-fold CV and final training

Table 4と事前学習は[Bacteria-ID実験計画](bacteria_id_experiment_plan.md)に従う。
Table 5では既存のreference30事前学習checkpointを使い、M11で分類学習の設定を
選んでから、M00/M11の両方へ同じ設定を適用する。SMAE等の行は出版版Table 5の
報告値を引用する。

## モデルと探索範囲

30クラスのCLS分類器は投影前CLS 256次元にLayerNorm→Linear(30)を接続する。
最後の$k$個のTransformer blockと分類headを更新し、patch projection、
CLS/位置パラメータ、$z$への投影、decoderは固定する。$k=0$はheadのみ、
$k=8$もstemを含む全encoder解凍ではない。

AdamW（betas=(0.9, 0.95)、eps=1e-8）、CrossEntropyLoss、FP32を使う。
headと更新blockは別の固定学習率とし、両groupの全学習パラメータに同じ
weight decayを適用する。scheduler、early stopping、AMP、Optuna pruningは使わない。
各学習は50 epoch固定で、epoch 1–5はheadのみ、epoch 6–50はheadと最後の$k$ blockを
更新する。$k=0$では50 epochを通じてheadのみを更新する。

| 項目 | 候補 |
| --- | --- |
| $k$ | 0, 1, 2, 3, 4, 5, 6, 7, 8 |
| head LR | $(1,2,\ldots,10)\times10^{-4}$ |
| encoder block LR | $(1,2,\ldots,10)\times10^{-5}$。$k=0$では使用しない |
| batch size | 8, 16, 32, 64, 128 |
| TGN / FS | off/off、on/off、off/on、on/on |
| weight decay | 0、$10^{-5}$、$10^{-4}$、$10^{-3}$、$10^{-2}$ |

TGN/FSの強度と独立適用確率0.5はWoodDegradationMapの既定値とする。
finetuneのtrainスペクトルはSNV後に有効なaugmentationを適用する。
validation/testにはaugmentationを適用しない。$k=0$でencoder LRを除いた
実効探索空間は41,000設定である。

## 設定選択

finetune subsetの3,000件（30 isolateクラス×100件）に
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)を適用する。
各foldはtrain 2,400件・validation 600件で、全3,000件が一度ずつvalidationになる。
全trialはM11の事前学習seed 0から各foldで独立にfine-tuningする。
foldごとの**epoch 50のvalidation Accuracy**を算術平均した値を最大化する。
途中epochの最大値は採点・checkpoint選択に使用しない。

Optunaの単一studyで異なる設定100件を上限として探索した。最初の30件はランダム提案、
以後はTPE提案とし、初期30件には$k=0,\ldots,8$を各3回以上含めた。
同じ設定が再提案された場合は学習せずに棄却し、繰り返し重複するときは
未評価設定をランダムに補充する。実際には64件の異なる設定が完了し、
ユーザー指定で**先に完了した60件**を設定選択の対象とした。残り4件と未完了trialは
選択に使わず、100件を完走したとは記載しない。
同点の場合は先に完了したtrialを選ぶ。testは探索と設定選択に使用しない。
foldごとの分割index、設定、epoch 50のAccuracy、実行条件を保存する。

## 全件学習とtest

選ばれた**1設定**をM00/M11の両方へ適用し、それぞれ事前学習seed 0–4から
独立にfine-tuningする。finetune 3,000件をすべて学習に使い、validationは
分離しない。CVと同じ50 epoch固定・最初5 epochはheadのみの更新とし、
**epoch 50の重み**だけをtest評価に用いる。全10 runが完了してから、
別に配布されたtest 3,000件を各runで一度採点する。

Table 5にはM00/M11の5 seed平均Accuracy ± t分布による95%信頼区間半幅を
w/ pretraining列へ追加する。個別seed値と完了64設定の5-fold値を保存し、
採用設定の選択範囲が最初の60件であることを主表に明記する。
共通設定はM11のvalidationで最適化されたものであり、M00にとっての最適設定を
探索したものではない。M00/M11間の比較と文献値による位置づけではこの点を明記する。
Table 5の事前学習・split・HPO・分類・checkpointは新80/20 Table 4から独立に維持する。
