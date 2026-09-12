# 修士論文の執筆計画

作成日: 2026-09-11 / 更新日: 2026-09-12

本書は[構成案](outline.md)を原稿にするための作業計画である。章ごとの論点は構成案、現在の本文・付録への入口は[原稿案内](README.md)に置き、ここでは執筆順、本文と付録の分担、参照資料、図表、未整備事項を扱う。

第3章と付録A〜Cの第1稿、画像生成による解析フロー参考図がある。第4章はLLA・補正LLAの部分のみ原稿があり、残りと付録Dは未執筆である。試料情報の整理と並行して第3・4章および対応する付録を整える方針とし、方法を記述できることと実験が完了していることを区別する。

## 1. 原稿と資料の置き場所

以下は現在の配置と将来の配置案を合わせた見取り図である。`[予定]`は未作成のファイル・ディレクトリを示し、執筆や資料の採用に合わせて用意する。

```text
thesis/
├── README.md                           # 現在の原稿への入口
├── outline.md
├── writing_plan.md
├── notation.md                         # 本文・付録の共通記号表
├── chapters/
│   ├── introduction.md                 # [予定]
│   ├── materials_measurements.md       # [予定]
│   ├── 3_analysis_methods/
│   │   ├── overview.md                  # 3.1
│   │   ├── preprocessing.md             # 3.2
│   │   ├── spectral_augmentation.md      # 3.3
│   │   ├── representation_learning.md   # 3.4
│   │   └── clustering_mapping.md        # 3.5
│   ├── evaluation_protocol/
│   │   ├── comparison_design.md         # [予定] 4.1〜4.3、4.6
│   │   ├── metrics_aggregation.md       # 4.4のLLA・補正LLAのみ。残りの4.4〜4.5は予定
│   │   └── interpretation_protocol.md   # [予定] 4.7
│   ├── results.md                      # [予定]
│   ├── discussion.md                   # [予定]
│   └── conclusions.md                  # [予定]
├── appendices/
│   ├── preprocessing_diagnostics.md     # A
│   ├── mathematical_details.md          # B
│   ├── implementation_details.md        # C
│   ├── evaluation_details.md            # [予定] D
│   └── supplementary_results.md         # [予定] E
├── figures/                            # 採用図・図の編集元
├── tables/                             # [予定] 採用表・表の編集元
└── references/                         # [予定] 書誌情報。形式は執筆環境に合わせる
```

原稿のファイル単位と論文の章単位は同じでなくてよい。長い第3・4章だけ節ごとに分ければ、見直す範囲を小さく保ちながら、完成原稿では一続きの章として読める。第3章のフォルダ名はユーザーが指定した `3_analysis_methods/` を用い、各節のファイル名は内容を表す名称とする。章の統合や掲載順を変える際は目次と参照リンクも同期する。

研究条件の正は既存の `docs/design/`、実行状況はToDo・runbook・run成果物とする。原稿には必要な条件を文章・表で記載し、執筆メモに元資料を残す。原稿側だけで条件を変更したり、既存の成果物を書き換えたりしない。

## 2. 参照資料と実装の対応

| 執筆単位 | 主に参照する文書 | 照合する実装・記録 |
| --- | --- | --- |
| 研究の問い、3.1 | [研究概要](../docs/research_overview.md)、[位置づけ](../docs/chemomae_positioning.md) | [固定設定](../src/wood_degradation_map/experiments/config.py)、[表現抽出](../src/wood_degradation_map/experiments/neural.py) |
| 3.2と付録A | [前処理仕様](../docs/design/preprocessing.md) | [production前処理](../src/wood_degradation_map/preprocessing/production_preprocessing.py)、[波長grid](../src/wood_degradation_map/preprocessing/spectral_grid.py)、[Multi-Otsu](../src/wood_degradation_map/preprocessing/intensity_multiotsu_masking.py)、[スペクトル品質](../src/wood_degradation_map/preprocessing/spectral_quality.py) |
| 3.3と付録B | [摂動定義](../docs/chemomae_positioning.md#spectral-augmentation)、[実験プロトコル](../docs/design/experiment_protocol.md) | [設定](../src/wood_degradation_map/experiments/config.py)、[学習入力・mask](../src/wood_degradation_map/experiments/neural.py)、[評価用の共通摂動](../src/wood_degradation_map/experiments/perturbations.py) |
| 3.4と付録B・C | [モデル・損失の説明](../docs/chemomae_positioning.md)、[モデル・学習設定](../docs/design/experiment_protocol.md) | [モデル構築と抽出](../src/wood_degradation_map/experiments/neural.py)、[学習loop](../src/wood_degradation_map/experiments/training.py) |
| 3.5と付録C | [クラスタリング仕様](../docs/design/experiment_protocol.md) | [baseline変換](../src/wood_degradation_map/experiments/baselines.py)、[クラスタリング](../src/wood_degradation_map/experiments/clustering.py)、[マップ作成pipeline](../src/wood_degradation_map/experiments/cluster_pipeline.py) |
| 4.1〜4.3、4.6 | [実験プロトコル](../docs/design/experiment_protocol.md)、[決定記録](../docs/design/decisions.md) | [条件・seed](../src/wood_degradation_map/experiments/config.py)、[manifest](../src/wood_degradation_map/experiments/manifests.py)、[学習](../src/wood_degradation_map/experiments/training.py) |
| 4.4〜4.5と付録D | [評価指標](../docs/design/evaluation_metrics.md) | [LLA](../src/wood_degradation_map/experiments/spatial_metrics.py)、[LFR](../src/wood_degradation_map/experiments/lfr.py)、[silhouette・ARI](../src/wood_degradation_map/experiments/diagnostic_metrics.py)、[集計](../src/wood_degradation_map/experiments/aggregation.py) |
| 4.7と付録E | [全体fitと解釈](../docs/design/visualization_and_interpretation.md)、[OOF sanity](../docs/design/oof_sanity_visualization.md) | [ToDo](../ToDo.md)。OOFの[既存実装](../src/wood_degradation_map/experiments/oof_sanity.py)は全体fitの代用にしない |
| 実施状況、最終の再現条件 | [runbook](../docs/experiment_runbook.md)、[検証履歴](../docs/verification_history.md) | 採用runのconfig・manifest・環境・source hash・完了記録 |

構成案作成時のコード照合では、前処理の順序・除外規約、条件IDと固定設定、clean target、明示的なmask、全帯域可視での抽出、固定中心、共通の評価摂動、LLAの近傍対集計、試料macro集計を確認した。これは設計文書・実装・既存成果物の一部のread-only確認であり、学習・評価・前処理・図の生成・テストの実行や外部文献の再調査を伴わない。上表には本文執筆時の追加照合先も含む。ChemoMAE内部の数式・層構成の照合範囲は、各原稿末尾の執筆メモに記録する。

## 3. 進める順番と各段階の完成物

次の表は、Methods草稿を組み立てる順序と各段階の完成物を示す。日数やページ数は固定しない。
現在は段階1〜3と段階4の3.5に対応する本文・付録の第1稿、および段階5のLLA・補正LLAの原稿があり、図表の仕上げは残っている。
既存草稿の見直しと、第4章・付録Dの執筆をこの対応に沿って進める。

| 順番 | 書く範囲 | 完成物 | 後回しにできるもの |
| --- | --- | --- | --- |
| 1 | 3.1の短い概要、3.2、付録Aの定義 | 前処理の本文草稿、主要条件表、処理図のラフとcaption案 | 試料由来、全試料図、図の最終体裁 |
| 2 | 3.3、付録BのSNV・TGN・shift | 提案摂動の定義・仮定、導出、操作例の図案 | 効果の優劣、化学状態保存の未検証な主張 |
| 3 | 3.4、付録Bの潜在・decoder、付録C | モデル図案、学習課題・loss・利用時の説明、設定表 | attentionの逐次展開、最終実行環境一覧 |
| 4 | 3.5、4.1〜4.3 | マップ作成手順、条件表、CV図案、主要な学習・K設定 | vMFの未決定な数値条件 |
| 5 | 4.4〜4.5、付録D | 指標の役割表、主要式、近傍・LFR図案、集計手順 | 結果の数値と比較の結論 |
| 6 | 4.6〜4.7の確定部分 | 補助実験の問い、CVと全体fitの区別、スペクトル要約・表示の設計 | vMF・全体fitの実施報告、FT-IRの未確定条件 |
| 7 | 第3・4章の通読 | 重複除去、記号統一、本文から付録への参照、未確定箇所一覧 | Materialsの情報待ちで全体を止めない |

本文の節と対応する付録を一組として見直し、定義から詳細へ無理なく進めるかを確認する。
図表はその説明を支えるものとして選び、各図が伝えることをcaptionの一文にしてから作成・再作図する。

各節は「目的 → 入力・記号 → 操作・式 → 採用条件 → 出力・適用範囲」の順で下書きする。執筆メモには参照資料と実装箇所を残し、未確認箇所を黙って補完しない。

## 4. 本文・付録・研究記録の分担

| 判断 | 掲載先 |
| --- | --- |
| 研究の問い、提案の違い、主要な結果を理解するために必要 | 本文 |
| 結論の解釈を変える前提・制約・失敗や退化の要約 | 本文 |
| 定義を追って導出・再現・厳密な確認をしたいときに必要 | 付録 |
| 同じ主張を補足する多数の試料例・反復別・全条件の図表 | 付録。ただし主要な反例や傾向の逆転は本文にも報告 |
| CLIの全help、デバッグtrace、全ログ、実験再開の作業メモ | 既存のリポジトリ記録 |

数式だから付録、画像が多いから付録、という機械的な分け方はしない。本文で定義を一度示し、付録ではその式を参照して導出・例外条件を補う。

## 5. 数式の配置計画

既存の第1稿では仮の式番号を用いている。未執筆部分も、内容と置き場所を決めてから番号を付ける。
本文では記号を式の直前・直後で説明し、何のための式かを文章でつなぐ。以下は本文と付録の分担表であり、新しい定義の採用ではない。最終稿ではLaTeXのlabel/refへ移す。

| 対象 | 本文に残す式・説明 | 付録へ置く内容 |
| --- | --- | --- |
| 反射率変換 | white/darkによる変換、入力の意味 | referenceの列対応、無効値、cutoff proxyの詳細 |
| SNV | 標本標準偏差による定義、平均ゼロ・一定normの性質 | 制約の導出、補間・分散・数値保護の細部 |
| TGN | 接方向と回転の定義、角度の分布 | 射影・直交性・制約保存の導出、角度と弧長、退化ケース |
| Fractional Shift | 補間操作と再中心化・再正規化、shift幅の分布 | チャネル補間・端点値延長の展開、極小normの扱い |
| Encoder・decoder | 単位潜在とbias付き線形再構成 | 必要に応じたattention等の標準式、復元範囲・潜在自由度 |
| Masked denoising | 追加摂動前targetに対するmasked MSE、A0の全領域lossとの差 | batch平均等の細部。lossの定義を付録だけに置かない |
| Cosine-KMeans | 類似度、固定中心への割当 | 初期化・中心更新・停止・数値保護 |
| LLA | クラスタ指示関数・近傍カーネル・畳み込みによる一致率、有効近傍対での正規化、ゼロ拡張と未定義条件 | 近傍対の数え上げとの等価性の展開、境界の具体例 |
| LFR | 固定モデルでのlabel flip率、摂動条件 | drawの共有規約と計算の詳細 |
| 補正LLA・silhouette・ARI | 役割と解釈範囲。補正LLAは帰無期待値・補正式・未定義条件を本文に記載 | 帰無期待値の詳細な導出、他の診断の全定義式、計算対象・退化条件の一覧 |
| 集計・比較 | 試料macro、対応差、2×2交互作用、SDの区別 | 全集計式、共通対象の判定、ARIの別集計 |
| vMF【数値仕様保留】 | 実施後、依存性の検討に必要な密度・割当の説明 | EM、数値計算、集中度、停止・退化処理 |

本文と付録の表記は[共通記号表](notation.md)を基準とする。画素・patch・クラスタの添字、測定bandと補間後のチャネル、SNVのnormと潜在の単位norm、reference由来proxyと学習率を区別する。各節の初出でも記号の意味を一言添え、付録では本文と同じ量・同じ記号を使う。添字を省略するときはその旨を明記する。未執筆の評価指標に記号を追加する際もこの表と照合し、既存の別の量へ流用しない。量の定義や研究条件を原稿の表記変更に合わせて変えない。

Markdownの数式はKaTeX互換のinline式またはdisplay式で記述する。独自macroや、数式を代用するcode blockは導入しない。

## 6. 図・表の配置計画

IDは執筆用の仮識別子とし、最終的な図番号・表番号ではない。枚数を埋めるために作るのではなく、伝える内容が重なる図は統合する。

| 仮ID | 内容と伝えること | 本文での位置 | 付録側の補足・状態 |
| --- | --- | --- | --- |
| F-flow | HSIから表現・マップへ進む全体像、学習と利用の違い | 3.1 | [参考図と作図記録](figures/analysis_workflow_notes.md)あり。最終図は再作図 |
| F-preprocess | 入力画像、mask、前処理前後のスペクトルの対応 | 3.2。F-flowと分ける必要がある場合 | 全試料・詳細診断はA。実例の選択と出典を記録 |
| F-augmentation | TGNとshiftで同じ観測がどう変わるか | 3.3 | 多数例・候補強度の比較はA。採用条件の説明図を用意 |
| F-model | patch、CLS、単位潜在、decoder、target・lossの流れ | 3.4 | 内部層の詳細は表で補足 |
| F-cv | 試料分割、trainのみのfit、固定test推論、3反復 | 4.2 | split・seed等の再現情報はC |
| F-metrics | LLAの有効近傍、clean/perturbed間のLFRの数え方 | 4.4 | 背景・境界・退化の追加例はD。小さな模式図で十分 |
| T-conditions | 主7条件で何が異なり、何を比較できるか | 4.1 | 全contrastと補助実験の詳細はD・E |
| T-recipe | 主なモデル・学習・Kの条件 | 3.4、4.3で役割を分ける | 全設定をCに集約 |
| T-metrics | 主評価・診断・必須併記、改善方向と解釈範囲 | 4.4 | 退化・未定義理由の詳細表はD |
| F-main-results | 主要比較、全Kの傾向、paired差、occupancy | 5.2〜5.3 | 最終報告用pipelineは未実装。反復別・追加contrastはE |
| F-maps | 固定7代表試料のマップとスペクトルの対応 | 5.5 | 全49試料をE等で提示する案。全体fitは実施待ち |
| F-ftir | 測定位置と対応するNIR・FT-IR観測 | 2.3、5.6 | 詳細設計・実施後に具体化 |

本文用の代表7試料は既存設計で指定済み。各樹種で保存有効画素数が最大の試料であり、原稿の見栄えや結果に合わせて再選択しない。本文ではこの7試料を複数の図に分けて配置できる。全体fitの各試料図は5条件×2手法という現行設計を維持し、読みやすい大きさで掲載する。

方法の操作説明に別のスペクトル例を使う場合は、本文結果の代表例とは役割を区別して選択根拠を残す。

## 7. 既存図の候補と使い方

構成案作成時にファイルの存在を確認した候補を挙げる。その時点で画像内容まで確認したのはcutoff図とTGN固定角度の例図である。cutoff図は現在の付録Aの第1稿に掲載しており、他の候補は採用時に図・設定・captionを照合する。掲載済みの図も最終稿での体裁と出典の確認を要する。

| 候補 | 出典 | 掲載案・注意点 |
| --- | --- | --- |
| 波長cutoffの確認 | [cutoff_decision.png](../outputs/preprocessing/production_v1/cutoff_decision.png) | 付録Aの図A.1に掲載済み。reference proxyと試料側診断が同じ図にあるため、cutoffを決めたのはreference側だけと明示 |
| 補間後の帯域分布 | [反射率](../outputs/preprocessing/production_v1/interpolated_reflectance_band_distribution.png)、[SNV](../outputs/preprocessing/production_v1/interpolated_snv_band_distribution.png) | 付録A。最終入力の記述的な確認 |
| 前処理後の候補画素 | [final_snv_anomaly_candidates.png](../outputs/preprocessing/production_v1/final_snv_anomaly_candidates.png) | 付録Aの候補。確認用の候補選択であり、異常・劣化の正解ラベルとしない |
| TGNの固定角度の例 | [snv_noise_exact_angles_examples.png](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_noise_exact_angles_examples.png) | 付録A。2.5・5・7.5度の比較であり、本実験の角度一様分布を示す図そのものではない |
| Shiftの固定幅の例 | [snv_shift_exact_endpoints_examples.png](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_shift_exact_endpoints_examples.png) | 付録A。候補幅と採用した分布をcaptionで区別 |
| 摂動の候補分布 | [noise](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_noise_uniform_ranges_distributions.png)、[shift](../outputs/sanity_checks/augmentation_strengths_train_fold1/snv_shift_uniform_ranges_distributions.png) | 付録A。強度のsanity checkであり、CV指標による最適化とは書かない |
| B0・B1のOOFマップ | [B0](../outputs/sanity_checks/b0_b1_oof_visualization/labels/B0_representatives_k8_repeat1.png)、[B1](../outputs/sanity_checks/b0_b1_oof_visualization/labels/B1_representatives_k8_repeat1.png) | 付録Eの途中確認資料候補。fold内B0基準であり、全体fitのA0基準の最終図とは別 |
| B0・B1のsilhouette | [silhouette_k_sweep.png](../outputs/sanity_checks/b0_b1_oof_visualization/silhouette_k_sweep.png) | 付録Eの候補。全主条件の性能比較として提示しない |

摂動sanityの元は[スクリプト](../scripts/experiments/sanity_check_augmentation_strengths.py)と[summary](../outputs/sanity_checks/augmentation_strengths_train_fold1/summary.json)にある。fold 1のtrainから8試料×128画素を確認する設計で、outer-testや学習モデルの評価を使う処理ではない。採用していない強度も含むので、単なる「学習augmentationの例」として全図を本文に貼らない。

採用図ごとに、原稿の執筆メモへ次を残す。

- 伝えたいこと、本文・付録の置き場所、caption案。
- 元のファイル・生成スクリプト・config・manifestと、その版やsource hash。
- 試料ID、画素またはROI、fold・反復・K・条件・手法、OOFか全体fitか。
- 実測、観測の要約、人工摂動、模式図のどれか。単位、凡例、色・ラベル整列の基準。
- 選択の根拠と時点、作成済み／再作図予定／結果待ちの状態。

図表を採用するときに `figures/`・`tables/` へ配置し、原本の `outputs/` は保持する。captionと出典メモは原稿側の一か所で管理する。

## 8. 後で確認する情報と、今の執筆を止めない境界

| 状態 | 対象 | 原稿上の扱い |
| --- | --- | --- |
| 情報待ち | 試料由来・採取関係・状態、撮像装置と測定条件 | 第2章に執筆メモ。第3・4章の確定した処理は進める |
| 定義・実装を参照して書ける | 前処理、主条件、学習、Cosine-KMeans、CV・指標・集計 | 実際の研究条件として下書きするが、全run完了とは書かない |
| 条件Fixed・完了確認が必要 | mask率補助実験、最終の全主条件比較 | 方法を先に記述し、結果は完了記録を待つ |
| 一部Open | vMF数値仕様・専用pipeline | 問いと確定した比較範囲のみ記述。既定値で埋めない |
| 設計あり・pipeline未実装 | 全体fitと最終報告用図表 | 設計の下書きと図表予約。実施済みとしない |
| 詳細Open | FT-IR、任意形状診断・責務マップ | 未確定メモ。新しい指標や処理を原稿の都合で採用しない |
| 最終稿で確認 | 実行環境、採用runと図表、文献の引用箇所・書誌、提出書式 | 該当部分の仕上げ時に照合 |

この状態表は本計画作成時に読んだ資料に基づく。実験は進行し得るため、原稿確定時は最新のrun記録とToDoを参照し、完了数や実行状況を本書だけから転記しない。CV開始後の追加判断は[決定記録](../docs/design/decisions.md)に従い、すべてが開始前から決まっていたようには記述しない。

## 9. 第3・4章を合わせたMethods草稿の完了条件

以下はMethods全体の見直し項目であり、第3章の草稿があることだけで完了とはしない。

- [ ] 試料の未整理情報を補わずに、前処理からマップ作成までの流れが読める。
- [ ] 提案摂動、潜在・decoder、target・lossの主要な定義が本文にある。
- [ ] 比較条件、train/testの範囲、反復、K、指標の役割が本文で分かる。
- [ ] CVによる比較と全体fit・FT-IRによる解釈を混同していない。
- [ ] 詳細の付録には本文からの参照があり、元資料・実装との対応を追える。
- [ ] 図のcaption案があり、模式図・実測例・人工摂動・結果を区別している。
- [ ] 未確定と未実施が明示され、検証結果や研究上の結論を先取りしていない。
