# 研究・実装・執筆 ToDo

更新日: 2026-09-12（文書・実装計画・執筆作業を同期）。実験の完了数と次のrunは、
2026-09-12までに確認・共有された状態を示す。今回の文書更新では学習・評価・成果物checkを再実行していない。

本書は残作業を管理する。[文書案内](docs/README.md)から、
[固定設計](docs/design/README.md)、[実行手順](docs/experiment_runbook.md)、[検証履歴](docs/verification_history.md)へ進む。
原稿と資料の対応は[修論ドラフト](thesis/README.md)・[執筆計画](thesis/writing_plan.md)を参照する。

## 文書整理を反映した作業の見取り図

主CVの条件・loss・split・seed・K・主評価は現行設計を継続する。今回の整理では、
確定した成果物を作る実装と、解釈のために採用を検討する診断、原稿の整備を次のように対応づけた。

| 作業 | 実装・執筆への反映 | 着手・完了に必要なもの |
| --- | --- | --- |
| 主CV・mask率補助（第1〜2節） | 既存CLIで残runとOOF集計を完了する | 各runの完了・checkとsnapshotの完全性 |
| vMF補助（第3節） | 既存表現を再利用する専用fit・評価・check・OOFを実装する | Openの数値仕様の確定と検証 |
| 最終報告用図表（第4節） | 比較の問いと、主表・診断表・曲線・paired差を対応づける | 固定報告規約に沿う実装と、図表生成時の完了OOF snapshot |
| 全体fit・解釈（第5節） | 共通モデルのfitからマップ・観測スペクトルの要約までを実装する | 全体fit用manifest・記録・推論経路。vMF部分は数値仕様確定後 |
| SVD・復元誤差等の診断候補（第6.2節） | 答えたい問いと採用範囲を決めてから、必要な診断を実装対象にする | 現時点では未採用。本文で使う説明・出力・評価対象の決定 |
| 論文原稿（第7節） | 第3章・付録A〜Cを基に、第4章・付録Dと図表を整える | 固定仕様から書ける部分を進め、結果の記述は実施記録と照合する |

数理的な導出を記載したこと自体を、新しい解析の実施決定とはしない。
実装する際は、成果物が本文のどの問いを支えるかを明示し、詳細な仕様は対応する設計文書で管理する。

## 現在の状態と次のrun

| 項目 | 確認済みの状態 |
| --- | --- |
| 本番入力 | `data/processed/production_v1/`、49試料、3,902,250有効画素 |
| 本番root | `outputs/experiments/production_v1/` |
| manifest | split・共通train座標・augmentation contractは現行仕様で確定済み |
| baseline | B1 PCAの5 fits、B0・B1の全5 folds×3反復のclustering・評価・checkが完了 |
| 主ニューラル条件 | A0のfold 1–4、M11のfold 1–2、M00・M10・M01のfold 1の各repeat 1–3で学習・clustering・評価・checkが完了 |
| 主実験の完了数 | NN学習・clustering・評価は各27/75。B0・B1を含むclustering・評価は各57/105組合せ |
| OOF sanity | B0・B1のPNG 3枚・CSV 3つを生成済み。[表示仕様](docs/design/oof_sanity_visualization.md) |
| 実行中 | なし（最終確認時点） |
| 次のrun | A0・fold 5・repeat 1–3。同repeatの学習 → clean test map → clustering checkを順次実行し、3反復をまとめて全test評価 → evaluation checkまで完了する |
| 実行環境 | ChemoMAE v0.2.2。固定設定と環境確認は[runbook](docs/experiment_runbook.md)・[検証履歴](docs/verification_history.md) |

前処理・入力照合、学習と再開、clustering、評価、OOF数値集計の実装とpreflightは完了済み。
本文の代表7試料も固定済み。詳細な完了記録を本書へ重複掲載しない。

## 1. 主ニューラルCVの残り

各runは800 epochとし、正常完了した重みからclean test mapと評価を作り、各checkまで完了する。
[1 runの手順](docs/experiment_runbook.md#neural-run)を使用する。

- [ ] A0のfold 5・repeat 1–3を完了する（12/15 runs完了。fold 1–4は完了）。
- [ ] M00のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
- [ ] M10のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
- [ ] M01のfold 2–5・repeat 1–3を完了する（3/15 runs完了）。
- [ ] M11のfold 3–5・repeat 1–3を完了する（6/15 runs完了。fold 1–2は完了）。
- [ ] 主7条件の5 folds×3反復の完全性を確認し、`main_oof_v1`を作成・checkする。
- [ ] 欠損・失敗・中断・未定義指標と理由がOOF集計に保持されていることを確認する。

## 2. Mask率補助実験

- [ ] M11-25の5 folds×3反復を学習し、clustering・評価・checkまで完了する（15 runs）。
- [ ] M11-75の5 folds×3反復を学習し、clustering・評価・checkまで完了する（15 runs）。
- [ ] 50%は主実験M11の結果を再利用し、3条件の完全性を確認して`mask_rate_oof_v1`を作成・checkする。

主条件とmask率のOOFは[runbookのOOF集計](docs/experiment_runbook.md#oof-aggregation)に従い、別snapshotにする。

## 3. vMF補助実験

主7条件×5 folds×3反復×7Kの735 fits。追加のNN学習・PCA fitは行わない。
範囲・利用版・退化成分の扱いは決定済みで、数値仕様と専用pipelineは未完了。
[実験プロトコル](docs/design/experiment_protocol.md#vmf-supplementary)と[評価規約](docs/design/evaluation_metrics.md#vmf-evaluation)に従う。

- [ ] v0.2.2の数値関数・公開helper・初期化・最終尤度・保存復元・退化成分を検証する。
  16次元・256次元の参照値比較、CPU小規模、chunk、GPU最小確認を含む。
- [ ] 数値精度・EM停止条件・集中度設定をvMFのtest結果を見る前に固定する。
- [ ] 専用のfit・評価・check・OOFと独立した出力先を設計・実装する。
- [ ] 本番CV後、既存の主7条件の重み・PCAと共通train画素で735 fitsを実施する。
- [ ] 同じtest全画素・共通摂動で評価し、完了・失敗・未定義値を保持してOOF集計する。
- [ ] 計画contrast・2×2交互作用・K依存性をCosine-KMeansと併記する。

## 4. 全CV完了後の図表生成

この節と次節の実装はAstraへ切り替えて進める予定（ユーザー指定）。
作業時のモデル選択であり、研究条件や成果物の再現性要件には含めない。
B0・B1 OOF sanityは生成済みだが、以下の最終報告用図表pipelineは未実装。
修論の結果章5.2〜5.4と付録Eへ渡す成果物として、[執筆計画の図表対応](thesis/writing_plan.md)と照合する。

- [ ] OOF snapshotを入力に、代表$K_0=8$の主表と、補正LLA・silhouette・ARI・occupancy・有効対象数の診断表を生成するpipelineを実装する。
- [ ] 全7Kの主指標・主要contrast・反復別曲線、mask率依存性、paired差、2×2交互作用の図表を実装する。主要比較はM11対B0・B1・M00とし、残りの計画比較も保持する。
- [ ] 未定義理由と共通対象数を保持し、試料間SD・反復間SD・ARIの集計を[報告規約](docs/design/evaluation_metrics.md#reporting)に合わせる。
- [ ] 元snapshotと図表のsource hash、条件・K・反復・集計対象、captionに必要な定義を保存し、原稿の図表から根拠をたどれるようにする。
- [ ] 完了した主条件・mask率の各OOF snapshotから図表を生成し、必須の表・図と原稿の掲載候補を照合する。vMFの比較図表は同補助実験の完了後に作成する。

## 5. 全体fitと解釈

対象はB0・B1・A0・M00・M11の5条件×Cosine-KMeans・vMFの2手法。
共通の表示番号は**A0＋Cosine-KMeans**を基準にする。[全体可視化設計](docs/design/visualization_and_interpretation.md)に従う。
全体解釈pipelineは未実装で、既存のCV用CLIをそのまま全体fitへ使わない。

- [ ] 全49試料の画素抽出・manifest・seed適用・実行記録・保存先・CLIを実装する。
- [ ] PCAの全体fit、A0・M00・M11の各1回（計3回）の全体学習を実装・実施する。モデル・学習条件は既存実装を再利用する。
- [ ] 同じ表現・抽出座標・$K_0=8$でCosine-KMeansを5 fits行う。
- [ ] 数値仕様確定・検証後、同じ表現を再利用してvMFを5 fits行う。CV補助実験の735 fitsとは分ける。
- [ ] 全49試料のhard label map、A0基準のmatching・overlap、occupancy、使用クラスタ数、潜在空間図を保存する。
- [ ] 観測反射率・SNVの代表線を試料内中央値→試料間中央値・四分位範囲で要約し、寄与試料・画素数と差スペクトルの引き算の向きを保存する。Decoderの復元値はこの観測スペクトル集計へ混ぜない。
- [ ] 固定7代表試料を本文表示に使い、行をCosine-KMeans・vMF、列をB0・B1・A0・M00・M11とする比較図、KYOw試料ID、共通の表示規約を確認する。
- [ ] 劣化との対応を探索的に記述し、CVの指標改善と化学的な対応を区別する。

これらは修論の結果章5.5と付録Eに対応する。SVD・再構成残差の診断は第6.2節の候補として扱い、
現在必須の潜在空間図・観測スペクトル要約とは採用状況を分ける。

## 6. 未決定事項

### 6.1 既定計画に残るOpen事項

研究上のOpen事項は[研究設計の一覧](docs/design/README.md#open-items)で確認する。
任意の形状診断・vMF責務マップは、採用する場合だけ結果を見る前に定義を固定する。
位置対応FT-IRは測定計画があり、対象・位置対応・反復・前処理・指標などの詳細設計はOpen。

実行ごとの記録と保存は[runbook](docs/experiment_runbook.md#artifact-records)に従う。
全体fit対象の拡張やmatching基準などの決定時点は[決定記録](docs/design/decisions.md)に残す。

### 6.2 表現と再構成の診断候補（未採用・未実装）

[解釈メモ第8節](docs/interpretation_notes.md#decoder-residual-discussion)と
[付録B.5](thesis/appendices/mathematical_details.md#latent-decoder)で整理した論点を、
今後の採否判断として残す。以下は確定した実行予定ではない。

| 確かめたい問い | 採用する場合の実装候補 | 解釈・比較の条件 |
| --- | --- | --- |
| 小さなtrain lossが、共通条件の復元でも小さいか | 固定重みのA0・M11へ追加摂動なし・全可視の同じ未学習画素を入力し、全チャネル誤差を記録する | 更新中のtrain loss、masked課題、固定モデルの再評価を区別する |
| どのような誤差が、どの波長・場所に残るか | 全チャネル誤差の平均・norm・方向への分解、符号付き波長別残差と画素位置への配置 | 中心化後normと角度の定義域、targetの数値的な制約誤差を確認する。残差を劣化ラベルとみなさない |
| Decoderのどの復元方向を実データが使うか | 特異値・左右特異ベクトル・回転後潜在座標の分布と空間配置を対応づける | 特異値単独を寄与率とせず、座標の変動も見る。反復間の軸は一意でなく、再構成とcosine距離の幾何も異なる |
| 共通形に加えて個々の観測差を復元したか | 学習画素の平均スペクトルを返す参照と、同じ対象画素で復元誤差を比較する | 参照は学習画素だけでfitする。入力全体の大きさと画素間変動の尺度を区別する |
| 固定モデルの演算精度が復元にどの程度影響するか | 同じ重み・入力・可視集合でAMPとFP32のforwardを比較する | 出力差とtargetへの誤差を分ける。FP16での学習履歴全体の影響を分離したとはしない |

採用前に、答える研究上の問い、対象条件・試料・fold・反復・画素範囲、CVか全体fitか、
入力・mask・dtype、集計方法、保存先・必要な図表、本文で述べる範囲を決める。
化学的意味や原因の説明にどの証拠が必要かを確認し、採用した項目だけを設計文書と実装タスクへ反映する。
主評価への追加、再学習、model・lossの変更は、この候補整理から自動的には導かない。

## 7. 論文原稿と図表の整備

章ごとの論点は[構成案](thesis/outline.md)、資料・実装・図表の対応は[執筆計画](thesis/writing_plan.md)を正とする。
ここでは執筆の残作業と、実験・実装への依存を管理する。

- [x] 第3章と付録A〜Cの第1稿を作成し、`docs/`・`thesis/`の論点、本文・付録・執筆メモ、記号・参照を整理した（2026-09-12）。
- [ ] 第4章の比較条件・CV・反復・K・評価・集計と、付録Dを固定仕様に基づいて執筆する。vMFとFT-IRのOpen部分は未確定として残す。
- [ ] 解析フロー参考図を最終作図へ移し、摂動・モデル・CV・指標の説明図と条件表を整える。Caption・記号・図の出典を本文と照合する。
- [ ] 第1章の背景・関連研究・採用理由を、主張できる範囲と引用根拠が対応する文章へ整える。標準手法の原典・書誌を確認する。
- [ ] 第2章に必要な試料由来・採取関係・状態・撮像装置・測定条件を整理する。FT-IRは詳細決定・実施後に対応する記載を整える。
- [ ] 本書の第4〜5節の成果物と実行記録を確認して、第5章・付録Eの結果、第6章の考察、第7章の結論を執筆する。第6.2節の候補を採用した場合は、その実施・確認範囲を反映する。
- [ ] 採用runの環境・重み・source hashと図表の出典、共通記号、引用、提出書式を照合し、LaTeXの最終稿へ移す。
