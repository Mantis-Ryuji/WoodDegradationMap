# ドキュメント案内

研究の目的、解析仕様、再現手順、結果の読み方をまとめる。

## 研究と手法

| 文書 | 内容 |
| --- | --- |
| [研究概要](research_overview.md) | 研究の問い、解析の構成、結果と主張の範囲 |
| [ChemoMAEの位置づけ](chemomae_positioning.md) | masked denoising、単位潜在、アフィンdecoderの役割 |
| [関連研究](related_work.md) | 参照手法と本研究の関係、参考文献 |
| [結果の読み方](interpretation_notes.md) | マップ・教師なし指標・スペクトル解釈の限界 |
| [数理的補足](mathematical_notes.md) | SNV、摂動、loss、表現幾何の導出 |
| [数値実装の補足](numerical_implementation_notes.md) | 混合精度、学習loss、クラスタリングの数値処理 |

## 仕様と再現手順

| 解析 | 仕様 | 実行・成果物 |
| --- | --- | --- |
| 古材NIR-HSI | [研究設計一覧](design/README.md) | [実験手順](experiment_runbook.md)・[成果物案内](results.md) |
| Mask率感度解析 | [実験条件](design/experiment_protocol.md)・[全体fit条件](design/visualization_and_interpretation.md#global-mask-rate) | [CV手順](experiment_runbook.md#mask-rate-sweep)・[全体fit手順](experiment_runbook.md#global-mask-rate) |
| 空間マップ再現性 | [評価仕様](design/spatial_map_reproducibility_plan.md) | [専用手順](spatial_map_reproducibility_runbook.md) |
| Ramanクラスタリング | [Bacteria-ID仕様](design/bacteria_id_experiment_plan.md) | [実行手順](bacteria_id_runbook.md) |
| Raman分類 | [HPO仕様](design/bacteria_id_hpo_protocol.md) | [実行手順](bacteria_id_hpo_runbook.md)・[解釈](bacteria_id_interpretation_plan.md) |

研究条件・計算方法・データ契約は`docs/design/`に、操作方法は各実行手順に記載する。
実測値と実行由来は、各出力のconfig・manifest・run記録・完了記録に対応付ける。
`data/`・`outputs/`以下はGit管理対象外であり、文書中のパスはローカルの入力・出力先を示す。

## 用語

| 用語 | 本リポジトリでの意味 |
| --- | --- |
| raw SNV / B0 | 前処理済み256次元SNVを直接使うbaseline。センサのraw強度ではない |
| 試料 | `KYOw...`で識別する分割・集計単位。同一元部材との対応や使用時の位置関係は確定していない |
| CVのrepeat | 同じsplit・画素集合に対する3回の学習・クラスタリング反復 |
| 全体fitのKMeans repeat | 各条件の表現を固定し、K8の初期化だけを変える3反復。NN再学習のばらつきは含まない |
| draw | LFR用に各摂動を5回生成する評価摂動反復 |
| OOF | 各試料をtrainに含めないfoldモデルから得る予測・評価 |
| 全体fit | 全49試料からの共通抽出画素でfit・学習し、全試料を記述する解析 |
| matching | クラスタ所属を変えずに表示番号を対応付ける操作。劣化順序や共通の化学的意味を与えない |
| clean | 追加noise・shiftを加える前の観測。測定ノイズのない真値ではない |

OOF図のfold内B0基準の整列と、全体fit図のM00基準の整列は異なる。
それぞれ[OOF診断図](design/oof_sanity_visualization.md)と[全体fitの可視化仕様](design/visualization_and_interpretation.md)に定義する。
