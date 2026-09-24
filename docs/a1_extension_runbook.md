# A1（denoising AE）追加の実行手順

**2026-09-25：ユーザー完了報告により、A1のCV 15 runs、主8条件OOF・図表の再生成、
全体fit・K8マップ・代表図6×7・PCA図2×6まで完了。** 保存先は既存の`main_oof_v1`・`global_k8_v1`を上書きした。
残るToDoはmask ratio sweepと位置対応FT-IRのみ（[ToDo](../ToDo.md)）。
今回の文書更新では学習・テスト・成果物checkを再実行していない。
以下はA1追加時の操作記録として残す。完了済みの学習・記録更新を最初からやり直す必要はない。

2026-09-22の追加条件。A1はA0と同じChemoMAE構成・mask 0%・全領域MSEで、
M11と同じTGN＋shift（各適用確率0.5、角度0–5度、shift −2–2）を入力に加え、
clean SNVを再構成する。推論はclean・全可視。学習budgetは800 epoch、latentは16次元。

主CVはB0・B1・A0・A1・M00・M10・M01・M11の8条件、全体fitは
B0・B1・A0・A1・M00・M11の6条件とする。追加比較はA1−A0とM11−A1。
M11−A1はmaskとloss対象の両方の変更を含む。既存のTGN×shift交互作用と主要3比較は保持する。
この追加は既存7条件の結果を得た後のablationとして扱う。

追加計算はCVの15学習・105 KMeans fits・15評価と、全体の1学習・K8の1 fit。
学習・clustering・評価・PCA用の表現抽出はGPU、OOF集計・図表・PCA projectionはCPUを使う。
PCA抽出は共通fit画素、全体のスペクトル図は全49試料の有効画素を読み直す。

通常の読込処理は現行config・codeとの一致を要求し、旧形式の互換分岐は持たない。
最初に下記の一度きりの更新で保存済みconfig・検証用contract・参照ハッシュを現行仕様へ揃える。
split・抽出画素・seed・重み・評価値と、実際の実行環境を記録した`execution`は保持する。
変更前JSONと変更前後のハッシュは各experimentの`config/a1_contract_update.json`へ記録する。
OOF集計・図表・PCAはv1保存先を上書きする。旧成果物をv2へ複製しない。

## 1. 検証と短い動作確認

以下はリポジトリrootのPowerShellでユーザーが実行する。各段階が非0終了なら停止する。
テストはsynthetic dataを使い、実データの学習は行わない。

```powershell
uv run --no-sync pytest tests/experiments/test_a1_records.py tests/experiments/test_artifact_output.py tests/experiments/test_records.py tests/experiments/test_manifests.py tests/experiments/test_neural.py tests/experiments/test_training.py tests/experiments/test_cluster_pipeline.py tests/experiments/test_evaluation_pipeline.py tests/experiments/test_oof_pipeline.py tests/experiments/test_oof_reporting.py tests/experiments/test_global_fit.py tests/experiments/test_global_analysis.py tests/experiments/test_global_pca.py
if ($LASTEXITCODE -ne 0) { throw 'A1 tests failed' }

uv run --no-sync python scripts/experiments/refresh_a1_records.py --scope cv --experiment-dir outputs/experiments/production_v1 --apply
if ($LASTEXITCODE -ne 0) { throw 'CV A1 record update failed' }
uv run --no-sync python scripts/experiments/refresh_a1_records.py --scope global --experiment-dir outputs/experiments/global_v1 --apply
if ($LASTEXITCODE -ne 0) { throw 'Global A1 record update failed' }
uv run --no-sync python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
if ($LASTEXITCODE -ne 0) { throw 'CV manifest check failed' }
uv run --no-sync python scripts/experiments/train_neural.py smoke --condition A1 --fold 1 --repeat 1 --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'A1 smoke failed' }
```

`refresh_a1_records.py`は`--apply`を省くと変更予定だけを表示する。完了済みrunと既知のA1差分を
全件検査してからJSONを更新し、学習や評価は実行しない。未完了・欠損・アクセス不能の記録があれば停止する。
smokeは本番とは別の出力へ保存する。既存条件の学習を再開する手順ではない。
旧optimizer checkpointの内部記録は更新しないため、旧完了学習を`--resume`する用途には使わない。

## 2. A1だけのCV追加

1 GPUで直列実行する。中断時は完了済みの組を再実行せず、既存runbookの明示的な
`train_neural.py --resume <同じrunのlast.pt>`で該当A1学習を再開する。
clustering・評価の既存出力は`check`で確認する。

```powershell
$experiment = 'outputs/experiments/production_v1'
foreach ($fold in 1..5) {
    foreach ($repeat in 1..3) {
        uv run --no-sync python scripts/experiments/train_neural.py train --condition A1 --fold $fold --repeat $repeat --experiment-dir $experiment
        if ($LASTEXITCODE -ne 0) { throw "A1 training failed: fold=$fold repeat=$repeat" }
        uv run --no-sync python scripts/experiments/cluster_representations.py run --condition A1 --fold $fold --repeat $repeat --experiment-dir $experiment
        if ($LASTEXITCODE -ne 0) { throw 'A1 clustering failed' }
        uv run --no-sync python scripts/experiments/cluster_representations.py check --condition A1 --fold $fold --repeat $repeat --experiment-dir $experiment
        if ($LASTEXITCODE -ne 0) { throw 'A1 clustering check failed' }
        uv run --no-sync python scripts/experiments/evaluate_representations.py run --conditions A1 --fold $fold --repeats $repeat --experiment-dir $experiment
        if ($LASTEXITCODE -ne 0) { throw 'A1 evaluation failed' }
        uv run --no-sync python scripts/experiments/evaluate_representations.py check --conditions A1 --fold $fold --repeats $repeat --experiment-dir $experiment
        if ($LASTEXITCODE -ne 0) { throw 'A1 evaluation check failed' }
    }
}
```

## 3. 主8条件のOOFと図表

旧7条件の完了済み評価を再利用し、`main_oof_v1`を`--overwrite`で再生成する。
新集計が成功してから保存先を置き換える。主条件の図表も同じv1保存先へ上書きする。

```powershell
uv run --no-sync python scripts/experiments/aggregate_oof.py run --conditions B0 B1 A0 A1 M00 M10 M01 M11 --snapshot main_oof_v1 --overwrite --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'A1 OOF aggregation failed' }
uv run --no-sync python scripts/experiments/aggregate_oof.py check --snapshot main_oof_v1 --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'A1 OOF check failed' }
uv run --no-sync python scripts/experiments/visualize_main_oof.py --snapshot main_oof_v1
if ($LASTEXITCODE -ne 0) { throw 'A1 OOF figures failed' }
```

完了記録は49試料・120 source runs・82,320 score records。図表は
`outputs/experiments/production_v1/results/figures/main_oof_v1/`へPNG 3枚・CSV 11個。
主指標図は既存の単段幅2×2を維持し、8条件を表示する。paired図・CSVにはA1−A0、M11−A1を追加する。

## 4. 全体fit・K8マップ・PCA

既存の全体manifestとB0/PCA・A0/M00/M11重みを再利用し、A1だけ学習する。
以下のCLIの既定experimentは`outputs/experiments/global_v1`。

```powershell
uv run --no-sync python scripts/experiments/global_fit.py check
if ($LASTEXITCODE -ne 0) { throw 'Global manifest check failed' }
uv run --no-sync python scripts/experiments/global_fit.py baseline-check
if ($LASTEXITCODE -ne 0) { throw 'Global baseline check failed' }
uv run --no-sync python scripts/experiments/global_fit.py smoke --conditions A1
if ($LASTEXITCODE -ne 0) { throw 'Global A1 smoke failed' }
uv run --no-sync python scripts/experiments/global_fit.py train --conditions A1
if ($LASTEXITCODE -ne 0) { throw 'Global A1 training failed' }
uv run --no-sync python scripts/experiments/global_fit.py training-check --conditions A1
if ($LASTEXITCODE -ne 0) { throw 'Global A1 training check failed' }
uv run --no-sync python scripts/experiments/global_cluster.py run --conditions A1
if ($LASTEXITCODE -ne 0) { throw 'Global A1 clustering failed' }
uv run --no-sync python scripts/experiments/global_cluster.py check
if ($LASTEXITCODE -ne 0) { throw 'Global six-condition check failed' }
uv run --no-sync python scripts/experiments/visualize_global.py run
if ($LASTEXITCODE -ne 0) { throw 'Global A1 figures failed' }
uv run --no-sync python scripts/experiments/global_pca.py prepare --overwrite
if ($LASTEXITCODE -ne 0) { throw 'Global A1 PCA inputs failed' }
uv run --no-sync python scripts/experiments/global_pca.py fit --overwrite
if ($LASTEXITCODE -ne 0) { throw 'Global A1 PCA fit failed' }
uv run --no-sync python scripts/experiments/global_pca.py check
if ($LASTEXITCODE -ne 0) { throw 'Global A1 PCA check failed' }
uv run --no-sync python scripts/experiments/visualize_global_pca.py run
if ($LASTEXITCODE -ne 0) { throw 'Global A1 PCA figure failed' }
uv run --no-sync python scripts/experiments/visualize_global_pca.py check
if ($LASTEXITCODE -ne 0) { throw 'Global A1 PCA figure check failed' }
```

全体図表は`results/figures/global_k8_v1/`にPNG 67枚・CSV 44個を上書きし、代表比較は6×7。
ファイル名は`01_representative_samples_6x7.png`で、旧`01_representative_samples_5x7.png`は
図表の再生成成功時に除去される。行順はB0・B1・A0・A1・M00・M11、列は従来の7代表試料。
PCAは`checkpoints/pca_projection/global_k8_v1/inputs/`と`results/pca/global_k8_v1/`へ上書きし、
図は全体図表配下の`pca-latent-2d/`に2×6のPNG 1枚・CSV 5個を出力する。
M00基準のSNV整列とB1の未正規化PCA得点は従来どおり。
各図の凡例・条件数・文字の重なりを確認する。全体fit図は記述的な可視化であり、OOF評価とは区別する。
