# 解析フロー参考図の記録

作成日: 2026-09-11

画像は[analysis_workflow_reference.png](analysis_workflow_reference.png)。組込みimage_genで新規生成し、同じツールで二回の修正を行った。最終稿を著者が作図するための構成参考であり、図中の木材・スペクトル・座標・クラスタ分布は生成された模式例である。

## 図の読み方

上段は提案条件を中心とする表現学習、下段は全帯域可視での抽出とマッピングを表す。原SNVから上段の入力と復元targetが分岐する。学習済みencoderの重みを下段へ引き継いで固定し、座標はスペクトル処理を迂回して最後の配置にのみ用いる。

主比較のMAE条件のmask率50%を示す図であり、A0・B0・B1やmask率補助条件を同じ絵の中へ網羅したものではない。背景ラベルは0、木材内のクラスタ表示は1始まりとする。図の4色は模式的な表示であり、実験の代表表示であるクラスタ数 $K=8$ を再現していない。

## 確認と最終作図時の整理

初稿にあったtarget分岐の位置、重み固定の矢印の行き先、SNV軸名、0始まりのラベル例を修正した。最終採用版では、targetは原SNVからMasked MSEへ、固定する重みはtrainable encoderからfrozen encoderへつながる。

最終作図では次を明確にすると読みやすい。

- Cosine-KMeansの枠内を、train表現から中心を求める経路と、各画素の表現を固定中心へ割り当てる経路に分け、各矢印を対応する小箱へ直接接続する。現在の参考図は大枠への入出力を簡略化している。
- 16個のspectral patchとCLSを、本文の構成どおりに描く。参考図のpatch表示は省略記号を含む模式表示である。
- SNVスペクトルは実例を使うか、平均ゼロの概念が伝わる模式線を描き、実測例と模式図を区別する。参考図の波形に数値的なSNV不変量を読み取らない。
- 入力形状の旧ラベルH・Wは行数・列数を意味するが、Wがwhite referenceと重なるため、最終作図では「行数 × 列数 × 256 bands」とする。xはSNVスペクトル、zは学習時の単位潜在であり、下段の全可視潜在は本文の表記にそろえる。以下の生成promptは生成当時の記録として残す。
- 日本語・英語のラベルを原稿と統一する。推論にdecoderが不要であること、座標がencoder入力ではないことを保つ。
- CVの図ではfit範囲をtrainに限定する。全体fitの図として使う場合は対象範囲の表記を改め、両者を同一視しない。

Caption案は[第3.1節](../chapters/3_analysis_methods/overview.md)に置く。

## 新規生成prompt

Use case: scientific-educational.
Asset type: one academic workflow diagram, raster concept reference for a Japanese master's thesis. The author will redraw the final figure.
Create a clean, precise, flat vector-style scientific infographic on a white landscape canvas, approximately 16:9. Generous whitespace, sharp arrows, readable English sans-serif labels. Muted navy for data, orange for training-only elements, teal for frozen inference, categorical muted blue/orange/green/purple for cluster labels. All spectra, cubes and maps are schematic, not experimental results. No photorealistic data, no quantitative performance claims, no logos.

Scientific content and layout:
A shared input block at the left contains a small wood-shaped hyperspectral cube, label "NIR-HSI". Arrow to a compact block "Preprocessing" with four stacked short lines: "Mask + reflectance", "Band selection", "Resample to 256", "Pixel-wise SNV". Arrow to a small line spectrum block "SNV spectrum x". Branch from this original spectrum into an upper training lane and a lower extraction / mapping lane. Coordinate metadata bypasses the neural network and joins only at the final spatial mapping.

Upper lane header "A  Representation learning".
Draw an ordered left-to-right chain: "TGN / Shift" -> "Random patch mask" -> "Trainable encoder" -> "Unit latent z (16-D)" -> "Affine decoder" -> "Reconstruction".
The mask icon is 16 consecutive spectral patch tiles with 8 hidden grey tiles (spectral patches, NOT image patches).
Connect Reconstruction to a small orange box "Masked MSE". A separate thin arrow from the ORIGINAL SNV spectrum x directly enters this loss box, labelled "Target: original x". Do NOT draw target going through augmentation, and do NOT use reconstructed spectrum as target.
A short line of explanatory text within the lane: "256 bands • 16 spectral patches • loss on hidden bands".
Represent the affine decoder as one linear layer, not a Transformer.
A dashed vertical transfer arrow from the Trainable encoder to the lower Frozen encoder is labelled "Freeze trained weights".

Lower lane header "B  Full-visible extraction and mapping".
An arrow directly from original SNV spectrum (no augmentation) enters "Frozen encoder" with a row of 16 fully visible patch tiles and small label "All bands visible". Then -> "Unit embeddings (16-D)" -> "Cosine-KMeans" -> "Labels + pixel coordinates" -> "Spatial label map".
The Cosine-KMeans block must clearly contain two small substeps: "Fit centers: train embeddings" and "Assign: fixed centers".
The final map is a schematic tall wood silhouette with several categorical colored regions on white background; NO ordered degradation severity or healthy/damaged labels.
Include a thin clearly routed bypass arrow "Stored pixel coordinates" from the input/preprocessing area to "Labels + pixel coordinates"; coordinates never enter the encoder or clustering as features.

Small footer, verbatim: "CV: fit on train samples only; apply frozen encoder and centers to test samples."
A second small footer: "Schematic reference — not experimental results."

Priorities: scientific correctness of branching, target path, frozen weights, full-band extraction, center fit vs assignment, coordinate bypass. Keep arrow crossings minimal and labels legible. No standalone large title. No extra stages, no classifier, no decoder in the mapping lane, no spatial image patches, no FT-IR panel.

## 修正prompt 1

Edit this scientific workflow diagram. Preserve its overall white landscape layout, two colored lanes, scientific icons and all correct content. Make these precise scientific corrections:
1. The line labeled "Target: original x" MUST start directly at the "SNV spectrum x" box BEFORE the TGN / Shift block. Disconnect it completely from the TGN / Shift block. Route it along the top outside edge of lane A if needed, with an arrow entering Masked MSE. It represents the original unaugmented SNV target, never augmented data.
2. The dashed orange arrow "Freeze trained weights" MUST connect the Trainable encoder to the Frozen encoder, not to Unit embeddings. Route a clear elbowed dashed arrow to the top of the Frozen encoder triangle and put its arrowhead there.
3. On both spectrum plots change the vertical label from "Reflectance (SNV)" to "SNV". Change the horizontal label to "Wavelength". No a.u. wavelength units.
4. Use cluster labels starting at 1: legend "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", same four colors. In Labels + pixel coordinates table, the three example labels must be 1, 2, 3. Background is white and has label 0, outside the wood. Do not show cluster 0 as a foreground color.
5. In the lower Cosine-KMeans block clearly separate train-only center fitting from fixed-center assignment. Frozen unit embeddings feed assignment. A small separately labeled branch "Train only" feeds "Fit centers", which supplies centers downward to "Assign". Avoid making test embeddings appear to refit the centers.
6. Change "(pretrained)" under Frozen encoder to "(fixed weights)".
No new content, no large title, no new stages. All text must remain legible. The categorical wood map is schematic and not a degradation severity scale. Preserve the footers.

## 修正prompt 2（最終）

Make one small local edit only. Keep the entire diagram, layout, arrows, wording, colors, legends and footers unchanged. Replace the contents of the small table under "Labels + pixel coordinates" with this exact four-row table (one header and three data rows):
Label | row | col
1 | 12 | 34
2 | 12 | 35
3 | 13 | 34
Do not put any zero label in this table. Foreground cluster labels start at 1. Do not add any extra data row or ellipsis row. This is the only requested change.
