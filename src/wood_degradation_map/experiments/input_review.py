"""Prepare a manual-review index from existing preprocessing tables and images."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .input_validation import InputInventory, _check_ids, _check_integer_column
from .manifests import _digest, _read_json, _write_json

TABLES = ("manifest", "sample_quality", "mask_quality", "reference_band_quality", "output_band_summary")
DOCUMENTS = ("config", "cutoff_decision", "preprocessing_summary")
OVERVIEW_IMAGES = (
    ("cutoff_decision.png", "cutoff境界と参照SNR・分布"),
    ("interpolated_reflectance_band_distribution.png", "補間後の反射率分布"),
    ("interpolated_snv_band_distribution.png", "最終SNVの分布"),
    ("final_snv_anomaly_candidates.png", "SNV二次差分の上位候補（発生率ではない）"),
)


def _relative_url(target: Path, output: Path) -> str:
    try:
        return quote(Path(os.path.relpath(target, output)).as_posix(), safe="/")
    except ValueError:  # Windows paths on different drives cannot be relative.
        return target.resolve().as_uri()


def _figure(path: Path, caption: str, output: Path, missing: list[str]) -> str:
    caption = html.escape(caption)
    if not path.is_file():
        missing.append(str(path))
        return f'<figure><p class="missing">画像がありません</p><figcaption>{caption}</figcaption></figure>'
    url = html.escape(_relative_url(path, output), quote=True)
    return (f'<figure><a href="{url}"><img src="{url}" loading="lazy" alt="{caption}"></a>'
            f'<figcaption>{caption} — クリックで原寸表示</figcaption></figure>')


def prepare_input_review(
    processed_dir: Path, figures_dir: Path, output_dir: Path, inventory: InputInventory,
) -> Path:
    """Export small stored tables and link every adopted sample's existing image.

    The caller validates inventory using load_input_inventory. No HDF5 data,
    spectra, image pixels, raw data, or models are opened here. No thresholds,
    classifications or representative sample choices are introduced. HTML is a
    navigation aid; successful generation is NOT a scientific quality approval.
    """
    processed, figures, output = processed_dir.resolve(), figures_dir.resolve(), output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Review output already exists: {output}")
    if output.is_relative_to(processed) or processed.is_relative_to(output):
        raise ValueError("Review output must be separate from immutable processed data")
    tables = {name: pd.read_parquet(processed / f"{name}.parquet") for name in TABLES}
    documents = {name: _read_json(processed / f"{name}.json") for name in DOCUMENTS}
    report_config = _read_json(figures / "report_config.json")
    ids = sorted(sample.sample_id for sample in inventory.samples)
    if not ids or len(ids) != len(set(ids)) or processed.name != inventory.preprocessing_id:
        raise ValueError("Invalid review inventory/preprocessing identity")
    counts = {sample.sample_id: sample.saved_pixel_count for sample in inventory.samples}
    for name in ("manifest", "sample_quality", "mask_quality"):
        table = tables[name]
        _check_ids(table, name)
        if set(table["sample_id"]) != set(ids):
            raise ValueError(f"{name}: review sample coverage differs from inventory")
        tables[name] = table.sort_values("sample_id").reset_index(drop=True)
    for name in ("manifest", "sample_quality"):
        table = tables[name]
        _check_integer_column(table, "saved_pixel_count", 1)
        if any(row.saved_pixel_count != counts[row.sample_id] for row in table.itertuples()):
            raise ValueError(f"{name}: review pixel count differs from inventory")
    summary = documents["preprocessing_summary"]
    if (summary.get("preprocessing_id") != inventory.preprocessing_id
            or summary.get("sample_count") != len(ids)
            or summary.get("saved_pixel_count") != sum(counts.values())
            or summary.get("cutoff_decision") != documents["cutoff_decision"]):
        raise ValueError("Preprocessing summary differs from inventory/cutoff record")
    if (report_config.get("preprocessing_id") != inventory.preprocessing_id
            or Path(report_config["processed_data_dir"]).resolve() != processed):
        raise ValueError("Existing figures refer to a different preprocessing source")

    missing: list[str] = []
    overview = "".join(_figure(figures / filename, caption, output, missing)
                       for filename, caption in OVERVIEW_IMAGES)
    cards = "".join(
        _figure(figures / "reflectance_l2_norm" / f"{sample}.png",
                f"{sample} · 保存画素 {counts[sample]:,}", output, missing) for sample in ids)
    sections = []
    for name, table in tables.items():
        sections.append(
            f'<details open><summary>{name} · {len(table)}行</summary>'
            f'<p><a href="{name}.csv">全列CSV</a></p><div class="table">'
            + table.to_html(index=False, escape=True, na_rep="NA（保存表の欠損）", border=0,
                            float_format=lambda value: format(value, ".17g"))
            + '</div></details>')
    for name, document in {**documents, "report_config": report_config}.items():
        sections.append(f'<details><summary>{name}</summary><pre>'
                        + html.escape(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
                        + '</pre></details>')
    scale = html.escape(json.dumps(report_config.get("reflectance_l2_norm", {}), ensure_ascii=False))
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>前処理入力の確認資料</title><style>
body{font-family:system-ui,sans-serif;color:#203040;background:#f6f8fa;margin:2rem auto;padding:0 1.5rem;max-width:1400px}
nav{display:flex;gap:1.5rem;flex-wrap:wrap}a{color:#075985}p{line-height:1.8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem}
figure,details{background:white;border:1px solid #ccd6df;border-radius:8px;padding:1rem;margin:1rem 0}
figure img{width:100%;height:330px;object-fit:contain}.overview img{height:auto;max-height:700px}
figcaption{line-height:1.6;margin-top:.7rem}.table{overflow:auto;max-height:520px}
th,td{padding:.5rem;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}
th{position:sticky;top:0;background:#edf2f6}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.missing{color:#9f1239}summary{cursor:pointer;font-weight:600}
</style><main><h1>前処理入力の確認資料</h1>
<p>既存の品質表と診断画像を照合するための資料です。生成成功は品質の承認を意味しません。
画像をクリックすると元ファイルを表示します。追加の前処理・画素除外・試料選択は行っていません。</p>
<nav><a href="#tables">品質表・保存記録</a><a href="#overview">分布・cutoff・候補図</a><a href="#samples">全試料の空間図</a></nav>
'''
    page += (f'<p>対象 {len(ids)}試料 · 保存画素 {sum(counts.values()):,} · 欠けている画像 {len(missing)}件</p>'
             '<p>確認点：品質表の除外数・範囲外反射率・SNV指標と数値誤差、cutoff周辺の分布、'
             '空間図の境界・欠け・孤立部分・強度分布を照合してください。'
             'SNV候補図は順位図であり、異常の発生率や追加除外の根拠にはしません。'
             '代表試料と採否の判断は別途記録します。</p>'
             '<h2 id="tables">品質表・保存記録</h2>' + "".join(sections)
             + '<h2 id="overview">分布・cutoff・候補図</h2><div class="overview">' + overview + '</div>'
             + '<h2 id="samples">全試料の空間図</h2><p>SNV前の補間反射率L2 norm。色設定は既存画像の記録をそのまま表示しています。</p>'
             + f'<pre>{scale}</pre><div class="grid">{cards}</div></main></html>')
    source_hashes = {f"{name}.parquet": _digest(processed / f"{name}.parquet") for name in TABLES}
    source_hashes.update({f"{name}.json": _digest(processed / f"{name}.json") for name in DOCUMENTS})
    output.mkdir(parents=True)
    for name, table in tables.items():
        with (output / f"{name}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            table.to_csv(handle, index=False, na_rep="NA")
    with (output / "index.html").open("x", encoding="utf-8") as handle:
        handle.write(page)
    _write_json(output / "review.json", {
        "status": "input_review_prepared", "manual_review_required": True,
        "preprocessing_id": inventory.preprocessing_id, "sample_ids": ids,
        "sample_count": len(ids), "saved_pixel_count": sum(counts.values()),
        "metadata_only_ids": list(inventory.metadata_only_ids), "missing_images": missing,
        "processed_dir": str(processed), "figures_dir": str(figures),
        "source_table_and_document_sha256": source_hashes,
        "report_config_sha256": _digest(figures / "report_config.json"),
        "html_sha256": _digest(output / "index.html"),
        "scope": "stored small tables and links to existing images; no HDF5 reads or scientific approval",
        "image_provenance": "linked images are not copied or hashed; their correspondence is not independently revalidated",
    })
    return output / "index.html"
