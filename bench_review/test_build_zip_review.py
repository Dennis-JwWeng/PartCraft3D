from pathlib import Path
import json
import zipfile

import build_zip_review as builder


def test_asset_name_is_stable_and_safe():
    assert builder.safe_asset_dir("add_obj/001") == "add_obj_001"


def test_write_assets_zip_uses_stored_entries(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"png")
    record = {
        "edit_id": "mat_obj_001",
        "edit_type": "material",
        "obj_id": "obj",
        "shard": "02",
        "h3d_before_png": str(img),
        "h3d_after_png": str(img),
        "two_d_input_png": str(img),
        "two_d_edited_png": str(img),
    }
    out = tmp_path / "assets.zip"
    builder.write_assets_zip([record], out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "assets/mat_obj_001/3d_before.png" in names
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest[0]["assets"]["two_d_edited"] == "assets/mat_obj_001/2d_edited.png"


def test_html_has_zip_drop_and_dynamic_exports(tmp_path):
    out = tmp_path / "h3d_review_tool.html"
    builder.write_tool_html(out)
    html = out.read_text(encoding="utf-8")
    assert "拖入 assets.zip" in html
    assert "parseStoredZip" in html
    assert "currentZipStem" in html
    assert "review_results_" in html
    assert "selected_edit_ids_" in html
    assert "review_results_000.json" not in html


def test_load_translations_by_edit_id(tmp_path):
    path = tmp_path / "translations.jsonl"
    path.write_text(
        '{"edit_id":"e1","prompt_zh":"中文一。"}\n'
        '{"edit_id":"e2","prompt_zh":"中文二。"}\n',
        encoding="utf-8",
    )
    assert builder.load_translations(path) == {"e1": "中文一。", "e2": "中文二。"}


def test_chunk_filename_can_start_from_offset(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"png")
    meta = tmp_path / "meta.json"
    meta.write_text('{"instruction":{"prompt":"remove wheel"}}', encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "edit_id": "del_obj_000",
            "edit_type": "deletion",
            "obj_id": "obj",
            "shard": "00",
            "h3d_before_png": str(img),
            "h3d_after_png": str(img),
            "h3d_meta_json": str(meta),
        }) + "\n",
        encoding="utf-8",
    )

    out = builder.build_chunk(manifest, tmp_path, chunk_index=0, chunk_size=100, chunk_offset=39)

    assert out.name == "h3d_test_review_039_assets.zip"


def test_html_exports_use_loaded_zip_stem(tmp_path):
    out = tmp_path / "h3d_review_tool.html"
    builder.write_tool_html(out)
    html = out.read_text(encoding="utf-8")

    assert "currentZipStem = file.name.replace" in html
    assert "review_results_${currentZipStem}.json" in html
    assert "selected_edit_ids_${currentZipStem}.txt" in html
