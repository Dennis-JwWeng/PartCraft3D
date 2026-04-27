import build_merged_test_review_html as builder


def test_prompt_translation_covers_common_templates():
    assert builder.prompt_to_zh("Add the metal handle.").startswith("添加")
    assert "移除" in builder.prompt_to_zh("Remove the red wheel")
    assert "替换" in builder.prompt_to_zh("Replace the chair back with a cushion")


def test_render_card_uses_flux_four_images_and_add_two_images():
    flux = {
        "edit_id": "mat_obj_001",
        "edit_type": "material",
        "obj_id": "obj",
        "shard": "02",
        "prompt_en": "Change the material to metal",
        "prompt_zh": "将材质改为金属",
        "h3d_before_data": "data:image/png;base64,before",
        "h3d_after_data": "data:image/png;base64,after",
        "two_d_input_data": "data:image/png;base64,input",
        "two_d_edited_data": "data:image/png;base64,edited",
    }
    html = builder.render_card(flux, 1)
    assert html.count("data:image/png;base64,") == 4
    assert "2D edited" in html
    assert "将材质改为金属" in html

    add = dict(flux)
    add["edit_id"] = "add_obj_001"
    add["edit_type"] = "addition"
    add["two_d_input_data"] = ""
    add["two_d_edited_data"] = ""
    html = builder.render_card(add, 1)
    assert html.count("data:image/png;base64,") == 2
    assert "2D edited" not in html
