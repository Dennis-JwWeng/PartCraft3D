#!/usr/bin/env python3
"""Quick test: feed flux a [original | highlight] panel for one edit and see
what comes back. For phase 1.5 design experiments.

Usage:
    python scripts/tools/test_flux_panel_edit.py \
        --obj-id 112204b2a12c4e25bbbdcc0d196b1ad5 --shard 01 \
        --view-index 3 --part-ids 3 \
        --new-part-desc "a ceramic white sink" \
        --prompt "Replace the sink basin with a ceramic white sink." \
        --flux-url http://localhost:8011 \
        --out outputs/_debug/flux_panel_sink.png
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_part_highlight import render_highlight  # noqa: E402


PROMPT_TEMPLATES = {
    # v1: original prompt (panel in, panel out) — failed.
    "panel_out": (
        "The image is a side-by-side panel. LEFT half: the original 3D object. "
        "RIGHT half: the same view with the parts to edit painted in bright color "
        "and all other parts in light gray (the right half is a guide only). "
        "Edit the LEFT half: {action} "
        "Keep all other parts of the object pixel-identical, same camera, same "
        "lighting, photorealistic. Output a side-by-side panel in the same layout."
    ),
    # v2: single image (no highlight, no panel) — flux-klein's native form.
    # The runner must construct a strong textual spatial anchor from
    # target_part_desc / part color name.
    "single": (
        "{action} {anchor} Keep every other part of the object pixel-identical, "
        "same camera, same lighting, photorealistic."
    ),
    # v3: panel as multi-region reference, but ask for a SINGLE-image output
    # (not a panel). flux sees both halves as context, edits the left half,
    # returns one object image.
    "panel_ref": (
        "The input is a 2-panel reference. The left panel is the original "
        "object photo. The right panel shows the SAME view with the part(s) "
        "to be edited painted in a bright color while all other parts are "
        "light gray — use this as a mask telling you WHERE to edit. "
        "Produce a SINGLE photo (not a panel) of the object after the edit, "
        "with the camera and unedited regions identical to the left panel. "
        "Edit: {action}"
    ),
}


def call_flux(url: str, image_bgr: np.ndarray, prompt: str,
              steps: int = 4) -> np.ndarray:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("encode fail")
    body = json.dumps({
        "image_b64": base64.b64encode(buf.tobytes()).decode(),
        "prompt": prompt,
        "steps": steps,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/edit",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    if out.get("status") != "ok":
        raise RuntimeError(f"flux error: {out}")
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(out["image_b64"]), np.uint8),
        cv2.IMREAD_COLOR)
    return img


def stitch_panel(orig: np.ndarray, hl: np.ndarray) -> np.ndarray:
    H, W = orig.shape[:2]
    if hl.shape[:2] != (H, W):
        hl = cv2.resize(hl, (W, H), interpolation=cv2.INTER_AREA)
    sep = np.full((H, 4, 3), 255, dtype=np.uint8)
    return np.hstack([orig, sep, hl])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-id", required=True)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--mesh-root", default="data/partverse/mesh")
    ap.add_argument("--images-root", default="data/partverse/images")
    ap.add_argument("--view-index", type=int, required=True)
    ap.add_argument("--part-ids", type=int, nargs="+", required=True)
    ap.add_argument("--prompt", required=True,
                    help="Original phase1 edit.prompt")
    ap.add_argument("--new-part-desc", default="",
                    help="modification new_part_desc / material / etc.")
    ap.add_argument("--flux-url", default="http://localhost:8011")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--mode", default="panel_ref",
                    choices=list(PROMPT_TEMPLATES.keys()))
    ap.add_argument("--anchor", default="",
                    help="Spatial anchor for --mode single, e.g. "
                         "'(the basin in the center of the cabinet)'")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--blender",
        default="/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender")
    args = ap.parse_args()

    mesh = Path(args.mesh_root) / args.shard / f"{args.obj_id}.npz"
    img = Path(args.images_root) / args.shard / f"{args.obj_id}.npz"

    print(f"[1/3] rendering highlight (view {args.view_index}, parts {args.part_ids})…")
    orig, hl = render_highlight(mesh, img, args.view_index, args.part_ids,
                                args.blender)
    panel_in = stitch_panel(orig, hl)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        flux_input = orig
    else:
        flux_input = panel_in
    in_path = args.out.with_name(args.out.stem + ".input.png")
    cv2.imwrite(str(in_path), flux_input)
    print(f"      flux input ({args.mode}): {in_path}  "
          f"({flux_input.shape[1]}x{flux_input.shape[0]})")

    action = args.prompt.strip()
    if args.new_part_desc:
        action = action.rstrip(".") + f". The replacement should be {args.new_part_desc}."
    tmpl = PROMPT_TEMPLATES[args.mode]
    flux_prompt = (tmpl.format(action=action, anchor=args.anchor)
                   if "{anchor}" in tmpl else tmpl.format(action=action))
    print(f"[2/3] flux prompt:\n      {flux_prompt}")

    print(f"[3/3] calling flux at {args.flux_url} (steps={args.steps})…")
    out_img = call_flux(args.flux_url, flux_input, flux_prompt, steps=args.steps)
    print(f"      flux output: {out_img.shape[1]}x{out_img.shape[0]}")

    # Stack input panel + flux output for side-by-side review
    # review image always shows: original | highlight | flux output
    H_in = panel_in.shape[0]
    if out_img.shape[0] != H_in:
        scale = H_in / out_img.shape[0]
        out_img = cv2.resize(out_img,
                             (int(out_img.shape[1] * scale), H_in),
                             interpolation=cv2.INTER_AREA)
    sep = np.full((H_in, 8, 3), 180, dtype=np.uint8)
    review = np.hstack([panel_in, sep, out_img])
    cv2.imwrite(str(args.out), review)
    print(f"[OK] {args.out}  (left=input panel, right=flux edit)")


if __name__ == "__main__":
    main()
