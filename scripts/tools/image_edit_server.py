#!/usr/bin/env python3
"""HTTP server for image editing (supports multiple backends).

Supported backends:
  - flux-klein:  FLUX.2-klein-9B (DiffusionPipeline, 4 steps, fast)
  - qwen:        Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline, 50 steps)

Run in the conda env that has diffusers (e.g. qwen_test).
Model is loaded once on startup, then serves edit requests over HTTP.

API:
  POST /edit
    Body (JSON):  {"image_b64": "<base64 PNG>", "prompt": "..."}
    Response:     {"status": "ok", "image_b64": "<base64 PNG>"}
              or: {"status": "error", "msg": "..."}

  GET /health     → {"status": "ok"}

Usage:
  conda activate qwen_test

  # FLUX.2-klein (default, fast 4-step editing, use CUDA_VISIBLE_DEVICES for GPU)
  CUDA_VISIBLE_DEVICES=2 python scripts/tools/image_edit_server.py

  # Qwen (legacy)
  python scripts/tools/image_edit_server.py --backend qwen --gpu 2

  # Then the pipeline connects via:
  #   image_edit_base_url: "http://localhost:8001"
"""

import argparse
import base64
import io
import json
import logging
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger("image_edit_server")

# Global pipeline reference (set in main)
PIPE = None
BACKEND = "flux-klein"
STEPS = 4
CFG_SCALE = 1.0


class EditHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok", "backend": BACKEND})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/edit":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception as e:
            self._json_response({"status": "error", "msg": f"bad request: {e}"}, 400)
            return

        image_b64 = body.get("image_b64", "")
        prompt = body.get("prompt", "")
        if not image_b64 or not prompt:
            self._json_response(
                {"status": "error", "msg": "image_b64 and prompt required"}, 400)
            return

        # Per-request overrides (optional)
        steps = body.get("steps", STEPS)
        cfg = body.get("cfg_scale", CFG_SCALE)

        try:
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            logger.info(f"Edit request: image={img.size}, steps={steps}, "
                        f"backend={BACKEND}, prompt={prompt!r}")

            with torch.inference_mode():
                if BACKEND == "flux-klein":
                    output = PIPE(
                        image=img,
                        prompt=prompt,
                        num_inference_steps=steps,
                        num_images_per_prompt=1,
                    )
                else:
                    # Qwen backend
                    output = PIPE(
                        image=[img],
                        prompt=prompt,
                        negative_prompt="blurry, artifacts, ghost, shadow, "
                                       "residual, double, transparent",
                        num_inference_steps=steps,
                        true_cfg_scale=cfg,
                        num_images_per_prompt=1,
                    )
            result_img = output.images[0]

            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            self._json_response({"status": "ok", "image_b64": result_b64})
            logger.info("Edit completed successfully")
        except BrokenPipeError:
            logger.warning("Client disconnected before response was sent")
        except Exception as e:
            logger.exception("Edit failed")
            try:
                self._json_response({"status": "error", "msg": str(e)}, 500)
            except BrokenPipeError:
                logger.warning("Client disconnected before error response sent")

    def _json_response(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        logger.info(fmt % args)


def main():
    parser = argparse.ArgumentParser(
        description="HTTP server for image editing (FLUX.2-klein / Qwen)")
    parser.add_argument("--backend", default="flux-klein",
                        choices=["flux-klein", "qwen"],
                        help="Model backend (default: flux-klein)")
    parser.add_argument("--model", default=None,
                        help="Model path override. Defaults: "
                             "flux-klein → /mnt/zsn/ckpts/FLUX.2-klein-9B, "
                             "qwen → /mnt/zsn/ckpts/Qwen-Image-Edit-2511")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default: 4 for flux-klein, 50 for qwen)")
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="CFG scale (default: 1.0 for flux-klein, 5.0 for qwen)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    global PIPE, BACKEND, STEPS, CFG_SCALE
    BACKEND = args.backend

    # Resolve defaults per backend (same default order as load_config ckpt_root)
    _proj = Path(__file__).resolve().parents[2]
    _mnt = Path("/mnt/zsn/ckpts")
    _default_root = (
        os.environ.get("PARTCRAFT_CKPT_ROOT", "").strip()
        or (str(_mnt) if _mnt.is_dir() else str(_proj / "checkpoints"))
    )
    _ckpt_root = _default_root
    model_defaults = {
        "flux-klein": {
            "model": os.path.join(_ckpt_root, "FLUX.2-klein-9B"),
            "steps": 4,
            "cfg_scale": 1.0,
        },
        "qwen": {
            "model": os.path.join(_ckpt_root, "Qwen-Image-Edit-2511"),
            "steps": 50,
            "cfg_scale": 5.0,
        },
    }
    defaults = model_defaults[BACKEND]
    model_path = args.model or defaults["model"]
    STEPS = args.steps if args.steps is not None else defaults["steps"]
    CFG_SCALE = args.cfg_scale if args.cfg_scale is not None else defaults["cfg_scale"]

    device = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
    logger.info(f"Backend: {BACKEND}")
    logger.info(f"Loading model from {model_path} ...")

    if BACKEND == "flux-klein":
        # FLUX.2-klein uses DiffusionPipeline with device_map="cuda".
        # To select a specific GPU, set CUDA_VISIBLE_DEVICES before launch.
        from diffusers import DiffusionPipeline
        PIPE = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        from diffusers import QwenImageEditPlusPipeline
        PIPE = QwenImageEditPlusPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16)
        PIPE.to(device)

    PIPE.set_progress_bar_config(disable=True)
    logger.info(f"Model loaded on {device}, steps={STEPS}, cfg={CFG_SCALE}")

    server = HTTPServer(("0.0.0.0", args.port), EditHandler)
    logger.info(f"Serving on http://0.0.0.0:{args.port}")
    logger.info(f"  POST /edit   — edit an image")
    logger.info(f"  GET  /health — health check")
    server.serve_forever()


if __name__ == "__main__":
    main()
