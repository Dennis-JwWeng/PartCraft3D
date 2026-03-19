#!/usr/bin/env python3
"""HTTP server for Qwen-Image-Edit-2511 (diffusers pipeline).

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
  python scripts/tools/image_edit_server.py --gpu 2

  # Then the pipeline connects via:
  #   image_edit_base_url: "http://localhost:8001"
"""

import argparse
import base64
import io
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from PIL import Image

logger = logging.getLogger("image_edit_server")

# Global pipeline reference (set in main)
PIPE = None
STEPS = 40
CFG_SCALE = 4.0


class EditHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok"})
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

        try:
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            logger.info(f"Edit request: image={img.size}, prompt={prompt[:80]!r}")

            with torch.inference_mode():
                output = PIPE(
                    image=[img],
                    prompt=prompt,
                    negative_prompt=" ",
                    num_inference_steps=STEPS,
                    true_cfg_scale=CFG_SCALE,
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
        description="HTTP server for Qwen-Image-Edit-2511")
    parser.add_argument("--model", default="/Node11_nvme/wjw/checkpoints/Qwen-Image-Edit-2511")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    global PIPE, STEPS, CFG_SCALE
    STEPS = args.steps
    CFG_SCALE = args.cfg_scale

    from diffusers import QwenImageEditPlusPipeline

    device = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
    logger.info(f"Loading model from {args.model} ...")
    PIPE = QwenImageEditPlusPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    PIPE.to(device)
    PIPE.set_progress_bar_config(disable=True)
    logger.info(f"Model loaded on {device}")

    server = HTTPServer(("0.0.0.0", args.port), EditHandler)
    logger.info(f"Serving on http://0.0.0.0:{args.port}")
    logger.info(f"  POST /edit   — edit an image")
    logger.info(f"  GET  /health — health check")
    server.serve_forever()


if __name__ == "__main__":
    main()
