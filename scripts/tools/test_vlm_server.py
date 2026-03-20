#!/usr/bin/env python3
"""Test VLM server connectivity and image support.

Usage:
    python scripts/tools/test_vlm_server.py [--port 8002]
"""
import argparse
import base64
import sys
import json
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from io import BytesIO

def test_health(base_url):
    """Test 1: Server reachable?"""
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        model_id = data["data"][0]["id"] if data.get("data") else "unknown"
        print(f"[PASS] Server reachable, model: {model_id}")
        return model_id
    except Exception as e:
        print(f"[FAIL] Server not reachable: {e}")
        return None

def test_text_only(base_url, model):
    """Test 2: Text-only request works?"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
    }
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        reply = data["choices"][0]["message"]["content"]
        print(f"[PASS] Text request OK, reply: {reply[:50]}")
        return True
    except Exception as e:
        print(f"[FAIL] Text request failed: {e}")
        return False

def test_image_request(base_url, model, image_path=None):
    """Test 3: Image + text request works?"""
    # Create a small test image or load from dataset
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        print(f"  Using image: {image_path}")
    else:
        # Create a tiny 64x64 red PNG
        from PIL import Image
        img = Image.new("RGB", (64, 64), (255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        print("  Using generated test image (64x64 red)")

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }],
        "max_tokens": 50,
    }
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        reply = data["choices"][0]["message"]["content"]
        print(f"[PASS] Image request OK, reply: {reply[:80]}")
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"[FAIL] Image request HTTP {e.code}: {body[:200]}")
        return False
    except Exception as e:
        print(f"[FAIL] Image request failed: {e}")
        return False

def test_gpu_memory():
    """Test 4: Check GPU memory."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
             "--format=csv,noheader"], text=True)
        print("  GPU memory:")
        for line in out.strip().split("\n"):
            print(f"    {line.strip()}")
    except Exception:
        print("  nvidia-smi not available")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--image", type=str, default=None,
                        help="Path to test image (optional)")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    print(f"Testing VLM server at {base_url}\n")

    print("=" * 50)
    print("Test 1: Server health")
    model = test_health(base_url)
    if not model:
        print("\nServer not running. Start it first.")
        sys.exit(1)

    print("\nTest 2: Text-only request")
    test_text_only(base_url, model)

    print("\nTest 3: Image + text request")
    test_image_request(base_url, model, args.image)

    print("\nTest 4: GPU memory")
    test_gpu_memory()

    print("\n" + "=" * 50)
    print("Done. Check [PASS]/[FAIL] above.")

if __name__ == "__main__":
    main()
