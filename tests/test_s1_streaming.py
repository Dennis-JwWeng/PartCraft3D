"""Unit tests for s1_phase1_vlm.run_many_streaming post_object_fn hook.

No Blender, no GPU, no VLM network calls — everything mocked.
"""
from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partcraft.pipeline_v2.paths import PipelineRoot
from partcraft.pipeline_v2.s1_phase1_vlm import Phase1Result


def _make_ctx(tmp: Path, obj_id: str):
    root = PipelineRoot(tmp / "pipeline_out")
    mesh = tmp / f"{obj_id}_mesh.npz"
    img = tmp / f"{obj_id}_img.npz"
    mesh.write_bytes(b"fake")
    img.write_bytes(b"fake")
    ctx = root.context("00", obj_id, mesh_npz=mesh, image_npz=img)
    (ctx.dir / "phase1").mkdir(parents=True, exist_ok=True)
    return ctx


def _run(coro):
    return asyncio.run(coro)


FAKE_PRE = (b"png", "user_msg", [0, 1], {"deletion": 1, "modification": 1,
            "scale": 1, "material": 1, "global": 1}, "menu")


def _stream_patches():
    """ProcessPoolExecutor cannot run mocked callables (pickle); use threads."""
    return patch(
        "partcraft.pipeline_v2.s1_phase1_vlm.ProcessPoolExecutor",
        ThreadPoolExecutor,
    )


class TestPostObjectFn:
    def test_hook_called_for_each_successful_object(self, tmp_path):
        """post_object_fn is called once per object after _call_one succeeds."""
        from partcraft.pipeline_v2.s1_phase1_vlm import run_many_streaming

        ctxs = [_make_ctx(tmp_path, f"obj_{i:03d}") for i in range(3)]

        async def _fake_call_one(client, ctx, png, user_msg, pids, quota,
                                  model, sem, *, part_menu=""):
            ctx.parsed_path.write_text(
                json.dumps({"parsed": {"edits": [], "object": {"parts": []}}})
            )
            from partcraft.pipeline_v2.status import update_step, STATUS_OK
            update_step(ctx, "s1_phase1", status=STATUS_OK, n_edits=0, resumed=False)
            return Phase1Result(ctx.obj_id, ok=True)

        hook_calls: list[tuple[str, str]] = []

        async def _hook(ctx, vlm_url: str):
            hook_calls.append((ctx.obj_id, vlm_url))

        vlm_urls = ["http://fake:8002/v1", "http://fake:8012/v1"]

        with _stream_patches(), \
             patch("partcraft.pipeline_v2.s1_phase1_vlm._prerender_worker",
                   return_value=FAKE_PRE), \
             patch("partcraft.pipeline_v2.s1_phase1_vlm._call_one",
                   side_effect=_fake_call_one):
            _run(run_many_streaming(
                ctxs, blender="/fake/blender",
                vlm_urls=vlm_urls, vlm_model="fake-model",
                post_object_fn=_hook, force=True,
            ))

        assert len(hook_calls) == 3
        obj_ids = {c[0] for c in hook_calls}
        assert obj_ids == {c.obj_id for c in ctxs}
        for _, url in hook_calls:
            assert url in vlm_urls

    def test_hook_not_called_for_too_many_parts(self, tmp_path):
        """post_object_fn is NOT called when s1 skips due to too_many_parts."""
        from partcraft.pipeline_v2.s1_phase1_vlm import run_many_streaming

        ctxs = [_make_ctx(tmp_path, "obj_big")]

        hook_calls: list = []

        async def _hook(ctx, vlm_url):
            hook_calls.append(ctx.obj_id)

        with _stream_patches(), \
             patch("partcraft.pipeline_v2.s1_phase1_vlm._prerender_worker",
                   return_value=None):
            _run(run_many_streaming(
                ctxs, blender="/fake/blender",
                vlm_urls=["http://fake:8002/v1"], vlm_model="fake-model",
                post_object_fn=_hook, force=True,
            ))

        assert hook_calls == []

    def test_no_hook_does_not_break(self, tmp_path):
        """run_many_streaming without post_object_fn works as before."""
        from partcraft.pipeline_v2.s1_phase1_vlm import run_many_streaming

        ctxs = [_make_ctx(tmp_path, "obj_000")]

        async def _fake_call_one(client, ctx, png, user_msg, pids, quota,
                                  model, sem, *, part_menu=""):
            ctx.parsed_path.write_text(
                json.dumps({"parsed": {"edits": [], "object": {"parts": []}}})
            )
            from partcraft.pipeline_v2.status import update_step, STATUS_OK
            update_step(ctx, "s1_phase1", status=STATUS_OK, n_edits=0, resumed=False)
            return Phase1Result(ctx.obj_id, ok=True)

        with _stream_patches(), \
             patch("partcraft.pipeline_v2.s1_phase1_vlm._prerender_worker",
                   return_value=FAKE_PRE), \
             patch("partcraft.pipeline_v2.s1_phase1_vlm._call_one",
                   side_effect=_fake_call_one):
            results = _run(run_many_streaming(
                ctxs, blender="/fake/blender",
                vlm_urls=["http://fake:8002/v1"], vlm_model="fake-model",
                force=True,
            ))

        assert len(results) == 1
        assert results[0].ok is True
