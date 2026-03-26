from typing import *
import os
import torch
import torch.nn as nn
from .. import models


def _resolve_model_entry(pipeline_dir: str, v: str) -> str:
    """Map pipeline.json ``models`` entry to a local path or HF hub id.

    Text checkpoints reference shared weights as ``org/TRELLIS-image-large/ckpts/...``.
    Those files live under the sibling ``TRELLIS-image-large`` folder next to
    ``TRELLIS-text-xlarge``, not under ``TRELLIS-text-xlarge/JeffreyXiang/...``.
    """
    pipeline_dir = os.path.abspath(pipeline_dir)
    joined = os.path.join(pipeline_dir, v)

    def _weights_ok(p: str) -> bool:
        return os.path.isfile(f"{p}.json") and os.path.isfile(f"{p}.safetensors")

    if _weights_ok(joined):
        return joined
    for marker in ("JeffreyXiang/TRELLIS-image-large/", "TRELLIS-image-large/"):
        if marker in v.replace("\\", "/"):
            suffix = v.split(marker, 1)[1].lstrip("/")
            img_root = os.path.join(os.path.dirname(pipeline_dir), "TRELLIS-image-large")
            alt = os.path.join(img_root, suffix)
            if _weights_ok(alt):
                return alt
    vn = v.replace("\\", "/")
    # HuggingFace-style org/repo/subpath (not relative ckpts/ under pipeline)
    if vn.count("/") >= 2 and not vn.startswith("ckpts/"):
        parts = [p for p in vn.split("/") if p]
        if len(parts) >= 3:
            return v
    return joined


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            load_id = _resolve_model_entry(path, v)
            _models[k] = models.from_pretrained(load_id)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
