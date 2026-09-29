from pathlib import Path
import torch
from omegaconf import OmegaConf
from futurecad.grounding_threshold import checkpoint_threshold
from grounding_brep.models.grounding import GroundingBRep
from grounding_brep.dataset import (
    GroundingBRepDataset,
)


def load_config(path, overrides=()):
    return OmegaConf.merge(
        OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides))
    )


def load_model(cfg, checkpoint=None, device="cpu"):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if checkpoint:
        cfg.model.text.initialize_pretrained = False
    model = GroundingBRep(cfg)
    if checkpoint:
        if Path(checkpoint).suffix == ".safetensors":
            from safetensors.torch import load_file
            from safetensors import safe_open

            with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
                embedded_threshold = (archive.metadata() or {}).get(
                    "inference_threshold"
                )
            payload = load_file(str(checkpoint), device="cpu")
        else:

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            embedded_threshold = payload.get("inference_threshold")
            if embedded_threshold is None:
                embedded_threshold = payload.get("threshold_calibration", {}).get(
                    "threshold"
                )
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
        model.checkpoint_inference_threshold = checkpoint_threshold(
            embedded_threshold
        )
    return model.to(device)


def dataset(cfg, split):
    return GroundingBRepDataset(
        cfg.data.dataset_path,
        split=split,
        limit=cfg.data.get("limit"),
        max_len_uv=cfg.data.get("max_len_uv"),
        split_path=cfg.data.get("split_path"),
    )
