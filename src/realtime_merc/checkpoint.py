"""Tensor-only checkpoints and compatibility with the original classifier heads."""

import inspect
from pathlib import Path

import torch

from .features import EMOTIONS
from .model import RealtimeMERC


MODEL_KEYS = set(inspect.signature(RealtimeMERC.__init__).parameters) - {"self"}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Expected model_config and model_state in checkpoint.")
    if checkpoint.get("label_names", EMOTIONS) != EMOTIONS:
        raise ValueError("Checkpoint label order differs from this implementation.")
    model_config = checkpoint["model_config"]
    if model_config.get("num_classes", 7) != len(EMOTIONS):
        raise ValueError("Checkpoint must use the seven MELD labels.")
    model = RealtimeMERC(**{k: v for k, v in model_config.items() if k in MODEL_KEYS})
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device), checkpoint


def save_checkpoint(path, model, model_config, data_config, epoch, best_f1,
                    optimizer=None, seed=42):
    payload = {"format_version": 1, "model_config": model_config,
               "model_state": model.state_dict(), "label_names": EMOTIONS,
               "data_config": data_config, "epoch": epoch,
               "best_f1": best_f1, "seed": seed}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
        payload["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
