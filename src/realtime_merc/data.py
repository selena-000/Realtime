"""Ordered MELD examples; oracle trajectories require an explicit opt-in."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (
    EMOTIONS, EMOTION_TO_IDX, MMAlignFeatureStore, _normalize_emotion, _pick_column,
)


def check_protocol(mode, trajectory_source):
    if mode not in {"offline", "realtime"}:
        raise ValueError("mode must be offline or realtime.")
    if trajectory_source not in {"predicted", "none", "oracle"}:
        raise ValueError("trajectory_source must be predicted, none or oracle.")
    if mode == "offline" and trajectory_source != "oracle":
        raise ValueError("offline is only supported for explicit legacy oracle reproduction.")


def resolve_features_dir(base_dir, split):
    base = Path(base_dir)
    aliases = {"dev": ("dev", "val"), "val": ("val", "dev")}.get(split, (split,))
    return str(next((base / name for name in aliases if (base / name).is_dir()), base))


class MELDFeatureDataset(Dataset):
    def __init__(self, csv_path, features_dir, mode="realtime", skip_missing=False,
                 feature_set="mm_align", split=None, mm_align_pool="mean",
                 trajectory_source="predicted", require_labels=True):
        check_protocol(mode, trajectory_source)
        if feature_set not in {"mm_align", "emotion_llama", "conv_emotion"}:
            raise ValueError("Unsupported feature_set: {}".format(feature_set))
        if trajectory_source == "oracle" and not require_labels:
            raise ValueError("Oracle reproduction requires labeled data.")
        self.mode, self.trajectory_source = mode, trajectory_source
        self.feature_set = feature_set
        self.feature_store = (
            MMAlignFeatureStore(str(features_dir), split=split, pool=mm_align_pool)
            if feature_set == "mm_align" else None
        )
        frame = pd.read_csv(csv_path)
        dia = _pick_column(frame.columns, ["Dialogue_ID", "dialogue_id"])
        utt = _pick_column(frame.columns, ["Utterance_ID", "utterance_id"])
        emotion_col = _pick_column(frame.columns, ["Emotion", "emotion"]) if require_labels else None
        frame[dia], frame[utt] = frame[dia].astype(int), frame[utt].astype(int)
        if frame.duplicated([dia, utt]).any():
            raise ValueError("Duplicate dialogue/utterance IDs in metadata.")
        frame = frame.sort_values([dia, utt], kind="mergesort")
        self.items, self.missing_count = [], 0
        self.original_count = len(frame)
        for dialogue_id, group in frame.groupby(dia, sort=False):
            labels = []
            if require_labels:
                for value in group[emotion_col]:
                    name = _normalize_emotion(value)
                    if name not in EMOTION_TO_IDX:
                        raise ValueError("Unknown emotion: {}".format(value))
                    labels.append(EMOTION_TO_IDX[name])
            for index, (_, row) in enumerate(group.iterrows()):
                dialogue_id, utterance_id = int(dialogue_id), int(row[utt])
                path = Path(features_dir) / "dia{}_utt{}.pt".format(dialogue_id, utterance_id)
                exists = (self.feature_store.has(dialogue_id, utterance_id)
                          if self.feature_store is not None else path.is_file())
                if not exists:
                    if not skip_missing:
                        raise FileNotFoundError("Missing features for dialogue {}, utterance {}".format(
                            dialogue_id, utterance_id))
                    self.missing_count += 1
                    continue
                history = []
                if trajectory_source == "oracle":
                    # Preserve the old experiment exactly, including its target-label access.
                    history = list(labels if mode == "offline" else labels[:index])
                self.items.append({
                    "dialogue_id": dialogue_id, "utterance_id": utterance_id,
                    "emotion": labels[index] if require_labels else -1,
                    "trajectory_emotions": history, "feature_path": path,
                })
        if not self.items:
            raise ValueError("No samples remain after matching metadata to features.")
        self.feature_dim = self._load_features(self.items[0]).numel()

    def _load_features(self, item):
        if self.feature_store is not None:
            value = self.feature_store.get(item["dialogue_id"], item["utterance_id"])
        else:
            value = torch.load(item["feature_path"], map_location="cpu", weights_only=True)
            if isinstance(value, dict):
                if "fused_features" not in value:
                    raise KeyError("Feature dict must contain fused_features.")
                value = value["fused_features"]
        value = torch.as_tensor(value).float().reshape(-1)
        if not value.numel() or not torch.isfinite(value).all():
            raise ValueError("Feature tensor must be nonempty and finite.")
        return value

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        features = self._load_features(item)
        if features.numel() != self.feature_dim:
            raise ValueError("Inconsistent feature dimensions at dialogue {}, utterance {}".format(
                item["dialogue_id"], item["utterance_id"]))
        return {key: value for key, value in item.items() if key != "feature_path"} | {
            "fused_features": features,
        }


def pack_trajectories(sequences, device=None):
    length = max((len(seq) for seq in sequences), default=0)
    values = torch.full((len(sequences), length), -1, dtype=torch.long, device=device)
    mask = torch.zeros_like(values, dtype=torch.bool)
    for index, seq in enumerate(sequences):
        if seq:
            values[index, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[index, :len(seq)] = True
    return values, mask


def collate_batch(batch):
    trajectory, mask = pack_trajectories([item["trajectory_emotions"] for item in batch])
    return {
        "fused_features": torch.stack([item["fused_features"] for item in batch]),
        "emotion": torch.tensor([item["emotion"] for item in batch], dtype=torch.long),
        "dialogue_id": [item["dialogue_id"] for item in batch],
        "utterance_id": [item["utterance_id"] for item in batch],
        "trajectory_emotions": trajectory, "trajectory_mask": mask,
    }
