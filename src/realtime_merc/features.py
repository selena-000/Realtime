import glob
import os
import pickle
import re

import numpy as np
import torch


EMOTIONS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

MM_ALIGN_MODALITY_KEYS = {
    "vision": [
        "vision",
        "visual",
        "video",
        "image",
        "resnet",
        "vision_feat",
        "visual_feat",
        "video_features",
        "video_feats",
        "v",
    ],
    "audio": [
        "audio",
        "wav2vec",
        "wav2vec2",
        "audio_feat",
        "audio_features",
        "audio_feats",
        "a",
    ],
    "text": [
        "text",
        "language",
        "bert",
        "roberta",
        "text_feat",
        "token_ids",
        "tokens",
        "t",
    ],
}
MM_ALIGN_MODALITY_ORDER = ["vision", "audio", "text"]
MM_ALIGN_SPLIT_ALIASES = {
    "train": ["train"],
    "dev": ["dev", "valid", "val", "validation"],
    "val": ["val", "valid", "validation", "dev"],
    "test": ["test"],
}

_MM_ALIGN_PAYLOAD_CACHE = {}
_MM_ALIGN_FEATURE_CACHE = {}


def _pick_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
        lower = candidate.lower()
        for col in columns:
            if col.lower() == lower:
                return col
    raise ValueError("Missing required column. Tried: {}".format(", ".join(candidates)))


def _normalize_emotion(emotion):
    if emotion is None:
        return ""
    text = str(emotion).strip().lower()
    mapping = {
        "happiness": "joy",
        "happy": "joy",
    }
    return mapping.get(text, text)


def _find_key(data, candidates):
    if not isinstance(data, dict):
        return None
    lower_map = {str(k).lower(): k for k in data.keys()}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            return key
    return None


def _as_list(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def _to_1d_vector(value, pool="mean"):
    tensor = torch.as_tensor(value).float()
    if tensor.dim() == 0:
        return tensor.view(1)
    if tensor.dim() == 1:
        return tensor
    if pool == "flatten":
        return tensor.reshape(-1)
    if pool == "mean":
        flat = tensor.reshape(tensor.size(0), -1)
        return flat.mean(dim=0)
    raise ValueError("Unknown mm_align pooling mode: {}".format(pool))


def _extract_modality_dict(sample):
    if isinstance(sample, dict):
        modalities = {}
        for modality, keys in MM_ALIGN_MODALITY_KEYS.items():
            key = _find_key(sample, keys)
            if key is not None:
                modalities[modality] = sample[key]
        return modalities
    if isinstance(sample, (list, tuple)) and len(sample) == 3:
        return {"vision": sample[0], "audio": sample[1], "text": sample[2]}
    return {}


def _concat_modalities(modalities, pool="mean"):
    parts = []
    for name in MM_ALIGN_MODALITY_ORDER:
        if name in modalities:
            parts.append(_to_1d_vector(modalities[name], pool))
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def _resolve_mm_align_embedding_path(features_dir):
    candidates = [
        "embedding.p",
        "embedding.pkl",
        "embeddings.p",
        "embeddings.pkl",
        "embedding.npy",
        "embeddings.npy",
    ]
    for name in candidates:
        path = os.path.join(features_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _extract_embedding_array(payload):
    if isinstance(payload, dict):
        for key in [
            "embedding",
            "embeddings",
            "word_embedding",
            "word_embeddings",
            "vectors",
            "vecs",
        ]:
            if key in payload:
                payload = payload[key]
                break
    return torch.as_tensor(payload).float()


def _load_mm_align_text_embedding(features_dir):
    path = _resolve_mm_align_embedding_path(features_dir)
    if path is None:
        return None, None

    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in {".npy", ".npz"}:
        payload = np.load(path, allow_pickle=True)
    else:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

    embedding = _extract_embedding_array(payload)
    if embedding.dim() != 2:
        raise ValueError(
            "MM-Align embedding at {} must be 2D, got shape {}".format(
                path, tuple(embedding.shape)
            )
        )
    return embedding, path


def _maybe_embed_text(text_value, text_embedding):
    if text_embedding is None:
        return text_value
    try:
        ids = torch.as_tensor(text_value)
    except Exception:
        return text_value
    if ids.dim() == 1:
        ids = ids.long().clamp(min=0, max=text_embedding.size(0) - 1)
        return text_embedding[ids]
    return text_value


def _parse_id_pair(value):
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
    return None


def _extract_ids_from_sample(sample, fallback_key=None):
    dialogue_id = None
    utterance_id = None
    if isinstance(sample, dict):
        dialogue_key = _find_key(
            sample, ["dialogue_id", "dialogue", "dia_id", "dia", "dialogueid"]
        )
        utterance_key = _find_key(
            sample, ["utterance_id", "utterance", "utt_id", "utt", "utteranceid"]
        )
        if dialogue_key is not None and utterance_key is not None:
            dialogue_id = int(sample[dialogue_key])
            utterance_id = int(sample[utterance_key])
        else:
            id_key = _find_key(sample, ["id", "uid", "sample_id", "index"])
            if id_key is not None:
                parsed = _parse_id_pair(sample[id_key])
                if parsed is not None:
                    dialogue_id, utterance_id = parsed

    if (dialogue_id is None or utterance_id is None) and fallback_key is not None:
        parsed = _parse_id_pair(fallback_key)
        if parsed is not None:
            dialogue_id, utterance_id = parsed
    return dialogue_id, utterance_id


def _extract_id_arrays(data):
    if not isinstance(data, dict):
        return None, None
    dialogue_key = _find_key(
        data, ["dialogue_id", "dialogue", "dia_id", "dia", "dialogueid"]
    )
    utterance_key = _find_key(
        data, ["utterance_id", "utterance", "utt_id", "utt", "utteranceid"]
    )
    if dialogue_key is not None and utterance_key is not None:
        return _as_list(data[dialogue_key]), _as_list(data[utterance_key])

    id_key = _find_key(data, ["id", "ids", "uid", "sample_id", "index"])
    if id_key is not None:
        ids = _as_list(data[id_key])
        dialogue_ids = []
        utterance_ids = []
        for value in ids:
            parsed = _parse_id_pair(value)
            if parsed is None:
                return None, None
            dialogue_ids.append(parsed[0])
            utterance_ids.append(parsed[1])
        return dialogue_ids, utterance_ids
    return None, None


def _resolve_mm_align_split(payload, split):
    if split is None:
        if isinstance(payload, dict):
            key = _find_key(payload, ["train"])
            if key is not None:
                return payload[key]
        return payload
    if isinstance(payload, dict):
        aliases = MM_ALIGN_SPLIT_ALIASES.get(split.lower(), [split])
        for alias in aliases:
            key = _find_key(payload, [alias])
            if key is not None:
                return payload[key]
        return payload
    if isinstance(payload, (list, tuple)) and len(payload) == 3:
        split_lower = split.lower()
        if split_lower == "train":
            return payload[0]
        if split_lower in {"dev", "val", "valid", "validation"}:
            return payload[1]
        if split_lower == "test":
            return payload[2]
    return payload


def _resolve_mm_align_pkl_path(features_dir, split=None):
    if os.path.isfile(features_dir) and features_dir.lower().endswith(".pkl"):
        return features_dir

    candidates = glob.glob(os.path.join(features_dir, "*.pkl"))
    if not candidates:
        for root, _, files in os.walk(features_dir):
            for name in files:
                if name.lower().endswith(".pkl"):
                    candidates.append(os.path.join(root, name))

    if not candidates:
        raise FileNotFoundError(
            "No .pkl files found under {} for mm_align features.".format(features_dir)
        )

    candidates = [p for p in candidates if not os.path.basename(p).lower().startswith("embedding")]
    if split is not None:
        aliases = MM_ALIGN_SPLIT_ALIASES.get(split.lower(), [split.lower()])
        matched = [
            path
            for path in candidates
            if any(alias in os.path.basename(path).lower() for alias in aliases)
        ]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            raise ValueError("Ambiguous MM-Align split files: {}. Pass an explicit .pkl path.".format(sorted(matched)))

    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Ambiguous MM-Align files: {}. Pass an explicit .pkl path.".format(sorted(candidates)))


def _load_mm_align_payload(pkl_path):
    payload = _MM_ALIGN_PAYLOAD_CACHE.get(pkl_path)
    if payload is None:
        with open(pkl_path, "rb") as handle:
            payload = pickle.load(handle)
        _MM_ALIGN_PAYLOAD_CACHE[pkl_path] = payload
    return payload


def _build_mm_align_feature_map(payload, pool="mean", text_embedding=None):
    feature_map = {}
    example_modalities = None

    def add_feature(key, value):
        if key in feature_map:
            raise ValueError("Duplicate MM-Align sample ID: {}".format(key))
        feature_map[key] = value

    if isinstance(payload, dict):
        dialogue_ids, utterance_ids = _extract_id_arrays(payload)
        modality_arrays = _extract_modality_dict(payload)
        if dialogue_ids is not None and utterance_ids is not None and modality_arrays:
            count = min(len(dialogue_ids), len(utterance_ids))
            for idx in range(count):
                modalities = {
                    name: values[idx] for name, values in modality_arrays.items()
                }
                if "text" in modalities:
                    modalities["text"] = _maybe_embed_text(
                        modalities["text"], text_embedding
                    )
                fused = _concat_modalities(modalities, pool)
                if fused is None:
                    continue
                if example_modalities is None:
                    example_modalities = {
                        name: tuple(torch.as_tensor(val).shape)
                        for name, val in modalities.items()
                    }
                add_feature((int(dialogue_ids[idx]), int(utterance_ids[idx])), fused)
            return feature_map, example_modalities

    if isinstance(payload, dict):
        iterator = payload.items()
    elif isinstance(payload, (list, tuple)):
        iterator = enumerate(payload)
    else:
        iterator = []

    for key, sample in iterator:
        dialogue_id, utterance_id = _extract_ids_from_sample(sample, fallback_key=key)
        if dialogue_id is None or utterance_id is None:
            continue
        modalities = _extract_modality_dict(sample)
        if "text" in modalities:
            modalities["text"] = _maybe_embed_text(modalities["text"], text_embedding)
        fused = _concat_modalities(modalities, pool)
        if fused is None:
            fused = _to_1d_vector(sample, pool)
        if example_modalities is None and modalities:
            example_modalities = {
                name: tuple(torch.as_tensor(val).shape)
                for name, val in modalities.items()
            }
        add_feature((int(dialogue_id), int(utterance_id)), fused)

    return feature_map, example_modalities


class MMAlignFeatureStore:
    def __init__(self, features_dir, split=None, pool="mean"):
        embedding_dir = os.path.dirname(features_dir) if os.path.isfile(features_dir) else features_dir
        self.text_embedding, self.text_embedding_path = _load_mm_align_text_embedding(
            embedding_dir
        )
        pkl_path = _resolve_mm_align_pkl_path(features_dir, split)
        cache_key = (pkl_path, split, pool, self.text_embedding_path)
        cached = _MM_ALIGN_FEATURE_CACHE.get(cache_key)
        if cached is not None:
            self.feature_map, self.example_modalities = cached
        else:
            payload = _load_mm_align_payload(pkl_path)
            split_payload = _resolve_mm_align_split(payload, split)
            self.feature_map, self.example_modalities = _build_mm_align_feature_map(
                split_payload, pool=pool, text_embedding=self.text_embedding
            )
            _MM_ALIGN_FEATURE_CACHE[cache_key] = (
                self.feature_map,
                self.example_modalities,
            )

        if not self.feature_map:
            raise ValueError(
                "No MM-Align features loaded from {}. "
                "Please check the .pkl structure with inspect_mm_align_features.py.".format(
                    pkl_path
                )
            )

        self.pkl_path = pkl_path
        self.pool = pool
        self.feature_dim = next(iter(self.feature_map.values())).numel()
        for sample_id, value in self.feature_map.items():
            if value.numel() != self.feature_dim or not value.numel() or not torch.isfinite(value).all():
                raise ValueError("Invalid or inconsistent MM-Align features at {}".format(sample_id))

    def has(self, dialogue_id, utterance_id):
        return (int(dialogue_id), int(utterance_id)) in self.feature_map

    def get(self, dialogue_id, utterance_id):
        key = (int(dialogue_id), int(utterance_id))
        if key not in self.feature_map:
            raise KeyError("Missing MM-Align feature for dialogue {}, utterance {}.".format(*key))
        return self.feature_map[key]
