"""Arguments common to the small feature-based commands."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import MELDFeatureDataset, collate_batch, resolve_features_dir


def add_data_args(parser, defaults=True):
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--feature-set", choices=["mm_align", "emotion_llama", "conv_emotion"],
                        default="mm_align" if defaults else None)
    parser.add_argument("--mm-align-pool", choices=["mean", "flatten"], default="mean" if defaults else None)
    parser.add_argument("--mode", choices=["realtime", "offline"], default="realtime" if defaults else None)
    parser.add_argument("--trajectory-source", choices=["predicted", "none", "oracle"],
                        default="predicted" if defaults else None)
    parser.add_argument("--skip-missing", action="store_true",
                        help="Explicitly allow missing feature samples; report their count.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")


def get_device(name):
    return "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name)


def get_data_config(args, checkpoint=None):
    saved = (checkpoint or {}).get("data_config", {})
    config = {key: getattr(args, key, None) or saved.get(key, value) for key, value in {
        "feature_set": "mm_align", "mm_align_pool": "mean", "mode": "realtime",
        "trajectory_source": "predicted",
    }.items()}
    if config["trajectory_source"] == "oracle":
        print("Protocol: legacy oracle labels; these scores do not measure autonomous online inference.")
    return config


def make_loader(csv_path, features_dir, split, config, args, require_labels=True):
    dataset = MELDFeatureDataset(
        csv_path, resolve_features_dir(features_dir, split), split=split,
        skip_missing=args.skip_missing, require_labels=require_labels, **config,
    )
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative.")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, collate_fn=collate_batch)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
