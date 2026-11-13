"""Predict on prepared features and metadata without reading target labels."""

import argparse

from .checkpoint import load_checkpoint
from .cli import add_data_args, get_data_config, get_device, make_loader, write_json
from .engine import run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser, defaults=False)
    parser.add_argument("--csv", required=True, help="Metadata with dialogue and utterance IDs; labels are ignored.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "val", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if model.use_text_embeddings:
        raise ValueError("Use the Python API with explicit text_embeddings for this checkpoint.")
    config = get_data_config(args, checkpoint)
    # Prediction never inherits an oracle protocol from a legacy experiment.
    config["mode"] = "realtime"
    config["trajectory_source"] = args.trajectory_source or "predicted"
    if config["trajectory_source"] == "oracle":
        parser.error("Prediction cannot use oracle labels. Choose predicted or none.")
    loader = make_loader(args.csv, args.features_dir, args.split, config, args, require_labels=False)
    if checkpoint["model_config"].get("feature_dim") != loader.dataset.feature_dim:
        raise ValueError("Checkpoint and dataset feature dimensions differ.")
    result = run_epoch(model, loader, device)
    write_json(args.output, result)
    print("Predicted {} samples; skipped {}.".format(result["samples"], result["skipped_samples"]))


if __name__ == "__main__":
    main()
