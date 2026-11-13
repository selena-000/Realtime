"""Evaluate a trained classifier with a named trajectory protocol."""

import argparse

from .checkpoint import load_checkpoint
from .cli import add_data_args, get_data_config, get_device, make_loader, write_json
from .engine import run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser, defaults=False)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="Write metrics and per-utterance predictions as JSON.")
    args = parser.parse_args()
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if model.use_text_embeddings:
        raise ValueError("Use the Python API with explicit text_embeddings for this checkpoint.")
    config = get_data_config(args, checkpoint)
    loader = make_loader(args.test_csv, args.features_dir, "test", config, args)
    if checkpoint["model_config"].get("feature_dim") != loader.dataset.feature_dim:
        raise ValueError("Checkpoint and dataset feature dimensions differ.")
    result = run_epoch(model, loader, device)
    result["checkpoint_training_protocol"] = checkpoint.get("data_config", "legacy_unspecified")
    print("Test Weighted-F1: {:.6f}".format(result["weighted_f1"]))
    print("Mode: {}; trajectory: {}; samples: {}; skipped: {}".format(
        config["mode"], config["trajectory_source"], result["samples"], result["skipped_samples"]))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
