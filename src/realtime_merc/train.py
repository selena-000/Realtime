"""Train the task head from ordered, precomputed multimodal features."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .cli import add_data_args, get_data_config, get_device, make_loader, write_json
from .engine import run_epoch
from .features import EMOTIONS
from .model import RealtimeMERC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser, defaults=False)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--proj-dim", type=int, default=768)
    parser.add_argument("--traj-dim", type=int, default=64)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-cem", dest="use_cem", action="store_true")
    group.add_argument("--no-cem", dest="use_cem", action="store_false")
    parser.set_defaults(use_cem=None)
    parser.add_argument("--seed", type=int, help="Default 42, or the saved seed when resuming.")
    parser.add_argument("--resume", help="Explicit checkpoint to resume; never resume automatically.")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and lr must be positive.")
    device = get_device(args.device)
    checkpoint = None
    model = None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
    args.seed = args.seed if args.seed is not None else (checkpoint or {}).get("seed", 42)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = get_data_config(args, checkpoint)
    if checkpoint and checkpoint.get("data_config", config) != config:
        raise ValueError("Resuming requires the saved data and trajectory protocol.")
    train_loader = make_loader(args.train_csv, args.features_dir, "train", config, args)
    val_loader = make_loader(args.val_csv, args.features_dir, "dev", config, args)
    feature_dim = train_loader.dataset.feature_dim
    if val_loader.dataset.feature_dim != feature_dim:
        raise ValueError("Training and validation feature dimensions differ.")
    if model is None:
        model_config = dict(feature_dim=feature_dim, proj_dim=args.proj_dim,
                            num_classes=len(EMOTIONS), traj_dim=args.traj_dim,
                            use_cem=args.use_cem is not False, use_text_embeddings=False)
        model = RealtimeMERC(**model_config).to(device)
    else:
        model_config = checkpoint["model_config"]
        if model_config.get("feature_dim") != feature_dim:
            raise ValueError("Checkpoint and training feature dimensions differ.")
        if args.use_cem is not None and args.use_cem != model.use_cem:
            raise ValueError("Cannot change CEM architecture while resuming.")
        if model.use_text_embeddings:
            raise ValueError("The clean CLI trains feature-only heads; provide a custom text embedding pipeline.")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best_f1 = 0, -1.0
    if checkpoint:
        if "optimizer_state" not in checkpoint:
            raise ValueError("Resume requires last.pt with optimizer state, not best.pt.")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch, best_f1 = checkpoint["epoch"], checkpoint["best_f1"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "cuda_rng_states" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    output = Path(args.output_dir)
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("Resume into the original output directory so best.pt and metrics are retained.")
    if not args.resume and any((output / name).exists() for name in ["best.pt", "last.pt", "metrics.json"]):
        raise FileExistsError("Output already contains a run. Choose a new directory or explicit --resume.")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.resume and (output / "metrics.json").is_file():
        rows = json.loads((output / "metrics.json").read_text())["epochs"]
        rows = [row for row in rows if row["epoch"] <= start_epoch]
    for epoch in range(start_epoch, args.epochs):
        train = run_epoch(model, train_loader, device, optimizer)
        validation = run_epoch(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": train["loss"],
               "val_loss": validation["loss"], "val_weighted_f1": validation["weighted_f1"]}
        rows.append(row)
        print("Epoch {}: train_loss={:.6f} val_weighted_f1={:.6f}".format(
            epoch + 1, row["train_loss"], row["val_weighted_f1"]))
        if validation["weighted_f1"] > best_f1:
            best_f1 = validation["weighted_f1"]
            save_checkpoint(output / "best.pt", model, model_config, config,
                            epoch + 1, best_f1, seed=args.seed)
        save_checkpoint(output / "last.pt", model, model_config, config,
                        epoch + 1, best_f1, optimizer=optimizer, seed=args.seed)
    write_json(output / "metrics.json", {"data_config": config, "epochs": rows,
               "best_val_weighted_f1": best_f1,
               "skipped_samples": {"train": train_loader.dataset.missing_count,
                                   "validation": val_loader.dataset.missing_count}})


if __name__ == "__main__":
    main()
