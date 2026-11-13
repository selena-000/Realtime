"""Compare the clean model with the actual original code and classifier heads."""

import argparse
import ast
import inspect
import io
import json
import subprocess

import torch
from torch import nn

from realtime_merc.checkpoint import MODEL_KEYS
from realtime_merc.model import RealtimeMERC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-ref", default="537d832022971fd142c6253872de02b72e143366")
    args = parser.parse_args()
    torch.set_num_threads(1)

    def read(path):
        return subprocess.check_output(["git", "-C", args.source_repo, "show", args.source_ref + ":" + path])

    module = ast.parse(read("train_baseline_meld.py").decode())
    original_class = next(node for node in module.body if isinstance(node, ast.ClassDef)
                          and node.name == "BaselineMELDModel")
    isolated = ast.Module(body=[original_class], type_ignores=[])
    namespace = {"torch": torch, "nn": nn}
    exec(compile(isolated, "original_model", "exec"), namespace)
    original_type = namespace["BaselineMELDModel"]
    generator = torch.Generator().manual_seed(23)
    results = []
    for name in ["mm_align_cem0_best.pt", "mm_align_cem1_best.pt"]:
        checkpoint = torch.load(io.BytesIO(read("checkpoints/" + name)), map_location="cpu", weights_only=True)
        config = {k: v for k, v in checkpoint["model_config"].items() if k in MODEL_KEYS}
        original, clean = original_type(**config).eval(), RealtimeMERC(**config).eval()
        original.load_state_dict(checkpoint["model_state"], strict=True)
        clean.load_state_dict(checkpoint["model_state"], strict=True)
        features = torch.randn(5, config["feature_dim"], generator=generator)
        trajectory = torch.tensor([[0, -1], [0, 1], [1, 2], [-1, -1], [3, -1]])
        arguments = (features, trajectory, trajectory >= 0, [0, 0, 0, 1, 1])
        with torch.no_grad():
            expected, expected_state = original(*arguments)
            actual, actual_state = clean(*arguments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if expected_state is not None:
            torch.testing.assert_close(actual_state["memory"], expected_state["memory"], rtol=0, atol=0)
        results.append({"checkpoint": name, "strict_load": True, "exact_logits": True,
                        "feature_dim": config["feature_dim"], "use_cem": config["use_cem"]})
    print(json.dumps({"source_ref": args.source_ref, "torch": str(torch.__version__), "results": results}, indent=2))


if __name__ == "__main__":
    main()
