import csv
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_cpu_training_resume_evaluation_and_unlabeled_prediction(tmp_path):
    generator = torch.Generator().manual_seed(11)
    for split in ["train", "dev", "test"]:
        directory = tmp_path / "features" / split
        directory.mkdir(parents=True)
        with (tmp_path / (split + ".csv")).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Dialogue_ID", "Utterance_ID", "Emotion"])
            for index in range(6):
                torch.save(torch.randn(6, generator=generator), directory / "dia0_utt{}.pt".format(index))
                writer.writerow([0, index, ["neutral", "joy", "sadness"][index % 3]])

    def command(module, *args):
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        return subprocess.run([sys.executable, "-m", "realtime_merc." + module,
                               "--features-dir", str(tmp_path / "features"),
                               "--feature-set", "emotion_llama", "--device", "cpu",
                               *map(str, args)], check=True, capture_output=True, text=True, env=env)

    train_args = ["--train-csv", tmp_path / "train.csv", "--val-csv", tmp_path / "dev.csv",
                  "--output-dir", tmp_path / "run", "--proj-dim", 8, "--traj-dim", 3]
    command("train", *train_args, "--epochs", 1)
    assert (tmp_path / "run/best.pt").is_file()
    command("train", *train_args, "--epochs", 2, "--resume", tmp_path / "run/last.pt")
    history = json.loads((tmp_path / "run/metrics.json").read_text())
    assert [row["epoch"] for row in history["epochs"]] == [1, 2]
    command("evaluate", "--test-csv", tmp_path / "test.csv", "--checkpoint", tmp_path / "run/best.pt",
            "--output", tmp_path / "evaluation.json")
    result = json.loads((tmp_path / "evaluation.json").read_text())
    assert result["samples"] == 6 and 0 <= result["weighted_f1"] <= 1
    assert result["trajectory_source"] == "predicted"
    with (tmp_path / "unlabeled.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Dialogue_ID", "Utterance_ID"])
        writer.writerows((0, index) for index in range(6))
    command("predict", "--csv", tmp_path / "unlabeled.csv", "--checkpoint", tmp_path / "run/best.pt",
            "--output", tmp_path / "prediction.json")
    prediction = json.loads((tmp_path / "prediction.json").read_text())
    assert prediction["samples"] == 6 and prediction["weighted_f1"] is None
    assert prediction["predictions"] == result["predictions"]
