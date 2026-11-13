import csv

import pytest


@pytest.fixture
def feature_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    from realtime_merc.data import MELDFeatureDataset

    # Two dialogues deliberately supplied out of chronological order.
    rows = [(1, 1, "anger"), (0, 2, "sadness"), (0, 0, "neutral"),
            (1, 0, "joy"), (0, 1, "joy")]
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    generator = torch.Generator().manual_seed(19)
    for dialogue, utterance, _ in rows:
        torch.save(torch.randn(6, generator=generator),
                   feature_dir / "dia{}_utt{}.pt".format(dialogue, utterance))

    def make(labels=None, **kwargs):
        path = tmp_path / "metadata.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Dialogue_ID", "Utterance_ID", "Emotion"])
            writer.writerows((dia, utt, labels[index] if labels else label)
                             for index, (dia, utt, label) in enumerate(rows))
        return MELDFeatureDataset(path, feature_dir, feature_set="emotion_llama", **kwargs)

    return make, feature_dir, tmp_path
