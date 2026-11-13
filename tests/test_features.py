import csv
import pickle

import pytest

torch = pytest.importorskip("torch")
from realtime_merc.data import MELDFeatureDataset
from realtime_merc.features import MMAlignFeatureStore


def test_mm_align_pooling_modality_order_and_split(tmp_path):
    path = tmp_path / "features.pkl"
    sample = {"dialogue_id": 0, "utterance_id": 0,
              "vision": [[1.0, 2.0], [3.0, 4.0]], "audio": [5.0], "text": [6.0, 7.0]}
    with path.open("wb") as handle:
        pickle.dump({"train": [sample], "test": [dict(sample, audio=[9.0])]}, handle)
    store = MMAlignFeatureStore(str(path), split="train")
    torch.testing.assert_close(store.get(0, 0), torch.tensor([2., 3., 5., 6., 7.]))
    test_store = MMAlignFeatureStore(str(path), split="test")
    assert test_store.get(0, 0)[2] == 9.0


def test_ambiguous_files_require_explicit_selection(tmp_path):
    for name in ["a_train.pkl", "b_train.pkl"]:
        with (tmp_path / name).open("wb") as handle:
            pickle.dump([], handle)
    with pytest.raises(ValueError, match="Ambiguous"):
        MMAlignFeatureStore(str(tmp_path), split="train")


def test_missing_features_fail_by_default(feature_dataset):
    make, feature_dir, _ = feature_dataset
    (feature_dir / "dia0_utt1.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing features"):
        make()
    selected = make(skip_missing=True)
    assert len(selected) == 4 and selected.missing_count == 1


def test_bad_feature_values_fail(feature_dataset):
    make, feature_dir, _ = feature_dataset
    torch.save(torch.full((6,), float("nan")), feature_dir / "dia0_utt0.pt")
    with pytest.raises(ValueError, match="finite"):
        make()


def test_mismatched_dimensions_fail(feature_dataset):
    make, feature_dir, _ = feature_dataset
    torch.save(torch.zeros(7), feature_dir / "dia0_utt2.pt")
    dataset = make()
    with pytest.raises(ValueError, match="dimensions"):
        dataset[2]


def test_duplicate_metadata_ids_fail(tmp_path):
    path = tmp_path / "metadata.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Dialogue_ID", "Utterance_ID", "Emotion"])
        writer.writerows([(0, 0, "neutral"), (0, 0, "joy")])
    with pytest.raises(ValueError, match="Duplicate"):
        MELDFeatureDataset(path, tmp_path, feature_set="emotion_llama")


def test_duplicate_feature_ids_fail(tmp_path):
    path = tmp_path / "features.pkl"
    sample = {"dialogue_id": 0, "utterance_id": 0, "audio": [1.0]}
    with path.open("wb") as handle:
        pickle.dump({"train": [sample, sample]}, handle)
    with pytest.raises(ValueError, match="Duplicate"):
        MMAlignFeatureStore(str(path), split="train")
