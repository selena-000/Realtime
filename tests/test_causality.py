import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from realtime_merc.data import collate_batch
from realtime_merc.engine import run_epoch
from realtime_merc.inference import RealtimePredictor
from realtime_merc.model import RealtimeMERC


def small_model():
    torch.manual_seed(7)
    return RealtimeMERC(feature_dim=6, proj_dim=8, traj_dim=3)


def run(model, dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return run_epoch(model, loader)["predictions"]


def test_causal_predictions_do_not_read_any_target_labels(feature_dataset):
    make, _, _ = feature_dataset
    original = make()
    changed = make(labels=["disgust"] * 5)
    assert run(small_model(), original, 2) == run(small_model(), changed, 2)
    assert all(item["trajectory_emotions"] == [] for item in original.items)


def test_future_features_cannot_change_past_predictions(feature_dataset):
    make, feature_dir, _ = feature_dataset
    dataset = make()
    before = run(small_model(), dataset, 2)
    torch.save(torch.full((6,), 1000.0), feature_dir / "dia0_utt2.pt")
    after = run(small_model(), dataset, 2)
    assert before[:2] == after[:2]


def test_batch_boundaries_do_not_change_causal_predictions(feature_dataset):
    make, _, _ = feature_dataset
    dataset = make()
    assert run(small_model(), dataset, 1) == run(small_model(), dataset, 4)


def test_streaming_matches_dataset_and_resets_between_dialogues(feature_dataset):
    make, _, _ = feature_dataset
    dataset = make()
    expected = run(small_model(), dataset, 3)
    predictor = RealtimePredictor(small_model())
    actual = []
    for item in dataset:
        prediction = predictor.step(item["fused_features"], item["dialogue_id"])
        assert abs(sum(prediction["probabilities"]) - 1.0) < 1e-6
        actual.append(prediction["label_id"])
    assert actual == [item["label_id"] for item in expected]
    first_other = next(item for item in dataset if item["dialogue_id"] == 1)
    fresh = RealtimePredictor(small_model()).step(first_other["fused_features"], 1)
    assert fresh["label_id"] == actual[3]


def test_oracle_requires_explicit_selection_and_retains_old_protocol(feature_dataset):
    make, _, _ = feature_dataset
    oracle = make(mode="offline", trajectory_source="oracle")
    assert oracle[0]["trajectory_emotions"] == [0, 1, 2]
    past_only = make(trajectory_source="oracle")
    assert past_only[0]["trajectory_emotions"] == []
    assert past_only[1]["trajectory_emotions"] == [0]
    with pytest.raises(ValueError, match="offline"):
        make(mode="offline")


def test_unlabeled_dataset_does_not_read_emotion_column(feature_dataset):
    make, _, _ = feature_dataset
    dataset = make(labels=["deliberately invalid target"] * 5, require_labels=False)
    assert all(dataset[index]["emotion"] == -1 for index in range(len(dataset)))
    with pytest.raises(ValueError, match="Unknown emotion"):
        make(labels=["invalid target"] * 5)


def test_text_embedding_checkpoint_cannot_silently_use_zeros():
    model = RealtimeMERC(feature_dim=6, proj_dim=8, use_text_embeddings=True, text_embed_dim=4)
    with pytest.raises(ValueError, match="explicit text_embeddings"):
        model(torch.zeros(1, 6))
    logits, _ = model(torch.zeros(1, 6), text_embeddings=torch.zeros(1, 4))
    assert logits.shape == (1, 7)
