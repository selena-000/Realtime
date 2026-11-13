import pytest

torch = pytest.importorskip("torch")
from realtime_merc.checkpoint import load_checkpoint, save_checkpoint
from realtime_merc.model import RealtimeMERC


def test_checkpoint_roundtrip_preserves_logits_and_protocol(tmp_path):
    torch.manual_seed(4)
    config = dict(feature_dim=6, proj_dim=8, traj_dim=3, use_cem=True)
    protocol = dict(feature_set="emotion_llama", mm_align_pool="mean",
                    mode="realtime", trajectory_source="predicted")
    original = RealtimeMERC(**config).eval()
    path = tmp_path / "head.pt"
    save_checkpoint(path, original, config, protocol, epoch=1, best_f1=0.5)
    loaded, metadata = load_checkpoint(path)
    assert metadata["data_config"] == protocol
    features = torch.randn(3, 6)
    first, first_memory = original(features, dialogue_ids=[0, 0, 1])
    second, second_memory = loaded(features, dialogue_ids=[0, 0, 1])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_memory["memory"], second_memory["memory"], rtol=0, atol=0)


def test_incompatible_label_order_fails(tmp_path):
    config = dict(feature_dim=6, proj_dim=8)
    path = tmp_path / "bad.pt"
    torch.save({"model_config": config, "model_state": RealtimeMERC(**config).state_dict(),
                "label_names": ["incorrect"]}, path)
    with pytest.raises(ValueError, match="label order"):
        load_checkpoint(path)
