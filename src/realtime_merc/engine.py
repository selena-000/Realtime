"""One ordered pass shared by training, validation and prediction."""

import torch
from torch.nn import functional as F

from .data import check_protocol, pack_trajectories
from .features import EMOTIONS
from .metrics import weighted_f1


def run_epoch(model, loader, device="cpu", optimizer=None):
    dataset = loader.dataset
    check_protocol(dataset.mode, dataset.trajectory_source)
    training = optimizer is not None
    model.train(training)
    state, current_dialogue, history = None, None, []
    total_loss, count, targets, predictions, records = 0.0, 0, [], [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            features = batch["fused_features"].to(device)
            labels = batch["emotion"].to(device)
            if dataset.trajectory_source == "predicted":
                outputs = []
                for index, dialogue_id in enumerate(batch["dialogue_id"]):
                    if dialogue_id != current_dialogue:
                        current_dialogue, history, state = dialogue_id, [], None
                    values, mask = pack_trajectories([history], device)
                    logits, state = model(features[index:index+1], values, mask,
                                          [dialogue_id], state)
                    outputs.append(logits)
                    history.append(int(logits.detach().argmax(dim=-1).item()))
                logits = torch.cat(outputs, dim=0)
            else:
                logits, state = model(
                    features, batch["trajectory_emotions"].to(device),
                    batch["trajectory_mask"].to(device), batch["dialogue_id"], state,
                )
            valid = labels >= 0
            if valid.any():
                loss = F.cross_entropy(logits[valid], labels[valid])
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                n = int(valid.sum().item())
                total_loss += float(loss.detach().item()) * n
                count += n
                targets.extend(labels[valid].detach().cpu().tolist())
                predictions.extend(logits[valid].detach().argmax(dim=-1).cpu().tolist())
            elif training:
                raise ValueError("Training requires labeled data.")
            for dia, utt, pred in zip(batch["dialogue_id"], batch["utterance_id"],
                                      logits.detach().argmax(dim=-1).cpu().tolist()):
                records.append({"dialogue_id": dia, "utterance_id": utt,
                                "label_id": pred, "label": EMOTIONS[pred]})
    return {"loss": total_loss / count if count else None,
            "weighted_f1": weighted_f1(targets, predictions) if count else None,
            "samples": len(records), "skipped_samples": dataset.missing_count,
            "mode": dataset.mode, "trajectory_source": dataset.trajectory_source,
            "predictions": records}
