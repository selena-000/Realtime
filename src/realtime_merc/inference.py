"""Streaming inference without access to current or historical target labels."""

import torch

from .data import pack_trajectories
from .features import EMOTIONS


class RealtimePredictor:
    def __init__(self, model, trajectory_source="predicted"):
        if trajectory_source not in {"predicted", "none"}:
            raise ValueError("Streaming inference supports predicted or none trajectories.")
        self.model = model.eval()
        self.trajectory_source = trajectory_source
        self.reset()

    def reset(self):
        self.dialogue_id = None
        self.memory = None
        self.history = []

    @torch.no_grad()
    def step(self, features, dialogue_id, text_embeddings=None):
        if dialogue_id != self.dialogue_id:
            self.reset()
            self.dialogue_id = dialogue_id
        device = next(self.model.parameters()).device
        features = torch.as_tensor(features, dtype=torch.float32, device=device).reshape(1, -1)
        if features.shape[1] != self.model.projector.in_features - (
            self.model.text_proj_dim if self.model.use_text_embeddings else 0
        ):
            raise ValueError("Input feature dimension does not match the model.")
        if not torch.isfinite(features).all():
            raise ValueError("Input features must be finite.")
        trajectory = self.history if self.trajectory_source == "predicted" else []
        values, mask = pack_trajectories([trajectory], device)
        if text_embeddings is not None:
            text_embeddings = torch.as_tensor(text_embeddings, dtype=features.dtype,
                                              device=device).reshape(1, -1)
        logits, self.memory = self.model(features, values, mask, [dialogue_id],
                                        self.memory, text_embeddings=text_embeddings)
        prediction = int(logits.argmax(dim=-1).item())
        self.history.append(prediction)
        return {"label_id": prediction, "label": EMOTIONS[prediction],
                "probabilities": logits.softmax(dim=-1)[0].cpu().tolist()}
