"""Projection, causal GRU memory, trajectory encoder and classification head.

Derived from Realtime-MERC final commit 537d832; parameter names are preserved.
"""

import torch
from torch import nn


class RealtimeMERC(nn.Module):
    def __init__(
        self,
        feature_dim=4096,
        proj_dim=768,
        num_classes=7,
        traj_dim=64,
        use_cem=True,
        use_text_embeddings=False,
        text_embed_dim=None,
        text_proj_dim=768,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.use_cem = use_cem
        self.use_text_embeddings = use_text_embeddings
        self.text_embed_dim = text_embed_dim
        self.text_proj_dim = text_proj_dim

        input_dim = feature_dim
        if self.use_text_embeddings:
            if text_embed_dim is None:
                raise ValueError("text_embed_dim is required when use_text_embeddings=True.")
            self.text_proj = nn.Linear(text_embed_dim, text_proj_dim)
            input_dim = feature_dim + text_proj_dim

        self.projector = nn.Linear(input_dim, proj_dim)
        self.cem_gru = nn.GRU(proj_dim, proj_dim, batch_first=True)
        self.trajectory_embed = nn.Embedding(num_classes, traj_dim)
        fused_dim = proj_dim * 2 + traj_dim
        self.classifier = nn.Linear(fused_dim, num_classes)
        self.classifier_residual = nn.Linear(fused_dim, num_classes)
        nn.init.zeros_(self.classifier_residual.weight)
        nn.init.zeros_(self.classifier_residual.bias)

    def encode_trajectory(self, trajectory_emotions, trajectory_mask, batch_size, device):
        if trajectory_emotions is None or trajectory_emotions.numel() == 0:
            return torch.zeros(batch_size, self.traj_dim, device=device)

        trajectory_emotions = trajectory_emotions.to(device)
        if trajectory_mask is None:
            trajectory_mask = trajectory_emotions >= 0
        else:
            trajectory_mask = trajectory_mask.to(device)

        safe_indices = trajectory_emotions.clone()
        safe_indices[safe_indices < 0] = 0
        embeddings = self.trajectory_embed(safe_indices)
        mask = trajectory_mask.unsqueeze(-1).float()
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        fused_features,
        trajectory_emotions=None,
        trajectory_mask=None,
        dialogue_ids=None,
        cem_state=None,
        text_embeddings=None,
    ):
        if fused_features.dim() == 3:
            fused_features = fused_features.view(fused_features.size(0), -1)
        if self.use_text_embeddings:
            if text_embeddings is None:
                raise ValueError("This checkpoint requires explicit text_embeddings.")
            text_proj = self.text_proj(text_embeddings)
            fused_features = torch.cat([fused_features, text_proj], dim=1)
        traj_embeds = self.encode_trajectory(
            trajectory_emotions,
            trajectory_mask,
            fused_features.size(0),
            fused_features.device,
        )
        e_t = torch.relu(self.projector(fused_features))

        if not self.use_cem:
            cem_t = torch.zeros_like(e_t)
            fused = torch.cat([e_t, cem_t, traj_embeds], dim=1)
            logits = self.classifier(fused) + self.classifier_residual(fused)
            return logits, cem_state

        cem_outputs = []
        if cem_state is None:
            cem_state = {"dialogue_id": None, "memory": None}
        current_dialogue = cem_state.get("dialogue_id")
        memory = cem_state.get("memory")

        if dialogue_ids is None:
            dialogue_ids = [None] * e_t.size(0)

        for idx, dialogue_id in enumerate(dialogue_ids):
            if memory is None or dialogue_id != current_dialogue:
                memory = torch.zeros(
                    1, 1, e_t.size(1), device=e_t.device, dtype=e_t.dtype
                )
                current_dialogue = dialogue_id
            cem_out, memory = self.cem_gru(e_t[idx : idx + 1].unsqueeze(1), memory)
            cem_outputs.append(cem_out.squeeze(0).squeeze(0))
            memory = memory.detach()

        cem_t = torch.stack(cem_outputs, dim=0) if cem_outputs else torch.zeros_like(e_t)
        fused = torch.cat([e_t, cem_t, traj_embeds], dim=1)
        logits = self.classifier(fused) + self.classifier_residual(fused)
        cem_state = {"dialogue_id": current_dialogue, "memory": memory}
        return logits, cem_state
