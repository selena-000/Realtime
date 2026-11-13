# Protocol and verification

The architecture is copied from the original `final` branch. Reorganizing it
does not establish the validity of the paper's reported scores.

| Mode | Trajectory source | Model input history | Intended use |
| --- | --- | --- | --- |
| realtime | predicted | Preceding model predictions | Default training and deployment |
| realtime | none | Empty trajectory; optional CEM state | Ablation |
| realtime | oracle | Preceding ground-truth labels | Explicit original experiment |
| offline | oracle | All true labels, including current target | Explicit original experiment |

The `offline/oracle` experiment exposes the current target through its input.
Its F1 should not be interpreted as ordinary held-out classification accuracy.
The `realtime/oracle` experiment assumes past true labels are supplied externally.
Neither setting is equivalent to deployment without annotations. The original
checkpoints were trained under those original protocols; causal evaluation of
an old checkpoint is a different experiment.

During predicted-trajectory training, argmax predictions build the trajectory
without gradients. The recurrent memory is detached at every utterance, as in
the original source. This implementation does not add full-dialogue BPTT.

## Checks

`pytest` covers label independence of causal predictions, reset at dialogue
boundaries, batch-boundary consistency, direct streaming inference, feature
formats, missing-data handling, checkpoint round trips, and a CPU CLI training,
resume, evaluation and unlabeled-prediction run on synthetic data. Synthetic
examples are generated in temporary directories; no MELD data is committed.

`scripts/check_source.py` independently checks syntax, the release file
allowlist and excluded artifacts without requiring PyTorch.

For original-head validation on a machine with PyTorch:

```bash
python scripts/verify_original.py \
  --source-repo /path/to/original/Emotion-LLaMA \
  --source-ref 537d832022971fd142c6253872de02b72e143366
```

It extracts the original model class into an isolated namespace, loads both
original MM-Align classifier heads from Git, compares their logits and memory
with the clean model on deterministic inputs, and checks that the original
checkpoint dictionaries can be loaded strictly. It does not rerun MELD or
validate published F1 claims. Generated verification output belongs outside
the source snapshot.
