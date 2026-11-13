# Protocol and verification

The architecture and label-supplied trajectory construction come from the
inspected `final` source snapshot. The predicted-trajectory option was added
during this cleanup. The settings below describe code behavior; they do not
establish the experimental settings used for the paper's reported results.

| Mode | Trajectory source | Model input history | Intended use |
| --- | --- | --- | --- |
| realtime | predicted | Preceding model predictions | Default training and deployment |
| realtime | none | Empty trajectory; optional CEM state | Ablation |
| realtime | oracle | Preceding ground-truth labels | Match inspected label-based input construction |
| offline | oracle | All true labels, including current target | Match inspected label-based input construction |

Here, `oracle` names trajectories supplied from dataset annotations. In
`realtime` mode those annotations cover preceding utterances; in `offline`
mode they cover the whole dialogue, including the current target. `predicted`
uses preceding model predictions instead. These are different input settings,
so comparisons should explicitly state and match the trajectory source and
context mode. Inspecting the loader alone does not establish how an existing
checkpoint was trained or how a reported paper score was obtained.

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
