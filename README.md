# Realtime-MERC

A small, feature-based implementation of **Real-Time Multimodal Emotion
Recognition in Conversations with MLLMs**
([paper](https://doi.org/10.1007/978-981-92-2480-7_22)).

This package contains the feature projection, Causal Emotion Memory (CEM),
emotion trajectory encoder and seven-class classifier, plus training,
evaluation and streaming inference. It is intended as a starting point for
extending the method. It consumes prepared features; upstream feature
extractors, foundation models and datasets are separate.

## Install

Use Python 3.9 or newer. Install the appropriate PyTorch build for your machine,
then install this package:

```bash
python -m pip install -e .
# For development and verification:
python -m pip install -e '.[dev]'
python -m pytest
```

The core dependencies are PyTorch, NumPy and pandas. CUDA is optional; every
command accepts `--device cpu`. There is no MiniGPT, Transformers or Gradio
dependency in the feature-based pipeline.

## Data

Download MELD separately from its official distribution. Provide its
`train_sent_emo.csv`, `dev_sent_emo.csv` and `test_sent_emo.csv` files and one
of these feature formats:

- `--feature-set mm_align`: split `.pkl` files or an explicit combined `.pkl`.
- `--feature-set emotion_llama` or `conv_emotion`: per-utterance tensors stored
  as `train/dia0_utt0.pt`, `dev/dia0_utt0.pt`, `test/dia0_utt0.pt`, etc.

These option names describe prepared feature formats; the package does not
extract them from raw audio/video. A `.pt` file must contain a tensor or a
dictionary with `fused_features`. Features must be finite and share one
dimension. See [the data contract](docs/data.md) for MM-Align formats.

Only load pickle feature files from a trusted source. Missing features fail by
default; `--skip-missing` opts into skipping and reports the excluded count.

```bash
realtime-merc-inspect --features-dir /path/to/features.pkl --split train
```

## Train

```bash
realtime-merc-train \
  --train-csv /path/to/train_sent_emo.csv \
  --val-csv /path/to/dev_sent_emo.csv \
  --features-dir /path/to/features \
  --feature-set mm_align \
  --mode realtime --trajectory-source predicted \
  --output-dir outputs/causal-cem
```

Utterances are ordered by dialogue and utterance IDs. CEM state and trajectory
history reset at dialogue boundaries. `predicted` uses only preceding model
predictions, during both training and evaluation. Training labels are used for
the loss, not trajectory construction. `--no-cem` disables recurrent memory;
`--trajectory-source none` also removes the trajectory input.

Outputs are `best.pt`, `last.pt` and `metrics.json`. Resume explicitly with
`--resume outputs/causal-cem/last.pt --epochs 40`. A new run will not silently
overwrite a previous run. Checkpoints record architecture, label order, feature
format and trajectory protocol.

## Evaluate and predict

```bash
realtime-merc-eval \
  --test-csv /path/to/test_sent_emo.csv \
  --features-dir /path/to/features \
  --checkpoint outputs/causal-cem/best.pt \
  --output outputs/causal-cem/test.json

realtime-merc-predict \
  --csv /path/to/utterance_ids.csv \
  --features-dir /path/to/features \
  --checkpoint outputs/causal-cem/best.pt \
  --output outputs/predictions.json
```

Prediction metadata needs only `Dialogue_ID` and `Utterance_ID`. Prediction
ignores any labels in the CSV and always uses a causal protocol. New checkpoints
carry their feature settings; pass `--feature-set` explicitly for legacy heads
whose feature format differs from the default MM-Align format.

For streaming applications and new feature extractors:

```python
from realtime_merc.checkpoint import load_checkpoint
from realtime_merc.inference import RealtimePredictor

model, metadata = load_checkpoint("outputs/causal-cem/best.pt")
predictor = RealtimePredictor(model)
result = predictor.step(current_feature_vector, dialogue_id="conversation-1")
print(result["label"])
```

The optional original text-projection layer is retained in the Python model
API for checkpoint compatibility. A model configured with text embeddings
requires explicit `text_embeddings` on each call. The CLI uses feature-only
heads; the large upstream text encoder is not bundled.

## Original experiment compatibility

The original `final` implementation constructs trajectories from **ground-truth
CSV labels**. Its `realtime` setting uses past true labels; its `offline`
setting includes the entire dialogue, including the current target. These
are oracle-conditioned experiments, not autonomous online prediction.

The old architecture and state-dict keys are preserved. To run that original
protocol explicitly, use `--trajectory-source oracle` together with
`--mode realtime` or `--mode offline`, for training and evaluation. These
settings are recorded in new checkpoints and results. The old published head
files can be loaded directly, but are not included in this source repository.

**Results obtained with the new default predicted trajectory must not be
reported as reproducing the original oracle-conditioned scores.** This cleanup
does not re-establish the paper's MELD results. See
[protocol and verification notes](docs/protocol.md).

## Layout

```text
src/realtime_merc/
  model.py           # Projection, CEM, trajectory embedding, classifier
  features.py        # Prepared-feature adapters
  data.py            # Ordered data and explicit trajectory protocols
  engine.py          # Shared training/evaluation pass
  inference.py       # Streaming prediction with no target labels
  checkpoint.py      # Checkpoint loading, saving and compatibility
  train.py           # Training CLI
  evaluate.py        # Evaluation CLI
  predict.py         # Unlabeled prediction CLI
  inspect_features.py
tests/               # Causality, compatibility and pipeline checks
```

This is a selected source snapshot from original commit
`537d832022971fd142c6253872de02b72e143366`, not a mirror of its research history.
Datasets, weights, caches, logs and local environments are excluded. Original
MLP/GRU/DialogueRNN comparison implementations and raw-feature extraction
remain in the research repository; they are not silently represented as part
of this cleaned main-method package.

## License and citation

BSD 3-Clause. Existing upstream notices are retained; see
[third-party notices](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff).
Dataset, feature and model licenses are separate.
