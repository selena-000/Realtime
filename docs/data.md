# Prepared-feature contract

All commands preserve chronological order within a dialogue. Labels use this
fixed order: `neutral`, `joy`, `sadness`, `anger`, `surprise`, `fear`, `disgust`.
Training and evaluation require `Dialogue_ID`, `Utterance_ID`, `Emotion` columns
(case-insensitive aliases are accepted). Prediction only requires the two IDs.
IDs must be integers and the pair must be unique within a split.

## Per-utterance tensors

For `emotion_llama` and `conv_emotion`, provide one `.pt` file for each ID pair:

```text
features/
  train/dia0_utt0.pt
  dev/dia0_utt0.pt
  test/dia0_utt0.pt
```

Each file is a tensor, or `{"fused_features": tensor}`. Multi-dimensional
tensors are flattened. Every utterance must have the same flattened dimension.

## MM-Align

Use a single `.pkl` with split keys (`train`, `dev`/`val`, `test`), or one file
per split. Ambiguous filenames fail; provide an explicit `.pkl` path to resolve
them. A split may be a list of sample dictionaries:

```python
{
    "dialogue_id": 0,
    "utterance_id": 0,
    "vision": vision_array,
    "audio": audio_array,
    "text": text_array,
}
```

A split may also be a dictionary keyed by ID pairs such as `(0, 0)` or
`"dia0_utt0"`, or a dictionary of aligned arrays with dialogue/utterance IDs
and modality arrays. The adapter concatenates modalities in
**vision, audio, text** order. Multi-dimensional modality arrays are mean-pooled
over their first dimension by default; `--mm-align-pool flatten` flattens them.
Absent modalities are omitted; the resulting fused dimension must be constant.

An optional `embedding.p`, `embedding.pkl` or `embedding.npy` beside the feature
file can map one-dimensional text token IDs to embeddings before pooling.
Use the inspection command to confirm shapes and dimensions before training.

Features and their source encoders must themselves satisfy your intended
causality requirements. Causal state handling here cannot undo future context
already embedded in an upstream feature vector.
