# Validation results

Verified on 14 November 2026 (Pacific/Auckland).

The source snapshot at commit `d6e2ef4de7efeb735402044a845b48704fc51359`
passed the [CPU verification workflow](https://github.com/selena-000/Realtime/actions/runs/43496906349):

| Python | PyTorch CPU | NumPy | pandas | Result |
| --- | --- | --- | --- | --- |
| 3.10 | 2.14.1+cpu | 1.26.4 | 2.3.3 | 19 passed, none skipped |
| 3.12 | 2.15.0+cpu | 1.26.4 | 2.3.3 | 19 passed, none skipped |

Both jobs also passed the 33-file source allowlist check, Python 3.9 syntax
check, and wheel/source-distribution builds. Tests include synthetic CPU
training, explicit resume, evaluation, unlabeled prediction, causal label
independence, future-feature independence, dialogue-state reset, feature
validation and checkpoint round trips.

The original model constructor and trajectory encoder were compared with the
original `final` source and are structurally identical. The model forward is
also structurally identical except that optional text embeddings now fail
explicitly when omitted instead of silently substituting zeros.

The local server did not have PyTorch and could not download it, so its local
test run passed the two NumPy metric tests and skipped the four PyTorch test
modules. The complete runtime suite above ran in GitHub Actions with real
PyTorch, without those skips.

Full MELD accuracy, CUDA execution, raw-feature extraction and strict loading
of the original large checkpoint files have not been rerun. The original-head
comparison script is supplied for use on a machine with those files and
PyTorch. This cleanup makes no new paper-score reproduction claim.
