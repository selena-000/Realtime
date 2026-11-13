# Source provenance

This code was extracted from `selena-000/Realtime-MERC`, branch `final`,
commit `537d832022971fd142c6253872de02b72e143366`.

The model is derived from `BaselineMELDModel` in `train_baseline_meld.py`.
Parameter names and the classification architecture are retained for checkpoint
compatibility. Feature adapters originate in
`minigpt4/datasets/datasets/meld.py`. The inspection command originates in
`inspect_mm_align_features.py`. Shared metric logic originates in the training
and evaluation scripts.

The source repository was built on Emotion-LLaMA and MiniGPT-4/MiniGPT-v2.
Their existing BSD notices are retained in `LICENSE.md` (Zebang Cheng, 2024)
and `LICENSE_MiniGPT4.md` (Deyao Zhu, 2023). The upstream MiniGPT model stack
is not bundled in this package.

Upstream projects:

- Emotion-LLaMA: https://github.com/ZebangCheng/Emotion-LLaMA
- MiniGPT-4: https://github.com/Vision-CAIR/MiniGPT-4
- MELD: https://github.com/declare-lab/MELD

MELD data, extracted features and model weights are distributed separately.
Their original access terms and licenses apply. This repository does not grant
rights to redistribute those artifacts.
