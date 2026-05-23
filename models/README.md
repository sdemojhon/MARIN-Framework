# Trained Model Checkpoints

This folder holds DDQN checkpoints produced by `experiments/train.py`.

## Files

| File | Description | How to obtain |
|---|---|---|
| `marin_demo.pt` | Small demonstration checkpoint (N = 80, 30 episodes) for reviewers who want to verify the code path without GPU access. ~MB-scale. | `python experiments/train.py --quick` |
| `marin_n1000.pt` | Full-paper checkpoint for N = 1,000 (2,000 episodes per Algorithm 1). Reproduces Tables 3–5 at the N = 1,000 row. | `python experiments/train.py --n_nodes 1000 --episodes 2000 --output models/marin_n1000.pt` |
| `marin_n5000.pt` | Scalability checkpoint for N = 5,000 (≈3,500 episodes, see Section 5.1 of the paper). | `python experiments/train.py --n_nodes 5000 --episodes 3500 --output models/marin_n5000.pt` |

## Loading a checkpoint

```python
from src.marin_agent import MARINAgent

agent = MARINAgent(node_feature_dim=10, embedding_dim=128,
                   n_intervention_types=4, budget=50)
agent.load("models/marin_n1000.pt")

# Now use agent.select_action(...) in evaluation mode (training=False)
```

## Why are the full-paper checkpoints not committed directly to the repo?

Each full-paper `.pt` file is on the order of tens of MB. Rather than bloat the
git history with binary blobs, we ship:

1. The deterministic training script (`experiments/train.py`)
2. Every hyperparameter (`configs/default_config.yaml`)
3. Every random seed (1–100)

These three together let any reader regenerate the checkpoints bit-for-bit on
their own hardware. For users who only want to run inference, the
`marin_demo.pt` checkpoint (committed) is sufficient to verify the import path
and the inference pipeline without retraining.

If you require the production checkpoints and cannot retrain locally, the
corresponding author (anjali.bhadre@raisoni.net) will share them via Zenodo on
request; a DOI will be added to this README upon deposition.

## Training cost (approximate)

| Setting | Hardware | Wall-clock |
|---|---|---|
| `--quick` (N = 80, 30 ep) | CPU laptop | < 5 min |
| N = 1,000, 2,000 ep | Single NVIDIA T4 GPU | ~6 hours |
| N = 5,000, 3,500 ep | Single NVIDIA A100 GPU | ~24 hours |
