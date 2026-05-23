# MARIN Framework

**Multiplex Adaptive Reinforcement Intervention Network for Real-Time Misinformation Containment**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Foundational paper DOI](https://img.shields.io/badge/Foundational%20paper-10.63562%2F2577--8439.1152-green.svg)](https://doi.org/10.63562/2577-8439.1152)

## Overview

MARIN is a novel framework that combines **deep reinforcement learning** with **co-evolving multiplex networks** for adaptive misinformation intervention. Unlike traditional approaches that assume static network topologies, MARIN allows the network structure to evolve based on both misinformation dynamics and intervention actions.

![MARIN Framework Architecture](docs/graphical_abstract.png)

## Key Features

- **Co-Evolving Multiplex Network Model**: Three-layer network (Information Sharing, Social Reinforcement, Private Communication) with dynamic topology
- **GNN-based State Encoding**: Graph Neural Network encoder with cross-layer attention aggregation
- **DDQN Agent**: Double Deep Q-Network for optimal intervention selection
- **Four Intervention Types**:
  1. Truth Injection
  2. Bridge-Node Activation
  3. Echo Chamber Disruption
  4. Cross-Layer Amplification
- **Phase Transition Analysis**: Identifies critical rewiring rate thresholds

## Key Results

| Metric | MARIN vs Static Baseline | MARIN vs Adaptive Baseline |
|--------|-------------------------|---------------------------|
| MRR Improvement (omega = 0.2) | **+47%** | **+21%** |
| MRR Improvement (omega = 0.05) | **+34%** | **+13%** |
| Critical Threshold | omega* ≈ 0.15 | – |

## Installation

```bash
# Clone the repository
git clone https://github.com/sdemojhon/MARIN-Framework.git
cd MARIN-Framework

# Create virtual environment
python -m venv marin_env
source marin_env/bin/activate          # Linux / macOS
# marin_env\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NetworkX 3.2+
- NumPy
- Pandas
- Matplotlib
- PyYAML

## Quick Start

```python
from src.simulation import run_simulation, run_monte_carlo

# Single trajectory
result = run_simulation(seed=1, n_nodes=1000, omega=0.1, n_steps=200)
print(result.summary())

# 100-run Monte Carlo ensemble (paper protocol)
agg = run_monte_carlo(
    seeds=range(1, 101),
    n_nodes=1000,
    omega=0.1,
    n_steps=200,
    output_dir="results/tables/quick_demo",
)
print(f"MRR = {agg['mrr']['mean']:.2f} +/- {agg['mrr']['std']:.2f}")
```

## Reproducing the Paper Tables

```bash
# Smoke-test the entire pipeline in < 1 minute
python experiments/run_baseline.py --quick

# Table 3 (Section 5.2) — MRR and IE across strategies
python experiments/run_baseline.py --omega 0.05 --output results/tables/table3_omega05.json
python experiments/run_baseline.py --omega 0.20 --output results/tables/table3_omega20.json

# Table 4 (Section 5.3) — ablation study
python experiments/run_ablation.py --output results/tables/table4.json

# Table 5 (Section 5.5) — scalability across N = 1k / 2k / 5k
python experiments/run_scalability.py --output results/tables/table5.json
```

## Training a Fresh Agent

```bash
# Quick demo checkpoint (≈5 min on a laptop CPU)
python experiments/train.py --quick

# Full paper checkpoint for N = 1,000 (≈6 h on a single GPU)
python experiments/train.py --n_nodes 1000 --episodes 2000 --output models/marin_n1000.pt
```

## Repository Structure

```
MARIN-Framework/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── src/
│   ├── __init__.py
│   ├── marin_network.py            # Co-evolving multiplex network (Eqs. 1-3)
│   ├── marin_agent.py              # DDQN agent + GNN encoder + replay (Eqs. 7, 8, 10)
│   ├── belief_dynamics.py          # Belief update mechanisms (Eqs. 4-6)
│   ├── interventions.py            # Four intervention action types
│   ├── gnn_encoder.py              # Encoder factory (re-export of GNNEncoder)
│   └── simulation.py               # Monte Carlo simulation orchestrator
├── experiments/
│   ├── train.py                    # DDQN training driver (Algorithm 1)
│   ├── run_baseline.py             # Reproduces Table 3
│   ├── run_ablation.py             # Reproduces Table 4
│   └── run_scalability.py          # Reproduces Table 5
├── configs/
│   ├── default_config.yaml         # Default hyperparameters (matches Table 2)
│   └── experiment_configs/         # Per-experiment YAML overrides
├── data/
│   ├── synthetic/
│   │   └── generate_networks.py    # Generates seeded synthetic networks
│   └── README.md                   # Data documentation
├── models/
│   └── README.md                   # How to obtain / load checkpoints
├── results/
│   ├── figures/                    # Auto-populated by experiment scripts
│   ├── tables/                     # Auto-populated by experiment scripts
│   └── README.md
└── docs/
    ├── graphical_abstract.png      # Architecture diagram
    └── equations.md                # Complete mathematical formulation
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 1,000–5,000 | Network size |
| omega | 0.01–0.5 | Co-evolution rate |
| lambda | 0.5–5.0 | Homophily strength |
| mu | 0.1 | Belief update rate |
| kappa | 1.5 | Confirmation bias parameter |
| gamma | 0.99 | Discount factor |
| tau | 1,000 | Target network update frequency |

See `configs/default_config.yaml` for the complete hyperparameter specification.

## Reproducibility

All experiments use the random seeds 1–100, set through `src.simulation.set_seed()`,
which seeds NumPy, Python's `random`, and PyTorch (CPU + CUDA where available).

```python
from src.simulation import set_seed, run_simulation

for seed in range(1, 101):
    set_seed(seed)
    results = run_simulation(seed=seed, n_nodes=1000, omega=0.1)
```

## Mathematical Formulation

### Link Formation (Equation 1)
```
f_l(i, j, t) = γ_l · exp(-λ_l · |b_i(t) - b_j(t)|) + ε_l
```

### Belief Update (Equation 4)
```
b_i(t+1) = (1 - μ) · b_i(t) + μ · [κ · B_i(m, t) + (1 - κ) · S_i(t)]
```

### Reward Function (Equation 9)
```
R(t) = w_1·ΔMR(t) + w_2·ΔBP(t) + w_3·ECD(t) - w_4·IC(t)
```

See `docs/equations.md` for the complete mathematical formulation.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bhadre2026marin,
  title   = {Adaptive Intervention Strategies in Co-Evolving Multiplex Networks:
             A Reinforcement Learning Approach to Real-Time Misinformation Containment},
  author  = {Bhadre, Anjali A. and Ghongade, Harshvardhan P.},
  journal = {Northeast Journal of Complex Systems (NEJCS)},
  year    = {2026},
  note    = {Forthcoming}
}
```

## Related / Foundational Work

This work builds upon our prior, peer-reviewed research:

> Ghongade, H. P., Bhadre, A. A., Agarwal, S., Pawar, H. U. and Rane, H. S. (2026).
> "Emergent Dynamics in Multiplex Social Networks: Agent-Based Modeling of Information
> Diffusion for Misinformation Control."
> *Northeast Journal of Complex Systems*, 8(1), Article 12.
> [DOI: 10.63562/2577-8439.1152](https://doi.org/10.63562/2577-8439.1152) ·
> [open-access at NEJCS](https://orb.binghamton.edu/nejcs/vol8/iss1/12)

```bibtex
@article{ghongade2026multiplex,
  title    = {Emergent Dynamics in Multiplex Social Networks: Agent-Based Modeling
              of Information Diffusion for Misinformation Control},
  author   = {Ghongade, Harshvardhan P. and Bhadre, Anjali A. and Agarwal, Shivani
              and Pawar, Harjitkumar U. and Rane, Harshal S.},
  journal  = {Northeast Journal of Complex Systems},
  volume   = {8},
  number   = {1},
  pages    = {Article 12},
  year     = {2026},
  doi      = {10.63562/2577-8439.1152},
  url      = {https://orb.binghamton.edu/nejcs/vol8/iss1/12}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

- **Anjali A. Bhadre** (Corresponding Author) — anjali.bhadre@raisoni.net
- **Harshvardhan P. Ghongade** — ghongade@gmail.com

## Acknowledgments

- G.H. Raisoni College of Engineering and Management, Pune
- Brahma Valley College of Engineering and Research Institute (SPPU), Nashik
- NERCCS 2026 Conference Committee
