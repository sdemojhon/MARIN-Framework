# Results

This folder is populated automatically when the experiment scripts are run.

## Subfolders

- `figures/` – generated figures (Figure 2 training curves, Figure 3 phase transition, Figure 4 scalability + interventions). Produced by `scripts/make_figures.py`.
- `tables/` – Monte Carlo aggregated tables (Tables 3, 4, 5 of the manuscript). Produced by the scripts in `experiments/`.

## Reproduction commands

```bash
# Table 3 (Section 5.2): MRR vs baselines, omega = 0.05 and omega = 0.2
python experiments/run_baseline.py --omega 0.05 --output results/tables/table3_omega05.json
python experiments/run_baseline.py --omega 0.2  --output results/tables/table3_omega20.json

# Table 4 (Section 5.3): Ablation study, N = 1,000, omega = 0.1
python experiments/run_ablation.py --output results/tables/table4.json

# Table 5 (Section 5.5): Scalability sweep across N = 1,000 / 2,000 / 5,000
python experiments/run_scalability.py --output results/tables/table5.json
```

All scripts accept `--quick` for a 5-seed, small-N smoke test that runs in
seconds, so reviewers can confirm the code path before launching a full run.
