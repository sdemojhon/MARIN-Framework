# Data Directory

## Synthetic Data

Synthetic multiplex networks are generated using the following parameters:

- **Network sizes**: N ∈ {1000, 2000, 5000}
- **Initial topology**: Barabási-Albert model (m=5)
- **Clustering coefficients**: 0.3-0.5
- **Random seeds**: 1-100 for reproducibility

## Twitter/X COVID-19 Dataset (Planned)

The empirical validation dataset will include:

- **Source**: Twitter Academic Research API
- **Period**: January-March 2020
- **Size**: ≥50,000 tweets
- **Ethics approval**: BVCOERI-2025-047

### Network Layers
1. **Retweet cascades** (Layer 1)
2. **Reply threads** (Layer 2)
3. **Mutual mention patterns** (Layer 3)

### Temporal Resolution
- 6-hour snapshots

## Data Availability

Due to Twitter's Terms of Service, the raw dataset cannot be directly shared.
Instead, we provide:

1. Network statistics and aggregated features
2. Anonymized node identifiers
3. Scripts to reconstruct the network from Twitter API

Contact the authors for data access requests.
