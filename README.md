# DREAM Target 2035 Drug Discovery Challenge

## Overview

This project tackles the **DREAM Target 2035 Challenge** — building ML models on DNA-encoded library (DEL) data to identify drug-like molecules ("hits") that bind the target protein **WDR91**.

Challenge page: https://dreamchallenges.org/target-35-drug-discovery-challenge/

## Challenge Description

**Phase 1 — Retrospective Validation:**
- Train ML models on WDR91 DEL screen data
- Retrieve 145 confirmed hits hidden in ~370,000 diverse molecules
- Used to validate model quality before prospective prediction

**Phase 2 — Prospective Prediction:**
- Use validated models to predict hits from a 4.4M compound commercial library
- Top predictions are experimentally tested at the Structural Genomics Consortium (SGC), University of Toronto
- Winners receive $5,000 cash prize + publication opportunity

## Methodology

DEL-ML approach: train on DNA-Encoded Library screening data, which provides count-based enrichment signals for each molecule. Molecules with high enrichment relative to controls are potential binders.

## Project Structure

```
.
├── data/
│   ├── raw/          # Original DEL screen data (from Synapse)
│   ├── processed/    # Featurized molecules, splits
│   └── external/     # Commercial library (~4.4M compounds)
├── notebooks/        # Exploratory analysis and experiments
├── src/
│   ├── features/     # Molecular featurization (fingerprints, GNN, etc.)
│   ├── models/       # Model architectures and training
│   ├── evaluation/   # Metrics, enrichment factor, hit retrieval
│   └── utils/        # Data loading, SMILES processing, helpers
├── configs/          # Hyperparameter configs (YAML)
├── outputs/
│   ├── predictions/  # Final hit predictions (CSV)
│   ├── figures/      # Plots and visualizations
│   └── checkpoints/  # Saved model weights
└── requirements.txt
```

## Setup

```bash
conda create -n del-ml python=3.10
conda activate del-ml
pip install -r requirements.txt
```

## Key Metrics

- **Enrichment Factor (EF)**: Primary metric — fraction of true hits retrieved in top-k predictions
- **AUROC**: Area under ROC curve for binary hit classification
- **Hit Rate**: Precision in top-N predictions
