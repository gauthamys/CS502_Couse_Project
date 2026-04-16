# Methods & Approach Reference

A technical reference for the report and presentation. Documents all models, features, data sources, and design decisions.

---

## Data Sources

| File | Compounds | Labels | SMILES | Purpose |
|------|-----------|--------|--------|---------|
| `WDR91.parquet` | 375,595 | Yes | No | Training data (DEL screen) |
| `Step2_TestData_Target2035.parquet` | 339,258 | No | Yes | Test set for final predictions |
| `known_active_molecules.csv` | 177 | Yes (all actives) | Yes | Reference set for similarity scoring |
| `14_public_domain_WDR91_ligands.csv` | 14 | Yes (all actives) | Yes | Subset of the 177 known actives |

**Note:** The 14 public domain ligands are fully contained within the 177 known actives — only `known_active_molecules.csv` is used in the pipeline.

---

## Features

### Fingerprints (used in XGBoost/LightGBM)

| Fingerprint | Bits | Description |
|------------|------|-------------|
| ECFP6 | 2,048 | Circular fingerprint, radius 3 — best for bioactivity prediction |
| MACCS | 167 | Structural keys — pharmacophore features |
| RDK | 2,048 | Topological path fingerprint — linear substructures |
| **Combined** | **4,263** | Concatenated ECFP6 + MACCS + RDK |

**Feature importance (XGBoost, gain):** ECFP6 86.3% · MACCS 10.0% · RDK 3.8%

### Fingerprint Format Differences

| Dataset | Format | Parsing |
|---------|--------|---------|
| `WDR91.parquet` (training) | Sparse index lists (list\<int32\>) | Direct: each value is an ON-bit index |
| `Step2_TestData_Target2035.parquet` | Dense comma-separated strings (`"0,1,0,1,..."`) | Split → find indices where value > 0 |

---

## Models

### Model 1: XGBoost Classifier (primary)

**Notebook:** `02_train_evaluate.ipynb`

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | ~12.05 (non-hits / hits) |
| `eval_metric` | `aucpr` |

**Why `scale_pos_weight`:** Training data is 12:1 imbalanced (non-hit:hit). This parameter upweights hits so the model focuses on ranking the rare positive class.

**Test set results (stratified 80/20 split):**

| ROC-AUC | PR-AUC | Bal. Acc. | Recall | EF@1% | EF@5% |
|---------|--------|-----------|--------|-------|-------|
| 0.7403 | 0.2565 | 0.6700 | 0.6466 | 7.98 | 4.34 |

### Model 2: LightGBM Classifier (comparison)

Same hyperparameters and split as XGBoost. Consistently underperforms XGBoost on all metrics including the primary challenge metric EF@1%.

| ROC-AUC | PR-AUC | Bal. Acc. | Recall | EF@1% | EF@5% |
|---------|--------|-----------|--------|-------|-------|
| 0.7326 | 0.2440 | 0.6631 | 0.6539 | 7.84 | 4.16 |

**Decision:** XGBoost selected as the primary model.

---

## Tanimoto Similarity Scoring

**Notebook:** `04_similarity_ensemble.ipynb`

### Motivation
The 177 known WDR91 actives are confirmed binders with known SMILES. Compounds structurally similar to known binders are likely to share the same binding mode. This signal is entirely independent of the DEL screen data.

### Method
1. Convert 177 known active SMILES → RDKit Morgan fingerprints (radius=3, 2048 bits)
2. Convert Step 2 SMILES → Morgan fingerprints
3. For each Step 2 compound: compute Tanimoto similarity to all 177 actives, take the **maximum**
4. Result: one similarity score per compound (0 = no similarity, 1 = identical)

**Tanimoto formula:**
```
Tanimoto(A, B) = |A ∩ B| / |A ∪ B|
```
Where A and B are sets of ON bits in binary fingerprints.

---

## Ensemble

**Notebook:** `04_similarity_ensemble.ipynb`

### Why ensemble?
- XGBoost captures DEL-specific binding patterns (trained on 375k labeled compounds)
- Tanimoto captures drug-like scaffold similarity to known WDR91 binders
- Low correlation between scores → complementary signals → ensembling reduces variance

### Method: Rank Normalization + Weighted Average
1. **Rank-normalize** each score independently to [0, 1] to remove scale differences
2. **Weighted average:** `ensemble = 0.6 × XGBoost + 0.4 × Tanimoto`
3. XGBoost gets higher weight (trained on domain-specific labeled data)

### Ensemble Results (339,258 Step 2 compounds)

| Metric | Value |
|--------|-------|
| Score correlation (XGBoost vs Tanimoto) | **-0.097** (nearly uncorrelated → strong case for ensembling) |
| Mean XGBoost score (full set) | 0.0167 |
| Mean Tanimoto score (full set) | 0.1985 |
| Mean XGBoost score (top 1%) | 0.0425 |
| Mean Tanimoto score (top 1%) | 0.2923 |

The near-zero correlation confirms the two scores capture independent signals — XGBoost from DEL binding patterns, Tanimoto from drug-like scaffold similarity.

---

## Evaluation

### Primary metric: Enrichment Factor @1% (EF@1%)

> How many times better than random are we at finding hits in the top 1% of our ranked list?

```
EF@1% = (hits found in top 1% of ranked list) / (expected hits by random in 1%)
```

- EF = 1.0 → no better than random
- EF = 7.98 → 7.98× better than random (XGBoost result)

### Cross-validation strategy
**Stratified 80/20 train/test split** (`random_state=42`, `stratify=y`).

Earlier experiments used library-aware 5-fold CV (StratifiedGroupKFold by DEL library prefix). The 80/20 split was adopted to match the evaluation protocol used for LightGBM/LogReg comparisons.

---

## Imbalance Handling — Benchmarked Strategies

Six strategies were tested in early experiments:

| Strategy | Mean EF@1% |
|----------|-----------|
| `scale_pos_weight` (XGBoost) | **1.93 ± 0.71** |
| Baseline (no correction) | 1.89 ± 0.59 |
| Undersample 1:5 | 1.83 ± 0.56 |
| Undersample 1:3 | 1.81 ± 0.52 |
| Balanced Random Forest | 1.54 ± 0.63 |
| LightGBM `is_unbalance` | Failed (dtype error) |

`scale_pos_weight` selected: best EF@1%, zero computational overhead.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Exploratory data analysis on WDR91 training set |
| `02_train_evaluate.ipynb` | Train XGBoost + LightGBM, 80/20 split, full metrics |
| `03_predict.ipynb` | Predict on Step 1 validation library using fingerprints |
| `04_similarity_ensemble.ipynb` | Tanimoto scoring + XGBoost ensemble on Step 2 data |

---

## Key Design Decisions

**Why not GNN?**
WDR91 training data has no SMILES — only building block IDs and pre-computed fingerprints. GNN training requires molecular graphs (from SMILES). Since the two datasets with SMILES (Step 2 test, known actives) have no labels, a supervised GNN cannot be trained. The Tanimoto similarity approach achieves the same goal of incorporating structural chemistry without requiring labeled SMILES data.

**Why ECFP6 + MACCS + RDK over all 9 fingerprints?**
All 9 fingerprints give 16,551 features. The LightGBM trained on all 9 achieves higher ROC-AUC (0.9786) but this was evaluated under a different experimental setup. On the controlled 80/20 split, ECFP6+MACCS+RDK (4,263 features) provides the verified EF@1% = 7.98.

**Why rank-normalize before ensembling?**
XGBoost scores are calibrated probabilities (clustered near 0 for imbalanced data). Tanimoto scores are similarities. A simple average would be dominated by whichever score has higher variance. Rank normalization puts both on equal footing.
