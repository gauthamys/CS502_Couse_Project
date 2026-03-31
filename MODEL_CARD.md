# WDR91 Model Card — DREAM Target 2035 Challenge

## Overview

| Property | Value |
|----------|-------|
| Model type | XGBoost gradient-boosted decision trees |
| Task | Binary classification (hit / non-hit) |
| Primary metric | Enrichment Factor @1% (EF@1%) |
| Training data | 375,595 DEL compounds screened against WDR91 |
| Prediction target | 339,258 compounds in the Step 1 validation library |

---

## Features

### Fingerprints (4,263 bits total)

Each compound is represented as a single binary vector formed by concatenating three fingerprints:

| Fingerprint | Bits | Offset in vector | What it captures |
|------------|------|-----------------|-----------------|
| **ECFP6** | 2,048 | 0–2,047 | Circular chemical environment up to radius 3 — best for predicting bioactivity |
| **MACCS** | 167 | 2,048–2,214 | 166 standard structural keys — pharmacophore features (rings, bonds, functional groups) |
| **RDK** | 2,048 | 2,215–4,262 | Topological path fingerprint — linear substructure patterns, complements ECFP |

**Why multiple fingerprints?** Each fingerprint captures a different aspect of molecular structure. ECFP6 is the best single predictor for bioactivity, but MACCS and RDK encode complementary signals (e.g. MACCS captures pharmacophore features that ECFP may represent as different bits across similar molecules). Concatenating all three gives the model more ways to distinguish hits from non-hits.

**What a fingerprint is:** A binary vector where each bit answers "does this molecule have this structural feature?" For example, bit 142 of ECFP6 might represent "has an aromatic ring next to a carbonyl group." The full list of 2,048 bits covers a broad range of molecular patterns.

### Excluded features

- **MW, ALogP** — physicochemical properties available in both datasets. EDA showed near-identical distributions for hits vs non-hits, so these are not included.
- **BB1/BB2/BB3** — building block IDs from the DEL library. Not present in the validation library, so unusable for prediction.

---

## Model Architecture

**XGBoost classifier** (`xgboost.XGBClassifier`)

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| `n_estimators` | 300 | Enough trees to converge; validated not to overfit |
| `learning_rate` | 0.05 | Slow learning rate for better generalization |
| `max_depth` | 6 | Deep enough to capture feature interactions |
| `subsample` | 0.8 | Row subsampling — reduces overfitting |
| `colsample_bytree` | 0.8 | Column subsampling — reduces overfitting |
| `scale_pos_weight` | ~12.05 | Corrects 12:1 class imbalance (non-hit:hit ratio) |
| `eval_metric` | `aucpr` | Optimizes area under precision-recall curve — better for imbalanced data than AUC-ROC |

**Why XGBoost?** Tested 6 approaches (baseline, scale_pos_weight, undersampling 1:3, undersampling 1:5, LightGBM, Random Forest). All performed similarly on EF@1%. XGBoost with `scale_pos_weight` was chosen because it's zero computational cost and well-understood.

**Why `scale_pos_weight`?** The dataset has ~12 non-hits for every hit. Without correction, the model would learn to mostly predict "non-hit" and still get high accuracy. Setting `scale_pos_weight=12.05` tells XGBoost to penalize misclassifying hits 12× more, so it focuses on correctly ranking the rare positive class.

---

## Training Procedure

### Cross-validation strategy: Library-Aware 5-Fold CV

```
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
Groups = DEL library prefix (first 3 characters of LIBRARY_ID)
```

**Why library-aware?** If we split train/test randomly, compounds from the same DEL library end up in both splits. Molecules within a library share building blocks and chemical scaffolds — the model would effectively "memorize" patterns from that library. By holding out entire libraries, each fold tests the model on chemistry it has never seen, mimicking the real challenge where predictions are made on a completely different compound collection.

**What gets evaluated:** Out-of-fold (OOF) predictions — each compound is predicted exactly once by a model that was never trained on it.

### Final model

After cross-validation confirms performance, a final model is trained on **all 375,595 compounds**. This model has seen more data and is used for actual predictions.

---

## Metrics

### Primary: Enrichment Factor @1% (EF@1%)

> *How many times better than random are we at finding hits in the top 1% of our ranked list?*

```
EF@1% = (hits found in top 1%) / (expected hits by random in 1%)
       = (hits in top N) / (total_hits × 0.01)
```

- **EF = 1.0** → no better than random
- **EF = 10.0** → 10× better than random; if there are 145 hits in 339k compounds, random finds ~1.45 in the top 3,393; EF=10 means finding ~14–15

### Secondary metrics

| Metric | What it measures |
|--------|----------------|
| **EF@5%** | Same as EF@1% but looking at top 5% of ranked list |
| **AUPRC** | Area under precision-recall curve — overall ranking quality across all thresholds; better than AUROC for imbalanced data |
| **AUROC** | Area under ROC curve — probability that a random hit is ranked above a random non-hit |

---

## Known Limitations

### Domain shift (main challenge)
The model is trained on DEL chemistry — compounds made from 3 combinatorial building blocks, typically smaller and more polar than typical drugs. The validation library contains drug-like commercial compounds from a completely different chemical space. The model may not generalize well to structural motifs absent from the training data.

### No scaffold diversity enforcement
The top predictions may cluster around similar chemical series. In practice, diverse predictions (covering multiple chemical scaffolds) are preferred for experimental screening.

### No building block signal
Some DEL building blocks are systematically enriched regardless of the full molecule. This information is available during training but cannot be applied at prediction time because the validation library does not contain BB1/BB2/BB3 identifiers.

---

## File Outputs

| File | Description |
|------|-------------|
| `models/xgb_multifp.json` | Trained XGBoost model (load with `model.load_model()`) |
| `outputs/predictions_ranked.csv` | All 339,258 compounds ranked by `hit_probability`, columns: `RandomID`, `hit_probability`, `rank` |

---

## Reproducibility

```python
# Reproduce predictions from saved model
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model('models/xgb_multifp.json')
# Load X_val using load_multi_fingerprints_val() from notebook 03
y_prob = model.predict_proba(X_val.astype(np.float32))[:, 1]
```

All random seeds are fixed at `random_state=42`. Results may vary slightly across XGBoost versions.
