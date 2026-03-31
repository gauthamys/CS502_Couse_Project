# DREAM Target 2035 — WDR91 Drug Discovery Challenge

## What are we trying to do?

We are trying to find **drug-like molecules** that bind to a protein called **WDR91**.

WDR91 is a human protein with no known drug — part of the *Target 2035* initiative which aims to have chemical tools for every human protein by 2035. Finding molecules that interact with it is the first step toward developing a drug.

---

## How the challenge works (2 phases)

### Phase 1 — Retrospective (can we find known hits?)

We are given a **DEL screen dataset**: 375,595 molecules that were already chemically tested against WDR91 using a technology called DNA-Encoded Libraries (DEL). We know which ones "hit" (stuck to WDR91) and which didn't.

**Task:** Train a machine learning model on this data, then use it to rank a *separate* library of ~370,000 diverse molecules. Hidden inside that library are **145 confirmed hits**. If our model ranks those 145 near the top, we pass Phase 1.

### Phase 2 — Prospective (find new drugs)

Use the validated model to screen a commercial library of **4.4 million compounds** that have never been tested. The top predictions get physically synthesized and tested at the Structural Genomics Consortium (SGC) in Toronto.

**Prize:** $5,000 + opportunity to publish the discovered compounds.

---

## What is a DEL?

A **DNA-Encoded Library (DEL)** is a way to screen millions of molecules at once cheaply.

Each molecule is attached to a unique DNA barcode. All molecules are mixed with the target protein. Those that bind stick around; those that don't get washed away. Sequencing the remaining DNA tells us which molecules bound — higher read count = stronger binder.

In our dataset:
- `TARGET_VALUE` = number of times a molecule's barcode was sequenced (higher = more binding)
- `LABEL = 1` → compound had read count > 0 (hit), `LABEL = 0` → no reads (non-hit)
- Each molecule is made from 3 **building blocks** (BB1, BB2, BB3) combined together

---

## What does the data look like?

| Column | What it is |
|--------|-----------|
| `COMPOUND_ID` | Unique identifier for each molecule |
| `LIBRARY_ID` | Which DEL sub-library it comes from (39 libraries total) |
| `BB1_ID / BB2_ID / BB3_ID` | The 3 building blocks the molecule is made of |
| `TARGET_VALUE` | Raw binding count from the screen |
| `LABEL` | 1 = hit, 0 = non-hit |
| `MW` | Molecular weight (size of molecule) |
| `ALOGP` | Lipophilicity (how fat-soluble the molecule is) |
| `ECFP6`, `ECFP4`, etc. | **Fingerprints** — binary encodings of molecular structure |

**Key numbers:**
- 375,595 total compounds
- 28,778 hits (7.66%) — imbalanced but manageable
- Fingerprints are stored as lists of "on-bit" indices (sparse format)

---

## What are fingerprints?

A fingerprint is a way to represent a molecule as a list of 1s and 0s (a bit vector).

Think of it as a checklist: "does this molecule have a benzene ring? a carbonyl group? a nitrogen here?" Each yes/no is one bit. We use 2048-bit vectors.

Different fingerprints capture different aspects of a molecule:

| Fingerprint | What it captures | Bits |
|------------|-----------------|------|
| ECFP6 | Circular chemical environment (most common, best for activity) | 2048 |
| ECFP4 | Same but shorter radius | 2048 |
| FCFP6 | Like ECFP but emphasizes pharmacophore features | 2048 |
| MACCS | 166 standard structural keys | 167 |
| RDK | Topological paths | 2048 |
| AVALON | Commercial, balanced | 512 |

---

## What are we maximising?

### Primary: Enrichment Factor @1% (EF@1%)

> *"How many times better than random are we at finding hits in our top 1% predictions?"*

Formula:
```
EF@1% = (hits found in top 1% of ranked list) / (expected hits by random chance in 1%)
```

- **EF = 1.0** → no better than random
- **EF = 10.0** → 10× better than random
- For the 145-hit validation set: random would find ~1.45 hits in the top 1% (3,700 compounds); EF=10 means finding ~14-15

### Secondary metrics

| Metric | What it means |
|--------|--------------|
| **AUPRC** | Area under precision-recall curve — overall ranking quality, good for imbalanced data |
| **AUROC** | Area under ROC curve — less sensitive to imbalance than accuracy |
| **EF@5%** | Same as EF@1% but looking at top 5% of ranked list |

---

## The modeling pipeline

```
WDR91.parquet
     │
     ├── Scalar columns (MW, ALOGP, BBs, LABEL)  ──→  EDA
     │
     └── Fingerprints (ECFP6, etc.)  ──→  Sparse CSR matrix
                                                │
                                         XGBoost classifier
                                         (scale_pos_weight=12)
                                                │
                                    Library-aware 5-fold CV
                                    (grouped by DEL library)
                                                │
                                         Rank by probability
                                                │
                                    EF@1%, AUPRC, AUROC
```

**Why library-aware CV?** We split train/test by DEL library, not randomly. This means the model is tested on compounds from libraries it has never seen — much closer to the real challenge where we predict on a completely different chemical library.

---

## Notebooks (run in order)

| Notebook | Purpose |
|----------|---------|
| [colab/01_eda.ipynb](notebooks/colab/01_eda.ipynb) | Explore data: distributions, hit rates by library and building block |
| [colab/02_train_evaluate.ipynb](notebooks/colab/02_train_evaluate.ipynb) | Train XGBoost, 5-fold CV, evaluate EF@1%, save model |
| [colab/03_predict.ipynb](notebooks/colab/03_predict.ipynb) | Load saved model, rank new compound library, export predictions |

**Setup for Colab:**
1. Upload `WDR91.parquet` to `My Drive/CS502/data/`
2. Run notebooks in order — each cell has comments explaining what it does
3. Model is automatically saved to `My Drive/CS502/models/` after training

---

## Why is class imbalance not the main problem?

We tested 6 imbalance strategies (XGBoost default, scale_pos_weight, undersampling 1:3, undersampling 1:5, LightGBM, Random Forest balanced). The results were nearly identical — all within one standard deviation of each other.

**The bigger challenges are:**
1. **Domain shift** — training on DEL compounds (3-BB combinatorial), predicting on commercial drug-like compounds (very different chemical space)
2. **Building block signal** — some BB combinations are enriched across the whole library; extracting BB-level features could improve results significantly
3. **Fingerprint choice** — combining multiple fingerprints or using learned representations (graph neural nets) may outperform single ECFP6
