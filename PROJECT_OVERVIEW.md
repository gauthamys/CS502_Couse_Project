# DREAM Target 2035 — Project Overview

## What is this challenge?

The **DREAM Target 2035 Challenge** is a drug discovery competition. The goal is to use machine learning to find molecules that can "bind" to a specific protein called **WDR91**.

When a molecule binds to a disease-related protein, it can block or change that protein's behaviour — this is the basic idea behind most drugs.

---

## The Protein: WDR91

WDR91 is a human protein being studied as part of the **Target 2035** initiative — a global science effort to find chemical tools for every human protein by 2035. We don't need to know the biology deeply; our job is to find molecules that interact with it.

---

## The Technology: DNA-Encoded Libraries (DEL)

Finding drug molecules used to mean testing one compound at a time in a lab — slow and expensive.

**DEL screening** is a smarter approach:
- Chemists attach tiny DNA "barcodes" to millions of molecules
- All molecules are mixed together and exposed to the target protein (WDR91)
- Molecules that stick to the protein get pulled out
- DNA sequencing reveals which barcodes (= which molecules) were pulled out
- A high count = molecule stuck to the protein = potential **hit**

Think of it like fishing with a million hooks at once, then using DNA to identify which hooks caught something.

### Our training data (`WDR91.parquet`)
- **375,595 molecules** from a DEL screen against WDR91
- Each molecule has a `TARGET_VALUE` — how many times it was "pulled out" during screening
- `LABEL = 1` → **hit** (TARGET_VALUE ≥ 6, pulled out enough times to be meaningful)
- `LABEL = 0` → **non-hit** (TARGET_VALUE = 0, never pulled out)
- **28,778 hits** (7.66%) vs **346,817 non-hits** (92.34%)

---

## The Two-Phase Challenge

### Phase 1 — Retrospective (what we do now)
Train a model on the DEL data above, then use it to **score a separate validation library** of ~370,000 diverse molecules.

Hidden inside that validation library are **145 confirmed hits** — molecules already verified in a lab to bind WDR91.

> **Goal:** Rank the 370k molecules so the 145 true hits appear near the top of the list.

### Phase 2 — Prospective (if Phase 1 succeeds)
Use the validated model to screen a **4.4 million compound commercial library** and nominate the best candidates for real experimental testing at the Structural Genomics Consortium (SGC) in Toronto.

Winners get a **$5,000 prize** and can publish the experimentally validated discoveries.

---

## What Our Model Actually Does

We are training a **binary classifier** — given a molecule, predict whether it is a hit or not.

But more precisely, we care about the **score (probability)** the model assigns, not just the yes/no prediction. We use that score to **rank** all molecules from most likely hit to least likely hit, then look at the very top.

### Input features (what we feed the model)
Each molecule is represented as a **fingerprint** — a fixed-length binary vector encoding which chemical substructures are present.

Pre-computed fingerprints already in the data:

| Fingerprint | What it captures | Size |
|---|---|---|
| ECFP4 / ECFP6 | Circular atom environments (radius 2 or 3) | 2048 bits |
| FCFP4 / FCFP6 | Same as ECFP but using pharmacophoric features | 2048 bits |
| MACCS | 166 standard medicinal chemistry keys | 166 bits |
| RDKit | Topological paths | 2048 bits |
| AVALON | Avalon substructure keys | 512 bits |
| ATOMPAIR | Atom-pair distances | 2048 bits |
| TOPTOR | Topological torsion angles | 2048 bits |

Plus physicochemical properties: **MW** (molecular weight) and **ALogP** (lipophilicity).

---

## What We Are Trying to Maximise

### Primary metric: Enrichment Factor at 1% (EF@1%)

> "If we take the top 1% of our ranked list, how many times more hits do we find than if we picked randomly?"

**Formula:**
```
EF@1% = (hits in top 1%) / (total hits)
         ─────────────────────────────────
              1% of total compounds
```

**Example:**
- 370,000 compounds, 145 true hits
- Random selection of top 1% (3,700 compounds) → expect 1.45 hits on average → EF = 1.0
- Our model puts 30 hits in top 1% → EF = 30/145 ÷ 0.01 = **20.7×**

A perfect model would get EF@1% ≈ **13.8** (all 145 hits in the top 3,700).
Random = **1.0**.
We want to be as far above 1.0 as possible.

### Secondary metrics

| Metric | What it measures |
|---|---|
| **AUPRC** | Area Under Precision-Recall Curve — overall ranking quality across all thresholds |
| **AUROC** | Area Under ROC Curve — overall discrimination ability |
| **EF@5%** | Same as EF@1% but looking at top 5% — less stringent |

---

## The DEL Structure (Why It Matters for Modelling)

Each molecule in the DEL is built from **3 building blocks** (BB1, BB2, BB3) combined by a chemical reaction. There are:
- 2,485 unique BB1s
- 2,319 unique BB2s
- 3,562 unique BB3s

This means some building blocks may be systematically "good" or "bad" for binding WDR91 — independent of the full molecule. This is a strong signal we can exploit beyond fingerprints alone.

---

## The Hard Part: The Domain Gap

We train on DEL molecules → predict on a completely different commercial library.

| | DEL training data | Validation library |
|---|---|---|
| Source | DNA-encoded library screen | Commercial vendor catalogue |
| # molecules | 375,595 | ~370,000 |
| Hit rate | 7.66% | 0.039% (145/370k) |
| Chemical diversity | Built from 3-BB combinations | Far more diverse |

The model must generalise from one type of chemistry to another. This is the main challenge — not the algorithm itself.

---

## Project Structure

```
CS502_Project/
├── data/
│   ├── raw/               # WDR91.parquet (DEL screen data)
│   └── processed/         # Cleaned / featurised versions
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   └── 03_imbalance_comparison.ipynb  # Strategy comparison
├── src/
│   ├── models/
│   │   ├── imbalance_comparison.py    # CV framework + 6 strategies
│   │   └── baseline.py
│   ├── features/          # Fingerprint processing
│   ├── evaluation/        # EF, AUPRC metrics
│   └── utils/
│       └── data_loader.py # Parquet → sparse matrix loader
└── outputs/
    ├── figures/           # Plots
    └── predictions/       # CV results, final rankings
```

---

## Where We Are Now

- [x] Data loaded and understood
- [x] Imbalance strategy comparison run (5-fold CV)
- [ ] Feature engineering (combine fingerprints, add BB-level features)
- [ ] Final model training on full DEL dataset
- [ ] Prospective predictions on validation library

---

## Key Insight So Far

> Imbalance handling barely matters. The bigger wins will come from **better features** (especially building-block-level signals) and handling the **domain shift** between DEL chemistry and the commercial library.
