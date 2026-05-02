# What Is This Project? (Plain English)

## The Big Picture

Every disease is caused or made worse by proteins in your body behaving in ways they shouldn't. One way to treat a disease is to find a small molecule — a drug — that can physically attach to a problem protein and stop it from misbehaving.

Finding that drug is like finding a key that fits a very specific lock. The lock is the protein. The key is the drug molecule. There are billions of possible "keys" in the world of chemistry — we need to figure out which ones actually fit.

This project is about using machine learning to do that search faster and smarter.

---

## The Target: WDR91

The protein we're trying to find a drug for is called **WDR91**. It plays a role in how cells process and recycle materials internally (a process called endosomal signalling). When this goes wrong, it can contribute to disease.

WDR91 has a particular shape — called a **WD40-repeat domain** — that makes it hard to target with traditional drug discovery methods. So we used data-driven approaches instead.

---

## The Data: DNA-Encoded Library (DEL) Screening

Instead of testing one drug at a time (which would take forever), scientists use a technique called **DNA-Encoded Library (DEL) screening**:

1. You take millions of tiny molecules, each with a unique DNA barcode attached
2. You mix them all with the target protein (WDR91)
3. The molecules that stick to the protein get "selected"
4. You sequence the DNA barcodes to figure out which molecules stuck

The result is a dataset of ~375,000 molecules, each labelled as a **hit** (stuck to WDR91) or a **non-hit** (didn't stick). Only about 1 in 13 molecules is a hit — so the data is heavily imbalanced.

---

## What We're Trying to Do

We have a separate library of **339,258 drug-like commercial molecules** that have never been tested against WDR91. Our job is to **rank them** — predict which ones are most likely to bind to WDR91 — so that scientists can pick the top candidates for lab testing.

This is called **virtual screening**: using a computer model to narrow down a huge list before running expensive lab experiments.

The main metric we're judged on is **Enrichment Factor at 1% (EF@1%)**:
> Out of the top 1% of our ranked list, how many times more hits do we find compared to picking randomly?
> An EF of 7.98 means we're finding hits ~8× better than random.

---

## How We Represent Molecules

Computers can't directly understand a molecule's shape. So we convert each molecule into a **fingerprint** — a long string of 0s and 1s where each bit represents whether a particular structural feature is present.

We used three types of fingerprints combined:
- **ECFP6** (2,048 bits) — captures circular neighborhoods around each atom
- **MACCS** (167 bits) — checks for presence of specific pharmacophore features
- **RDK** (2,048 bits) — captures linear substructures along the molecule

Together: a **4,263-bit vector** representing each molecule.

---

## Our Three Approaches

### 1. XGBoost Classifier (Primary Model)
A machine learning model trained on the 375k DEL screen compounds. It learns which fingerprint patterns are associated with hitting WDR91 and uses that to score the 339k commercial molecules.

- Handles the 12:1 imbalance by upweighting hits during training
- Achieves **EF@1% = 7.98×** on a held-out test set

### 2. Tanimoto Similarity + Ensemble
We also have 177 molecules that are already confirmed WDR91 binders (from published research). The idea: molecules that look structurally similar to known binders are more likely to also bind.

- For each commercial molecule, we compute its maximum **Tanimoto similarity** to the 177 known binders (Tanimoto = a measure of structural overlap, 0 = nothing in common, 1 = identical)
- This score is almost uncorrelated with the XGBoost score (r = −0.097) — meaning they're capturing completely different information
- We combine both scores into an **ensemble** (60% XGBoost + 40% Tanimoto)
- Validated against confirmed binders in the commercial library: **EF = 12.3×** in the top 200

### 3. Graph Neural Network (GNN) via Pseudo-Labelling
A more advanced approach: instead of fingerprints, we represent each molecule as a **graph** — atoms are nodes, bonds are edges. A Graph Neural Network learns directly from this structure.

The problem: our training data has no SMILES (the text format needed to build molecular graphs). So we used a workaround called **pseudo-labelling**:
- Use XGBoost scores to assign temporary labels to the 339k commercial molecules
- Top 5% → pseudo-positive · Bottom 60% → pseudo-negative · Middle 35% → discarded
- Train the GNN on these pseudo-labels

The GNN finds candidates that barely overlap with the other methods (only 9.4% shared with the ensemble) — it's exploring a different region of chemical space. This is high-risk but potentially high-reward: novel chemotypes not captured by fingerprint models.

---

## Key Results

| Method | What it's good at | EF@1% |
|--------|-------------------|-------|
| XGBoost | DEL binding patterns | 7.98× |
| XGBoost + Tanimoto Ensemble | Known scaffold similarity + DEL patterns | 6.52× (top 1%), 12.29× (top 200) |
| GNN | Novel structural space | Below random on known actives — explores new chemotypes |

---

## The Core Challenge: Domain Shift

The training data (DEL molecules) and the test data (commercial drug-like molecules) are chemically very different:
- DEL molecules are small, built from 3 building blocks, more polar
- Commercial molecules are larger, more drug-like, from diverse chemical families

A model trained on DEL data may not generalise well to commercial molecules. This is called **domain shift** and is the central limitation of our approach. The Tanimoto and GNN components were specifically designed to bridge this gap.

---

## Why This Matters

Finding a molecule that binds to WDR91 is the first step toward potentially developing a treatment for diseases linked to endosomal dysfunction. Virtual screening with machine learning lets us search millions of molecules in minutes instead of years — and our ensemble approach is validated at **12× better than random** at identifying true binders.
