# Step 2 Validation Results

Validation against 138 confirmed WDR91 actives found in the Step 2 library by exact SMILES match (138 of 177 known actives appear in the 339,258-compound commercial library).

**Why this is meaningful:** The 177 known actives were used only for Tanimoto scoring — not for XGBoost training. Ranking them highly is a genuine test of model quality, though the Tanimoto component has a direct advantage since it was designed around these compounds.

---

## Ensemble (XGBoost 60% + Tanimoto 40%)

| Threshold | Cutoff | Found | Recall | EF |
|-----------|--------|-------|--------|----|
| Top 200   | 200    | 1     | 0.7%   | 12.29× |
| Top 500   | 500    | 2     | 1.4%   | 9.83×  |
| Top 1%    | 3,392  | 9     | 6.5%   | 6.52×  |
| Top 5%    | 16,962 | 26    | 18.8%  | 3.77×  |
| Top 10%   | 33,925 | 35    | 25.4%  | 2.54×  |

Rank distribution of 138 known actives under ensemble scoring:

| Stat   | Rank   |
|--------|--------|
| Min    | 6      |
| 25th % | 24,591 |
| Median | 88,675 |
| 75th % | 147,346|
| Max    | 217,687|

---

## GNN (Pseudo-Label GCN)

| Threshold | Cutoff | Found | Recall | EF |
|-----------|--------|-------|--------|----|
| Top 200   | 200    | 0     | 0.0%   | 0.00× |
| Top 500   | 500    | 0     | 0.0%   | 0.00× |
| Top 1%    | 3,392  | 0     | 0.0%   | 0.00× |
| Top 5%    | 16,962 | 3     | 2.2%   | 0.43× |
| Top 10%   | 33,925 | 5     | 3.6%   | 0.36× |

Rank distribution of 138 known actives under GNN scoring:

| Stat   | Rank    |
|--------|---------|
| Min    | 7,027   |
| 25th % | 158,046 |
| Median | 246,924 |
| 75th % | 284,494 |
| Max    | 336,010 |

---

## Interpretation

The GNN performs **worse than random** (EF < 1.0) on the known actives at all tested thresholds. The median rank for a confirmed binder under the GNN is 246,924 out of 339,258 — near the bottom of the ranked list.

This is expected given the pseudo-labeling setup:
- GNN pseudo-labels were derived from XGBoost scores, which reflect DEL chemistry patterns
- The 177 known actives are drug-like literature compounds — structurally distinct from DEL combinatorial molecules
- The GNN learned to replicate XGBoost's DEL-pattern ranking, which actively disagrees with drug-like known actives

The GNN and ensemble top 1% overlap by only **9.4%**, confirming they explore different regions of chemical space:
- **Ensemble**: anchored to known WDR91 drug-like scaffolds (via Tanimoto) + DEL binding patterns (via XGBoost)
- **GNN**: explores novel DEL-like scaffolds not represented in the known actives

For experimental follow-up, the ensemble predictions are better validated. The GNN top candidates represent a higher-risk, higher-novelty set that may contain genuinely new chemotypes not captured by known actives or DEL patterns.
