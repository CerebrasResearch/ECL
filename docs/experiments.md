# Experiment Reproduction Guide

This guide describes how to reproduce each of the paper's experiments. All scripts are in the `scripts/` directory and use synthetic model surrogates by default. To run with real genomic models, replace the `SyntheticModel` instances with the corresponding model wrappers (see comments at the top of each script).

All scripts write outputs (figures and tables) to the `outputs/` directory, which is created automatically.

---

## Prerequisites

```bash
pip install -e ".[all]"
```

Run all scripts from the project root:

```bash
cd ECL
python scripts/<script_name>.py
```

---

## Experiment 1: Influence Profiles at Promoter Loci (Figure 1)

**Protocol:** Compute binned influence profiles $\hat{I}(d; r)$ for six model surrogates at promoter-centered loci, using both random substitution and dinucleotide shuffle perturbations. Average over 10 loci with 5 sequences each. Plot on log scale with $\pm 1$ SE shaded bands.

**Script:**

```bash
python scripts/fig01_influence_profiles_promoter.py
```

**Output:** `outputs/fig01_influence_profiles_promoter.{pdf,png}` -- six-panel figure (2x3 grid), one panel per model.

---

## Experiment 2: Influence Profiles at Enhancer Loci (Figure 2)

**Protocol:** Same as Experiment 1 but for enhancer-centered loci. Uses different synthetic decay characteristics to reflect the distinct chromatin context of distal regulatory elements.

**Script:**

```bash
python scripts/fig02_influence_profiles_enhancer.py
```

**Output:** `outputs/fig02_influence_profiles_enhancer.{pdf,png}`

---

## Experiment 3: Cumulative Influence and ECL Estimates (Figure 3)

**Protocol:** Compute $I_{\le l}(r) / I_{\text{tot}}(r)$ for all six models on a single panel. Overlay horizontal lines at $\beta \in \{0.5, 0.8, 0.9, 0.95, 0.99\}$ and vertical lines at the corresponding ECL estimates. Show bootstrap 95% CIs as horizontal error bars.

**Script:**

```bash
python scripts/fig03_cumulative_influence.py
```

**Output:** `outputs/fig03_cumulative_influence.{pdf,png}`

---

## Experiment 4: Directional ECL (Figure 4)

**Protocol:** Compute upstream ($I^-$) and downstream ($I^+$) influence profiles for synthetic surrogates of Enformer and Borzoi. Two-panel figure showing directional asymmetry. Report $\text{ECL}^-_\beta$, $\text{ECL}^+_\beta$, and asymmetry ratio.

**Script:**

```bash
python scripts/fig04_directional_ecl.py
```

**Output:** `outputs/fig04_directional_ecl.{pdf,png}`

---

## Experiment 5: Perturbation Sensitivity Scatter (Figure 5)

**Protocol:** For each of 500 synthetic promoter loci and six models, compute $\text{ECL}_{0.9}$ under both random substitution and dinucleotide shuffle. Scatter plot of $\text{ECL}_{0.9}(\text{shuffle})$ vs $\text{ECL}_{0.9}(\text{substitution})$, colored by model. Diamond markers at per-model means.

**Script:**

```bash
python scripts/fig05_perturbation_scatter.py
```

**Output:** `outputs/fig05_perturbation_scatter.{pdf,png}`

**Note:** This script is computationally intensive (6 models x 500 loci x 2 perturbations). Runtime is approximately 10--30 minutes on CPU with synthetic models.

---

## Experiment 6: Trained vs Random Weights (Figure 6)

**Protocol:** Compare influence profiles of a "trained" synthetic model (structured exponential decay) with a "random" model (near-uniform weights, simulating untrained initialization). Demonstrates that training develops distance-dependent context usage.

**Script:**

```bash
python scripts/fig06_trained_vs_random.py
```

**Output:** `outputs/fig06_trained_vs_random.{pdf,png}`

---

## Experiment 7: ECL Distribution by Locus Class (Figure 7)

**Protocol:** Compute $\text{ECL}_{0.9}$ for six locus classes (promoter, proximal enhancer, distal enhancer, CTCF, intronic, intergenic) across two models. Violin plots showing the distribution.

**Script:**

```bash
python scripts/fig07_locus_class_violin.py
```

**Output:** `outputs/fig07_locus_class_violin.{pdf,png}`

---

## Experiment 8: ECL vs Gene Length (Figure 8)

**Protocol:** Scatter plot of $\text{ECL}_{0.9}$ vs simulated gene length, colored by expression level (low/medium/high). Demonstrates that longer genes tend to have longer effective contexts.

**Script:**

```bash
python scripts/fig08_ecl_vs_gene_length.py
```

**Output:** `outputs/fig08_ecl_vs_gene_length.{pdf,png}`

---

## Experiment 9: Paired Model Comparison (Figure 9)

**Protocol:** Compute paired differences $\text{ECL}^A_{0.9}(r_j) - \text{ECL}^B_{0.9}(r_j)$ at 500 matched loci. Histogram with kernel density overlay, mean $\pm$ 95% CI, and permutation test p-value. Stratification by locus class.

**Script:**

```bash
python scripts/fig09_model_comparison_paired.py
```

**Output:** `outputs/fig09_model_comparison_paired.{pdf,png}`

---

## Experiment 10: Biological Validation at Known Long-Range Loci (Figure 10)

**Protocol:** At three well-characterized long-range regulatory loci (SHH/ZRS, MYC/super-enhancer, SOX9/regulatory desert), plot the full influence profile centered on the target gene TSS. Mark the known enhancer position. Overlay three model surrogates with different decay lengths.

**Script:**

```bash
python scripts/fig10_biological_validation.py
```

**Output:** `outputs/fig10_biological_validation.{pdf,png}`

---

## Additional Figures

### Figure 11: Interaction Influence Heatmap

Heatmap of pairwise interaction influence $I_{\text{int}}(i, j; r)$, revealing synergistic and redundant blocks.

```bash
python scripts/fig11_interaction_heatmap.py
```

### Figure 12: gNIAH Sensitivity

gNIAH sensitivity vs distance for CTCF, GATA, and SP1 motifs across models.

```bash
python scripts/fig12_gniah_sensitivity.py
```

---

## Tables

| Script | Description |
|---|---|
| `tab01_model_taxonomy.py` | Model architecture taxonomy (LaTeX + CSV). |
| `tab02_ecl_estimates.py` | $\text{ECL}_\beta$ estimates with 95% bootstrap CIs across models and locus classes. |
| `tab03_utilization.py` | Context utilization ratio $\text{ECL} / L_{\text{nominal}}$. |
| `tab04_perturbation_sensitivity.py` | Perturbation sensitivity analysis across kernel types. |
| `tab05_multiscale_block.py` | Multi-scale block ECL at different block sizes and $\beta$ values. |
| `tab06_pairwise_comparison.py` | Pairwise model comparison with permutation test results. |
| `tab07_biological_validation.py` | Biological validation at known regulatory loci. |

Run any table script the same way:

```bash
python scripts/tab02_ecl_estimates.py
```

Tables are saved as both CSV and LaTeX in the `outputs/` directory.

---

## Running All Experiments

To reproduce all figures and tables:

```bash
for script in scripts/fig*.py scripts/tab*.py; do
    echo "Running $(basename $script)..."
    python "$script"
done
```

Or use the Makefile target:

```bash
make figures
```

---

## Using Real Genomic Models

Each script contains commented-out import blocks at the top showing how to swap synthetic models for real ones. For example:

```python
# Replace:
models["Enformer"] = SyntheticModel(seq_length=SEQ_LENGTH, decay_length=150.0)

# With:
from ecl.models.enformer import EnformerWrapper
models["Enformer"] = EnformerWrapper(device="cuda")
```

When using real models, also replace the random sequences with real genomic sequences (e.g., from a BED file of promoter coordinates extracted via pysam or pybedtools).
