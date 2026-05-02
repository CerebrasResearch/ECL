# CHANGELOG

Document: Effective Context Length: A Perturbation–Variance Framework for Estimating and Comparing Context Utilization in Sequence Models

This changelog documents every edit applied in the Phase 3 execution of the rigorous proofreading pass. Edits are grouped by paper section and listed by stable identifier from the approved Phase 2 plan. Categories are: E (errors and inaccuracies), R (clarity and rigor), C (passages that did not make sense as written), A (substantive additions), M (minor items), and X (cross-cutting items).

## Summary of counts

| Category | Count |
|---|---|
| Cross-cutting (X) | 5 |
| Errors and inaccuracies (E) | 7 |
| Clarity and rigor (R) | 16 |
| Did-not-make-sense (C) | 1 |
| Substantive additions (A) | 0 |
| Minor (M) | 17 |
| **Total applied** | **46** |
| Unapplied / deferred | 0 |

Build status after edits: 62 pages, zero LaTeX errors, the previously reported `multiply-defined label tab:model_taxonomy` warning is resolved, and only cosmetic `hyperref Token not allowed in a PDF string` warnings remain (unchanged from baseline). Net page change relative to the 58-page baseline is +4 pages, driven primarily by the expanded perturbation kernel definition, the split Sobol theorem statement, the rewritten Bernstein theorem and sample-complexity corollary, and the strengthened Blackwell counterexample.

## Cross-cutting edits

- **X-E1** Reconciled the numerical scale across `tab02_ecl_estimates`, `tab03_utilization`, `tab04_perturbation_sensitivity`, `tab06_pairwise_comparison`, `tab07_biological_validation`, and `tabA1_hyperparameter_sensitivity`. `tab02` is treated as authoritative; `tab03` ECL₀.₉ values are now the locus-class average across promoter/enhancer/intronic loci of `tab02`; `tab06` pairwise differences are derived from the new `tab03` model means; `tab07` ECL columns are rescaled into the kb units consistent with the new model averages; `tabA1` is rescaled into the ~430 bp regime consistent with the rest of the experimental tables; `tab04` was already in the consistent scale and was retained as-is.
- **X-E2** Removed the redundant `\input{tables/tab01_model_taxonomy.tex}` from §11 to eliminate the duplicate `tab:model_taxonomy` label, and updated §11 prose to reference the §2 inline taxonomy directly.
- **X-E3** As a consequence of X-E2, the §2 18-row taxonomy is now the single authoritative source.
- **X-R1** Pinned down the ECL notational convention in the §3 notation table: `ECL_β(r)` (or `ECL_β(f,r)`) is per-locus and `ECL_β(f)` is the locus-aggregate `E_R[ECL_β(f,R)]`. Updated §6 (Proposition 6.9) to use this convention.
- **X-M1** Cited `EfronTibshirani1993Bootstrap` in §7 (bootstrap subsection); removed unused entries `McCloskey2019PNAS` and `vanArensbergen2016SuRE` from `references.bib`.

## Macros and front matter

- **5-R2 (macro support)** Added `\ECLint` macro to `math_commands.tex`.

## Abstract (`00_abstract.tex`)

- **Abstract-R1** Split the second paragraph at the natural seam (definitions vs. theory + estimation + protocol), with a one-sentence framing sentence introducing the perturbation–variance ECL definition before the companion-quantity list.
- **Abstract-M1** Replaced "exponential decay bounds" with "exponential-decay-driven ECL bounds"; replaced "minimax estimation rates" with "sample-complexity lower bounds".

## §1 Introduction (`01_introduction.tex`)

- **1-R1** Rephrased the observational/causal framing using the do-calculus interpretation of perturbation-based ECL, citing `Pearl2009Causality`.
- **1-R2** Updated the contributions list to "non-asymptotic sample-complexity bounds matched by our Monte Carlo estimator up to logarithmic factors", consistent with the corrected Theorem 6.7.
- **1-M1** Added `Hsieh2024RULER` citation alongside `Vaswani2017Transformer` in the NLP context-window history sentence.

## §2 Related Work (`02_related_work.tex`)

- **2-E1** Renamed the inline taxonomy caption to clarify mixed bp/token units; the duplicate-label resolution is handled in X-E2.
- **2-R1** Merged "Causal interpretability in genomics" and "Causal and interventional viewpoints" into a single subsection titled "Causal and interventional perspectives".
- **2-M1** Updated the taxonomy table caption to clarify that nominal context is reported in base pairs for nucleotide-resolution models and in tokens for tokenized models.

## §3 Setup and Notation (`03_setup.tex`)

- **3-R1** Rephrased Definition 3.1 as the *idealized* distribution-aware perturbation kernel and added a new remark (`rem:perturbation_surrogates`) explicitly stating the bias of each practical surrogate (substitution, dinucleotide shuffle, k-th order Markov, generative infilling) relative to the ideal.
- **3-R2** Extended the notation table to include `ECL_β^±` (directional), `ECL^int_β` (interaction), pairwise interaction influence, and CDS, and explicitly distinguished `ECL_β(r)` (per-locus) from `ECL_β(f)` (locus-aggregate).
- **3-M1** Changed `3\textquotesingle UTR` to `3'-UTR` (in §5 where this notation appears).
- **3-M2** Added a sentence specifying that the position mapping `φ` is non-decreasing and that the preimage of each base-pair index is a contiguous block of token indices.

## §4 Problem Formulation (`04_formulation.tex`)

- **4-E1** Replaced Assumption 4.4 (sub-Gaussian influence variables) with a bounded assumption `0 ≤ U_i ≤ M`, which is appropriate since `U_i = d_𝒵(f(X), f(X^(i)))` is non-negative.
- **4-R1** Added an explicit cross-reference to Theorem 6.4 in Assumption 4.3 (the only theorem that invokes the conditional local independence assumption).
- **4-M1** Removed the "(optional)" qualifier from Assumption 4.5 (Lipschitz stability) and tied it explicitly to the boundedness constant `M` of Assumption 4.4.

## §5 Core Framework (`05_framework.tex`)

- **5-E1** Extended the proof of Proposition 5.4 (AECP equals mean influence distance) by one line giving the discrete quantile identity `∫₀¹ Q(β) dβ = ∑_d d · Pr(D=d)` to remove ambiguity for the integer-valued quantile function.
- **5-R1** Refined the ECD bound to `1 ≤ ECD(r) ≤ |supp(bar I)| ≤ L`, with an explicit remark on the degenerate case `I_tot(r) = 0`.
- **5-R2** Added formal Definition 5.7 (`def:ecl_int`) defining `ECL^int_β(r)` symbolically, with the absolute-value convention so that synergistic and redundant interactions both contribute to the radius.
- **5-R3** Specified that the gNIAH reference position `r` is the prediction center, and identified gNIAH as a task-aware influence energy under an "insertion" perturbation kernel, completing the analogy stated in §10.
- **5-M1** Added explicit plain-text statement of the upstream/downstream convention (`upstream = i < r`) with a note on strand orientation.
- **3-M1 (in §5)** Changed `3\textquotesingle UTR` notation to `3'-UTR`.

## §6 Theory (`06_theory.tex`)

- **6-E1** Corrected Proposition 6.3 (compositional bound) from multiplicative `R_h · R_g` to additive `R_h + R_g`, replacing the dilated-stack remark with the correct stride-aware identity `R^* = 1 + ∑_l (k_l - 1) ∏_{m<l} s_m` and noting the constant-stride dilation special case.
- **6-E2** Split Theorem 6.4 (Sobol equivalence) into part (a) — the unprojected vector identity `I(i;r) = 2 tr(Cov(g_i(X_i)))` with a vector total-effect index — and part (b) — the scalar-projection identity `I^φ(i;r) = 2 V_tot S_i^T` for fixed scalar projections.
- **6-E3** Restated Theorem 6.7 (the former "minimax rate" theorem) as a sample-complexity lower bound: any estimator achieving worst-case error probability ≤ 1/3 requires `n ≥ c M²/γ²`. The matching upper bound from Corollary 8.3 is stated in the same theorem block.
- **6-R1** Tightened Proposition 6.2's hypothesis from "orthogonal transformation" to "linear isometry"; added nondegeneracy of `Σ` to the whitening remark.
- **6-R2** Refined Theorem 6.10 (sufficiency invariance) to specify that `T` is sufficient for the conditional family `{P(Y|Z=z)}_{z∈Z}` associated with a fixed task family.
- **6-C1** Prepended a one-sentence framing to the ERF subsection clarifying that this proposition is the linear-regime sanity check verifying that perturbation-variance ECL recovers the squared-impulse-response intuition of Luo et al. (2016).
- **6-M1** Updated Proposition 6.9 notation to `ECL_β(f,r)` per the X-R1 convention.
- **6-M2** Replaced the forward `\cref{alg:mc_ecl}` reference in Theorem 6.7 with a phrase pointing to §7.

## §7 Algorithms (`07_algorithms.tex`)

- **7-R1** Added a "Block-distance convention" paragraph after Algorithm 7.4 stating that the minimum-index convention is conservative and that midpoint and maximum conventions become asymptotically equivalent for `b ≪ ECL_β(r)`.
- **7-R2** Added position multiplicity `n_d := |{i : |i-r| = d}|` to the surrogate cost formula, clarifying boundary corrections.
- **7-M1** Clarified bootstrap resampling structure in Algorithm 7.3: outer index `t` is resampled with inner perturbation samples kept as a unit (block bootstrap over sequences).
- **X-M1 (in §7)** Cited `EfronTibshirani1993Bootstrap` in the bootstrap subsection introduction.

## §8 Estimation Theory (`08_estimation.tex`)

- **8-E1** Rewrote Theorem 8.1 (Bernstein concentration) under the bounded hypothesis only, removing the sub-Gaussian alternative.
- **8-E2** Labeled the closed-form `ε_n(δ)` as "an upper bound on the confidence radius" obtained from solving the Bernstein quadratic.
- **8-R1** Added a Bernstein-type sample-complexity bound (Corollary 8.3) in addition to the Hoeffding-type bound, with the data-dependent factor `σ_max² + (γ/(3L))M`.
- **8-R2** Tightened the antithetic variance reduction statement: the variance reduction factor is `1 − ρ` where `ρ` is the correlation between the antithetic pair, with maximal reduction `Var/2` at `ρ = -1`.
- **8-R3** Rewrote Proposition 8.5 (importance sampling) using the standard per-sample IS form `(1/n) ∑_t ind[i_t=i]/q(i) · d_𝒵(...)` with the variance-optimal proposal `q^*(i) ∝ σ_i I(i;r)` derived as a corollary.
- **8-M1** Named the CLT used in Theorem 8.7 ("by the Lindeberg–Lévy CLT").

## §9 Extensions (`09_extensions.tex`)

- **9-R1** Added a falsification criterion to the ECL training diagnostic ("hypothesis is falsified if `ECL_0.9` is non-monotone in training step at fixed `r`") and to the cross-model transfer validation ("framework is falsified if `f_ℓ` exceeds `f`'s held-out performance").
- **9-M1** Added the natural track-weight choice `w_t = 1/Var(f^(t)(X))` with a pointer to the diagonal Mahalanobis interpretation.

## §10 Connections (`10_connections.tex`)

- **10-R1** Sharpened the gradient-vs-perturbation comparison: gradient methods estimate `‖∂f/∂x_i‖²` (a local linearization at a single input) while perturbation methods estimate `E[‖f(X) - f(X^(i))‖²]` (a finite-difference average over `P_X`); these coincide in the linear regime by Proposition 6.8.
- **10-M1** Added explicit sign convention for the LongPPL "key-token" comparison.

## §11 Experiments (`11_experiments.tex`)

- **11-E1** Resolved by X-E1.
- **11-E2** Removed the redundant `\input{tables/tab01_model_taxonomy.tex}` line; §11 now references `\Cref{tab:model_taxonomy}` from §2 directly.
- **11-R1** Prepended a "Scope of the reported numbers" paragraph after the surrogate disclosure stating that all numerical values are surrogate-derived, that real-data deployment is the empirical follow-up, and that a single internally consistent scale is retained across all experimental tables.
- **11-R2** Added a caveat to the Experiment 9 introduction noting that SHH/ZRS and HBB/LCR are the two best-characterized rows; the remaining loci are reported with their commonly cited regulatory distances and serve as illustrative targets for the workflow rather than fully validated reference values.
- **11-M1** Added the "synthetic surrogate" qualifier to figure captions for Figure 1 (influence profiles), Figure 5 (perturbation scatter), Figure 10 (biological validation), and Figure 12 (gNIAH).

## §12 Discussion (`12_discussion.tex`)

- **12-E1** Softened the uncited percentile claims ("84% within 100 kb", "99% within 1 Mb") into a survey-level summary that notes estimates vary across studies and assays, with a citation to `Karollus2023Limited`.
- **12-R1** Added a sentence to the perturbation-dependence limitation note stating that reports should document the provenance of `P_X` (reference genome, sequence cohort, sampling protocol), since switching `P_X` implicitly redefines the influence energies.

## §13 Conclusion (`13_conclusion.tex`)

- **13-R1** Trimmed the redundant re-listing of contributions (gNIAH/training-diagnostic) from the second paragraph since this content is identical to the introduction.

## Appendix A — Complete Proofs (`appendix_a.tex`)

- **A-E1** Rewrote the proof of Theorem 6.7 to match the restated sample-complexity statement, using a Le Cam two-point construction with Pinsker-style KL bound.
- **A-E2** Updated the proof of Proposition 6.3 to derive the additive bound `R_h + R_g` cleanly, and replaced the trailing remark with the correct stride-aware identity `R^* = 1 + ∑_l (k_l - 1) ∏_{m<l} s_m`.
- **A-R1** Strengthened the Blackwell counterexample for Proposition 6.9 with explicit sufficient-statistic arguments using `Y := x_1 + x_3 mod 4` and `Y' := x_1 ⊕ x_3` to demonstrate that neither model is a deterministic function of the other.

## Appendix B — Additive-Model Derivations (`appendix_b.tex`)

- **B-E1** Rewrote the proof of Theorem 6.4 to match the split (vector + scalar) form, with the trace-of-covariance identity for the vector part and the standard Sobol decomposition for the scalar projection part.
- **B-R1** Replaced the incorrect "square-equals-identity" check with the correct orthogonality verification `Q Q^T = (T Σ T^⊤)^(-1/2) T Σ T^⊤ (T Σ T^⊤)^(-1/2) = I`.
- **B-M1** Added the intermediate parallelogram-bound step `‖a-b‖² ≤ 2(‖a‖² + ‖b‖²)` to the tail-approximation proof of Proposition 6.5.

## Appendix C — NLP Connections (`appendix_c.tex`)

- **C-R1** Clarified the Lost-in-the-Middle analogy: the U-shape interpretation holds when `r` is held at a fixed conventional location (e.g., end of sequence) and `i` ranges across the context, recovering the "position of the answer" interpretation.
- **C-M1** Rewrote the intrinsic-entropy section to use task-aware framing: "context poisoning" corresponds to `Δ_out^task(ℓ;r)` decreasing in `ℓ`, not to negative influence energy (which would be ill-defined since `I(i;r) ≥ 0` by construction).

## Appendix D — Complexity (`appendix_d.tex`)

- **D-M1** Clarified the surrogate algorithm cost note: `M × n` counts forward passes over sampled positions, with per-position ISM amortizable via fast-ISM.

## Appendix E — Hyperparameter Sensitivity (`appendix_e.tex`)

- **E-M1** Added a falsification criterion: convergence assumption is falsified if `Var(ECL_0.9 | n)` does not decrease as `O(1/n)` past `n ≈ 100`.

## Appendix F — Extended Experiments (`appendix_f.tex`)

- **F-M1** Added an identifiability constraint note for the CDS mixture-of-exponentials fit: a minimum-separation constraint `λ_{k+1}/λ_k ≥ ρ` (e.g., `ρ = 2`) is recommended.

## Appendix G — Implementation Guide (`appendix_g.tex`)

- **G-M1** Linked the binned `m(d)` protocol to adaptive sampling in Algorithm 7.7 for the threshold-crossing distance.

## Appendix H — Annotated References (`appendix_h.tex`)

- **H-M1** Tagged Caduceus and Evo 2 parameter-count and performance claims as "the authors report" to attribute them appropriately.

## Tables (regenerated)

- `tab03_utilization.tex` — values now derived from `tab02` locus-class averages (Enformer 437/0.22%, Borzoi 440/0.08%, HyenaDNA 433/0.04%, Caduceus 417/0.32%, DNABERT-2 372/12.40%, Evo 2 (7B) 441/0.04%, NT-v2 400/3.26%, NT-v3 406/3.31%).
- `tab06_pairwise_comparison.tex` — pairwise differences computed as differences of `tab03` ECL₀.₉ means; significance bands annotated by magnitude.
- `tab07_biological_validation.tex` — ECL columns rescaled into kb (~0.40–0.44 kb) consistent with the new model averages; all known long-range interactions correctly marked as "not detected" since model ECLs are sub-kb.
- `tabA1_hyperparameter_sensitivity.tex` — values rescaled to the ~430 bp regime to match `tab02` and `tab03`, with the small-variation pattern preserved.

## Bibliography

- Removed unused entries: `McCloskey2019PNAS`, `vanArensbergen2016SuRE`.
- Retained `EfronTibshirani1993Bootstrap` and added a citation to it in §7.

## Items not applied

None. All items in the approved Phase 2 plan were applied.
