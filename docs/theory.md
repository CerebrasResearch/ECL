# Mathematical Background and Theory

This document provides the mathematical foundations behind the ECL framework as described in the paper *"Effective Context Length: A Perturbation-Variance Framework for Estimating and Comparing Context Utilization in Sequence Models."*

---

## 1. Setup and Notation

Let $f : \mathcal{X}^L \to \mathbb{R}^d$ be a sequence model that maps a DNA sequence $x = (x_1, \ldots, x_L)$ over alphabet $\mathcal{X} = \{A, C, G, T\}$ to a $d$-dimensional embedding. Let $r \in \{1, \ldots, L\}$ be a fixed reference position (e.g., the center of a gene promoter).

A perturbation kernel $\Pi_S(\cdot \mid x)$ replaces the nucleotides at a set $S \subseteq \{1, \ldots, L\}$ with randomised alternatives while leaving the remaining positions unchanged. The perturbed sequence is denoted $\tilde{x}^S$.

An embedding discrepancy $d_Z : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\ge 0}$ measures how much the embedding changes. The default is squared Euclidean distance: $d_Z(z, z') = \lVert z - z' \rVert_2^2$.

---

## 2. Influence Energy (Section 3, Equation 1)

**Definition.** The influence energy of position $i$ on reference position $r$ is

$$
I(i; r) \;=\; \mathbb{E}_{X \sim P_X}\!\Bigl[\, \mathbb{E}_{\tilde{X}^{\{i\}} \sim \Pi_{\{i\}}(\cdot \mid X)}\!\bigl[\, d_Z\bigl(f(X)_r,\; f(\tilde{X}^{\{i\}})_r\bigr) \,\bigr] \,\Bigr].
$$

Informally, $I(i; r)$ measures the expected change in the embedding at $r$ when position $i$ is perturbed. Positions that the model ignores contribute zero influence.

### Normalized Influence (Equation 2)

The normalized influence profile is the probability distribution over positions

$$
\bar{I}(i; r) \;=\; \frac{I(i; r)}{\sum_{j=1}^{L} I(j; r)}.
$$

### Binned Influence Profile (Equation 3)

For computational tractability, influence is binned by distance $d = |i - r|$:

$$
I(d; r) \;=\; \frac{1}{|S_d|} \sum_{i:\,|i-r|=d} I(i; r),
$$

where $S_d = \{i : |i - r| = d\}$ and $|S_d|$ is the multiplicity (1 for $d=0$, typically 2 for $d > 0$ at interior positions).

---

## 3. Cumulative Influence (Section 5, Equation 4)

**Definition.** The cumulative influence up to radius $l$ is

$$
I_{\le l}(r) \;=\; \sum_{d=0}^{l} |S_d| \cdot I(d; r).
$$

The total influence is $I_{\text{tot}}(r) = I_{\le L}(r)$.

---

## 4. Effective Context Length -- ECL (Definition 4.1, Equation 5)

**Definition 4.1.** The perturbation-variance effective context length at threshold $\beta \in (0, 1]$ is

$$
\text{ECL}_\beta(r) \;=\; \min\!\bigl\{\, l : I_{\le l}(r) \;\ge\; \beta \cdot I_{\text{tot}}(r) \,\bigr\}.
$$

$\text{ECL}_\beta(r)$ is the smallest radius (in base pairs) that captures at least a fraction $\beta$ of the total influence. The default threshold is $\beta = 0.9$.

---

## 5. Effective Context Profile -- ECP (Definition 4.2)

**Definition 4.2.** The effective context profile is the monotone non-decreasing map

$$
\text{ECP}(r) \;:\; \beta \;\mapsto\; \text{ECL}_\beta(r), \qquad \beta \in (0, 1].
$$

The ECP provides a complete picture of how context usage scales with the fraction of influence captured. Models with long, heavy-tailed influence profiles produce ECPs that rise steeply.

---

## 6. Area Under the ECP -- AECP (Definition 4.3)

**Definition 4.3.** The area under the effective context profile is

$$
\text{AECP}(r) \;=\; \int_0^1 \text{ECL}_\beta(r)\, d\beta \;=\; \sum_{i=1}^{L} |i - r|\, \bar{I}(i; r).
$$

The second equality shows that AECP equals the mean influence distance -- the expected distance from the reference weighted by normalized influence.

---

## 7. Effective Context Dimension -- ECD (Definition 4.5)

**Definition 4.5.** The effective context dimension is the exponential entropy of the normalized influence:

$$
\text{ECD}(r) \;=\; \exp\!\Bigl(-\sum_{i=1}^{L} \bar{I}(i; r)\, \log \bar{I}(i; r)\Bigr).
$$

ECD measures the effective number of positions contributing to the embedding. A model that focuses on a single position has $\text{ECD} = 1$; a model that weights all $L$ positions equally has $\text{ECD} = L$.

---

## 8. Directional ECL (Equations 10--11)

To distinguish upstream (5') from downstream (3') context usage, define the one-sided influence profiles:

$$
I^{-}(d; r) = I(r - d; r), \qquad I^{+}(d; r) = I(r + d; r), \qquad d = 1, 2, \ldots
$$

**Directional ECL:**

$$
\text{ECL}^{-}_\beta(r) = \min\bigl\{l : \textstyle\sum_{d=1}^{l} I^{-}(d;r) \ge \beta \cdot \sum_{d=1}^{D} I^{-}(d;r)\bigr\}
$$

and analogously for $\text{ECL}^{+}_\beta(r)$.

The asymmetry ratio $\text{ECL}^{+}_\beta / \text{ECL}^{-}_\beta$ quantifies directional bias. A ratio greater than 1 means the model uses more downstream context.

---

## 9. Interaction Influence (Equation 9)

**Definition.** The pairwise interaction influence is

$$
I_{\text{int}}(i, j; r) \;=\; I(\{i, j\}; r) - I(i; r) - I(j; r).
$$

- $I_{\text{int}} > 0$: synergistic -- jointly perturbing $i$ and $j$ has a greater effect than the sum of individual perturbations.
- $I_{\text{int}} < 0$: redundant -- the information from $i$ and $j$ overlaps.
- $I_{\text{int}} = 0$: additive.

The interaction ECL extends ECL to capture non-additive dependencies between distant positions.

---

## 10. Context Decay Spectroscopy -- CDS (Section 5.8)

CDS decomposes the influence profile into a mixture of exponentials:

$$
I(d; r) \;\approx\; \sum_{k=1}^{K} a_k \, e^{-\lambda_k\, d}.
$$

Each component represents a distinct context channel:
- $a_k$ is the amplitude (relative contribution).
- $\lambda_k$ is the decay rate; the half-life is $\ln 2 / \lambda_k$ bp.

The number of components $K$ is selected by minimizing the Bayesian Information Criterion (BIC).

**Spectral ECL.** The ECL can be computed analytically from the CDS fit using the closed-form cumulative:

$$
I_{\le l}^{\text{CDS}} \;=\; \sum_{k=1}^{K} \frac{a_k}{\lambda_k}\bigl(1 - e^{-\lambda_k\, l}\bigr).
$$

---

## 11. Genomic Needle-in-a-Haystack -- gNIAH (Section 5.10, Equation 12)

The gNIAH protocol provides a complementary, biologically interpretable measure of context sensitivity. A known regulatory motif $m$ is inserted into a neutral background sequence at varying distances from the prediction center:

$$
\text{gNIAH}(d, m) \;=\; \mathbb{E}_{X_{\text{neutral}}}\!\bigl[\, d_Z\bigl(f(X_{\text{neutral}}),\; f(X_{\text{neutral}}^{+m@d})\bigr) \,\bigr].
$$

A model with short effective context will show gNIAH sensitivity that drops to zero at large $d$, while a model that truly uses long-range context will remain sensitive.

---

## 12. Estimation Theory (Section 8)

### Bernstein Bound (Theorem 8.1, Equation 14)

For $n$ i.i.d. samples with variance $\sigma^2$ and bound $|U_i| \le M$, the influence estimate satisfies

$$
\Pr\!\bigl[\,|\hat{I}_n(i;r) - I(i;r)| \ge \varepsilon_n(\delta)\,\bigr] \;\le\; \delta,
$$

where the confidence radius is

$$
\varepsilon_n(\delta) = \frac{M \log(2/\delta)}{3n} + \sqrt{\frac{2\sigma^2 \log(2/\delta)}{n}}.
$$

### Sample Complexity (Corollary 8.3, Equation 15)

The minimum sample size for exact ECL estimation with probability $1 - \delta$ is

$$
n \;\ge\; \frac{128\, L^2 M^2}{\gamma^2}\, \log\!\Bigl(\frac{2L}{\delta}\Bigr),
$$

where $\gamma$ is the influence margin at the true ECL boundary.

### Bootstrap Confidence Intervals (Algorithm 3)

Given per-sample influence values $\{d_Z^{(t)}\}_{t=1}^n$, bootstrap $B$ resamples of size $n$ with replacement. For each resample, compute $\text{ECL}_\beta$. The $100(1 - \alpha)\%$ CI is $[\hat{q}_{\alpha/2}, \hat{q}_{1-\alpha/2}]$.

### Antithetic Variance Reduction (Proposition 8.4)

Using complementary perturbation pairs $(x^{+}, x^{-})$, the antithetic estimator $\hat{I}^{\text{anti}} = (d_Z^{+} + d_Z^{-})/2$ achieves lower variance than the standard estimator when the perturbation responses are negatively correlated.

### Permutation Test for Model Comparison (Algorithm 5)

To test $H_0: \mathbb{E}[\text{ECL}^f(R)] = \mathbb{E}[\text{ECL}^g(R)]$, compute paired differences $D_j = \text{ECL}^f(r_j) - \text{ECL}^g(r_j)$ at $N$ matched loci. Under $H_0$, each $D_j$ is equally likely to be positive or negative. The test statistic $T = |\bar{D}|$ is compared against $P$ sign-flip permutations.

---

## 13. Theoretical Properties (Section 6)

Key results from the paper (stated informally):

- **Proposition 6.1 (Faithfulness).** If position $i$ has zero influence on the embedding at $r$ (for all inputs), then $I(i; r) = 0$.

- **Proposition 6.3 (Locality bound).** If the model has a finite receptive field of radius $R$ around $r$, then $\text{ECL}_\beta(r) \le R$ for all $\beta \le 1$.

- **Theorem 6.4 (Sobol equivalence).** For additive models $f(X) = \sum_i g_i(X_i)$, the perturbation-variance influence $I(i; r)$ equals the Sobol first-order sensitivity index (up to a constant).

- **Proposition 6.5 (Perturbation ordering).** If perturbation $\Pi'$ is "stronger" than $\Pi$ in a precise sense, then $I_{\Pi'}(i; r) \ge I_{\Pi}(i; r)$ for all $i$, implying ECL is monotone in perturbation strength.

---

## References

For full proofs, see the paper:

> *Effective Context Length: A Perturbation-Variance Framework for Estimating and Comparing Context Utilization in Sequence Models.*
