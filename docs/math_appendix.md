Derivations underlying FinMetrica's core models, provided for reference and study.

### 1. Black-Litterman posterior (Bayesian normal-normal update)

**Setup.** Let $x = \Pi$ denote the (unknown) vector of true equilibrium excess returns. The prior belief is
$$x \sim \mathcal{N}(\pi,\ \tau\Sigma), \qquad \pi = \delta \Sigma w_{mkt}$$
where $\delta$ is the risk-aversion coefficient and $\tau$ is a small scalar controlling confidence in the prior.

The investor's $K$ views are expressed as a linear system with noise:
$$Q = Px + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\ \Omega)$$
where $P$ is the $K \times N$ picking matrix and $\Omega$ is the (diagonal) view-uncertainty matrix.

**Derivation.** By Bayes' rule, the posterior density is proportional to the product of the prior and likelihood densities:
$$p(x \mid Q) \propto \exp\left(-\tfrac{1}{2}(x-\pi)^\top(\tau\Sigma)^{-1}(x-\pi)\right)\cdot\exp\left(-\tfrac{1}{2}(Q-Px)^\top\Omega^{-1}(Q-Px)\right)$$

Combine the two exponents and expand each quadratic form:
$$(x-\pi)^\top(\tau\Sigma)^{-1}(x-\pi) = x^\top(\tau\Sigma)^{-1}x - 2x^\top(\tau\Sigma)^{-1}\pi + \pi^\top(\tau\Sigma)^{-1}\pi$$
$$(Q-Px)^\top\Omega^{-1}(Q-Px) = x^\top P^\top\Omega^{-1}Px - 2x^\top P^\top\Omega^{-1}Q + Q^\top\Omega^{-1}Q$$

Summing, the exponent is $-\tfrac{1}{2}\left[x^\top A x - 2x^\top b + c\right]$ where
$$A = (\tau\Sigma)^{-1} + P^\top\Omega^{-1}P, \qquad b = (\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q$$
and $c$ collects the terms not involving $x$ (absorbed into the normalizing constant).

**Completing the square:** $x^\top Ax - 2x^\top b = (x - A^{-1}b)^\top A (x - A^{-1}b) - b^\top A^{-1}b$. Since the second term doesn't depend on $x$, it folds into the constant, leaving
$$p(x\mid Q) \propto \exp\left(-\tfrac{1}{2}(x - A^{-1}b)^\top A (x-A^{-1}b)\right)$$

This is exactly the kernel of a Gaussian, so
$$x \mid Q \sim \mathcal{N}\big(A^{-1}b,\ A^{-1}\big)$$

Substituting $A$ and $b$ back gives the posterior mean and covariance used in `m7_optimisation.py`:
$$\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q\right], \qquad \Sigma_{post} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}$$

**Interpretation:** the posterior mean is a precision-weighted average of the equilibrium prior and the views — an asset's final expected return leans toward the view the more confident that view is ($\Omega^{-1}$ large) relative to prior confidence ($(\tau\Sigma)^{-1}$).

---

### 2. Maximum Sharpe Ratio: closed form and KKT conditions

**The unconstrained case (no box constraints), via Charnes–Cooper scale-invariance trick.** Let $\mu_e = \mu - r_f\mathbf{1}$ (excess returns). The Sharpe ratio $SR(w) = \dfrac{w^\top\mu_e}{\sqrt{w^\top\Sigma w}}$ is invariant to positive rescaling of $w$: $SR(kw) = SR(w)$ for any $k>0$. This lets us replace the true budget constraint temporarily with a numerator normalization, solve, then rescale:

$$\min_w \tfrac{1}{2}w^\top\Sigma w \quad \text{s.t.} \quad w^\top\mu_e = 1$$

Lagrangian: $L = \tfrac{1}{2}w^\top\Sigma w - \gamma(w^\top\mu_e - 1)$. Stationarity: $\Sigma w - \gamma\mu_e = 0 \Rightarrow w = \gamma\Sigma^{-1}\mu_e$. Substituting into the constraint: $\gamma\, \mu_e^\top\Sigma^{-1}\mu_e = 1 \Rightarrow \gamma = 1/(\mu_e^\top\Sigma^{-1}\mu_e)$.

Rescaling so weights sum to 1 (recovering the true budget constraint, valid since $SR$ is scale-invariant):
$$w^* = \frac{\Sigma^{-1}\mu_e}{\mathbf{1}^\top\Sigma^{-1}\mu_e}$$
This is the classical **tangency portfolio**.

**Why the code doesn't use this closed form.** `optimise_max_sharpe` adds box constraints $0 \le w_i \le 0.40$ (long-only, 40% concentration cap). $\Sigma^{-1}\mu_e$ can contain negative or highly concentrated entries, violating these bounds — so the closed form is infeasible and the problem must be solved numerically (SLSQP) as a genuinely constrained nonlinear program.

**Full KKT conditions for the constrained problem.** With $f(w) = SR(w)$, $\sigma_p = \sqrt{w^\top\Sigma w}$, the gradient is
$$\nabla f(w) = \frac{\mu_e}{\sigma_p} - \frac{(w^\top\mu_e)\,\Sigma w}{\sigma_p^3} = \frac{1}{\sigma_p}\left[\mu_e - SR(w)\,\frac{\Sigma w}{\sigma_p}\right]$$

Lagrangian with budget multiplier $\lambda$ and box multipliers $\mu^{hi}_i, \mu^{lo}_i \ge 0$:
$$L(w,\lambda,\mu^{hi},\mu^{lo}) = f(w) - \lambda(w^\top\mathbf{1}-1) - \sum_i \mu^{hi}_i(w_i - w_{max}) + \sum_i \mu^{lo}_i\, w_i$$

**Stationarity:** $\nabla f(w^*) - \lambda^*\mathbf{1} - \mu^{hi*} + \mu^{lo*} = 0$
**Complementary slackness:** $\mu^{hi*}_i(w_i^* - w_{max}) = 0$, $\ \mu^{lo*}_i\, w_i^* = 0$ for all $i$
**Primal feasibility:** $w^{*\top}\mathbf{1}=1,\ 0\le w_i^* \le w_{max}$
**Dual feasibility:** $\mu^{hi*}_i,\ \mu^{lo*}_i \ge 0$

For any asset strictly inside its bounds ($\mu^{hi}_i=\mu^{lo}_i=0$), stationarity reduces to
$$\mu_{e,i} - SR(w^*)\,\frac{(\Sigma w^*)_i}{\sigma_p(w^*)} = \lambda^*\sigma_p(w^*)$$
i.e. every unconstrained asset's marginal-return-minus-risk-adjusted-marginal-risk is equalized. Setting all box multipliers to zero everywhere (no active bounds) recovers exactly the proportionality $\Sigma w^* \propto \mu_e$ — the closed-form tangency portfolio above is the special case of this KKT system when no constraint is active.

---

### 3. Ledoit–Wolf shrinkage: the bias–variance argument

**The problem.** The sample covariance $S$ is unbiased, $\mathbb{E}[S] = \Sigma_{true}$, but has high variance, especially when the number of assets $N$ is not small relative to the sample size $T$. This inflates the largest eigenvalues and deflates the smallest ones relative to their true values, which is exactly why the condition number of $S$ (Task B) is large and the optimizer amplifies noise.

**The estimator.** Shrinkage blends $S$ with a low-variance, structured target $F$ (e.g. constant-correlation or scaled-identity matrix):
$$\Sigma_{shrink} = \alpha F + (1-\alpha)S, \qquad \alpha \in [0,1]$$

**Bias–variance decomposition.** Since $\mathbb{E}[S]=\Sigma$, write the estimation error as
$$\Sigma_{shrink} - \Sigma = \alpha(F - \Sigma) + (1-\alpha)(S-\Sigma)$$
Taking expectation, $\mathbb{E}[\Sigma_{shrink}-\Sigma] = \alpha(F-\Sigma)$ — this is the **bias**, growing linearly in $\alpha$ and in how wrong the target $F$ is. The second term $(1-\alpha)(S-\Sigma)$ has mean zero but variance $(1-\alpha)^2\mathrm{Var}(S)$ — this is the **variance**, shrinking as $\alpha$ grows.

The expected squared Frobenius-norm loss is approximately
$$\mathbb{E}\left[\|\Sigma_{shrink}-\Sigma\|_F^2\right] \approx \alpha^2\|F-\Sigma\|_F^2 + (1-\alpha)^2\,\mathbb{E}\left[\|S-\Sigma\|_F^2\right]$$

where the first term is the squared bias and the second is the variance.

This is the classical bias–variance tradeoff: $\alpha=0$ recovers the unbiased-but-noisy $S$; $\alpha=1$ recovers the low-variance-but-biased $F$. There exists an interior $\alpha^* \in (0,1)$ minimizing the sum. Ledoit & Wolf (2004) derive a consistent, closed-form estimator of this optimal $\alpha^*$ directly from the data (without needing to know $\Sigma_{true}$) by estimating the asymptotic variance and misspecification terms from the sample; see Ledoit, O. and Wolf, M. (2004), *"A well-conditioned estimator for large-dimensional covariance matrices,"* Journal of Multivariate Analysis, for the exact estimator formulas.

**Connection to Task B.** Shrinkage pulls $S$'s extreme eigenvalues toward $F$'s (typically flatter) eigenvalue spectrum, which is precisely why $\Sigma_{LW}$'s condition number is lower than $\Sigma_{sample}$'s in the Track 1 experiment — this section is the theoretical explanation for that empirical result.

---

### 4. CPCV purging and embargo (leakage mechanism)

Standard $k$-fold cross-validation assumes i.i.d. samples. Financial observations are not i.i.d.: a label $y_t$ built from a forward return over horizon $h$ spans $[t, t+h]$, so if an observation at $t$ is in the training set and an observation at $t' \in (t, t+h)$ is in the test set, the training label's window overlaps the test period — information from the test period leaks into training through the label construction, inflating apparent skill.

**Purging** removes any training observation whose label window $[t, t+h]$ overlaps the test set's window $[t_{test,start}, t_{test,end}]$:
$$\text{drop } t \text{ from training if } \neg\big(t+h < t_{test,start} \ \text{or}\ t > t_{test,end}\big)$$

**Embargo** additionally removes a further window of length $e$ immediately after the test period, because autoregressive dependence in financial time series (volatility clustering, momentum) means information can leak backward even without direct label overlap, through serial correlation in the *features* rather than the labels.

**The calendar-time subtlety (see Task A).** Both windows must be defined in true calendar time — `pd.Timedelta`-based comparisons against actual timestamps — not integer index-position offsets, because trading calendars have irregular gaps (weekends, holidays, halts). An index-position embargo of "20 samples" is a different true calendar length depending on how many non-trading days fall inside it, which silently changes the embargo's actual protection. This is exactly the bug Task A fixes.

---

### 5. Jobson–Korkie / Memmel test for equality of two Sharpe ratios

**Setup.** Two paired return series (same dates) for methods $a$ and $b$, with sample means $\hat\mu_a,\hat\mu_b$, sample standard deviations $\hat\sigma_a,\hat\sigma_b$, and sample covariance $\hat\sigma_{ab}$ between them. Define $\widehat{SR}_a = \hat\mu_a/\hat\sigma_a$, $\widehat{SR}_b=\hat\mu_b/\hat\sigma_b$.

**Reformulating the null hypothesis.** $H_0: SR_a = SR_b \iff \mu_a/\sigma_a = \mu_b/\sigma_b \iff \sigma_b\mu_a - \sigma_a\mu_b = 0$. Testing the difference of two ratios directly is awkward (ratio of two noisy quantities); testing the equivalent linear functional $\psi = \sigma_b\mu_a - \sigma_a\mu_b$ is not, and is what Jobson & Korkie (1981) propose.

**Asymptotic distribution.** The sample moments $(\hat\mu_a,\hat\mu_b,\hat\sigma_a^2,\hat\sigma_b^2,\hat\sigma_{ab})$ are jointly asymptotically normal by the multivariate CLT (standard result for sample moments of a stationary series). Applying the multivariate delta method to the nonlinear map $\hat\psi = \hat\sigma_b\hat\mu_a - \hat\sigma_a\hat\mu_b$ and propagating the covariance structure of the input moments yields, after Memmel's (2003) correction of an error in the original 1981 derivation:
$$\hat\theta = \frac{1}{T}\left[2\hat\sigma_a^2\hat\sigma_b^2 - 2\hat\sigma_a\hat\sigma_b\hat\sigma_{ab} + \tfrac{1}{2}\hat\mu_a^2\hat\sigma_b^2 + \tfrac{1}{2}\hat\mu_b^2\hat\sigma_a^2 - \frac{\hat\mu_a\hat\mu_b}{\hat\sigma_a\hat\sigma_b}\hat\sigma_{ab}^2\right]$$

**Test statistic:**
$$z = \frac{\hat\sigma_b\hat\mu_a - \hat\sigma_a\hat\mu_b}{\sqrt{\hat\theta}} \ \xrightarrow{d}\ \mathcal{N}(0,1) \ \text{under } H_0$$
Two-sided $p$-value: $p = 2\big(1-\Phi(|z|)\big)$.

Verify this formula against Memmel, C. (2003), *"Performance Hypothesis Testing with the Sharpe Ratio,"* Finance Letters, when implementing.

---

### 6. Deflated Sharpe Ratio (Bailey & López de Prado)

**The problem being corrected.** If you try $N$ strategy configurations and report the best observed Sharpe ratio, that maximum is biased upward even if every configuration has zero true skill — this is a multiple-testing/selection problem, directly analogous to the expected maximum of $N$ draws from a null distribution growing with $N$.

**Expected maximum Sharpe under the null, via extreme value theory.** For $N$ (effectively independent) trials with Sharpe-ratio estimates of variance $V$ across trials, Bailey & López de Prado (2014) approximate the expected maximum using the extreme-value asymptotics of the normal distribution:
$$\mathbb{E}[\max_n \widehat{SR}_n] \approx \sqrt{V}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\frac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1-\frac{1}{Ne}\right)\right]$$
where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant, $\Phi^{-1}$ the inverse standard normal CDF, and $e$ Euler's number. This is the multiple-testing-adjusted benchmark $SR^*$ that the observed best Sharpe must be compared against — not zero.

**Probabilistic Sharpe Ratio (accounting for non-normality).** Bailey & López de Prado (2012) show that the Sharpe ratio estimator's own sampling distribution is distorted by the skewness $\gamma_3$ and kurtosis $\gamma_4$ of the underlying returns. The probability that the true Sharpe ratio exceeds a benchmark $SR^*$, given $T$ observations:
$$PSR(SR^*) = \Phi\left[\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2}}\right]$$

**Deflated Sharpe Ratio.** $DSR = PSR(SR^*)$ evaluated at $SR^* = \mathbb{E}[\max_n \widehat{SR}_n]$ from above — i.e. "what is the probability the observed best strategy's true Sharpe ratio is actually positive, once you account for both (a) having tried multiple strategies and (b) non-normal returns."

Cross-check both formulas against Bailey, D.H. and López de Prado, M. (2014), *"The Deflated Sharpe Ratio,"* Journal of Portfolio Management, and Bailey & López de Prado (2012), *"The Sharpe Ratio Efficient Frontier,"* Journal of Risk, before finalizing the implementation in `src/stats/deflated_sharpe.py`.

---

> These derivations were compiled with AI assistance and cross-referenced against the cited papers; verify each against the primary source before relying on them, and be prepared to reproduce each derivation independently by hand.
