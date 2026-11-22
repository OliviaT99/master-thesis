# Budget-Constrained PT-CATE — Progress Summary for Supervisor Meeting

**Olivia Tverdohleb · Master's thesis · Status as of 13 November 2026**

---

## 0. In short

- **What changed since the last version you saw (9 Oct, "Budget_Constrained_PT_CATE"):** I rewrote the methodology completely. The old version learned a budget "shadow price" λ inside the PT-CATE loss and updated it with dual ascent. I dropped that design because it had several theoretical problems (Section 2). Now the budget is handled by a separate **allocation step**. It takes any CATE score, ranks customers by benefit per unit of cost, and fills the budget. PT-CATE then trains the score so that this allocation step makes good decisions.
- **Theory status:** The core formulation is defined and its basic properties are derived. In a small worked example I showed that the problem PT-CATE addresses (best CATE fit ≠ best decisions) **also exists under a budget**. I also found and verified a **limitation of the smoothing approach**: even an exactly optimal smooth solution can lose value when it is deployed as a hard ranking.
- **Empirical status:** The X5 data pipeline is audited and a serious leakage bug is fixed. A first signal and power check on X5 (validation split only) shows that **causal targeting clearly beats predictive targeting**. But it also shows that X5 (a) gives **little evidence that the restricted linear model loses anything to a flexible one**, and (b) **probably can't statistically detect** differences between policies as small as I expect from PT-CATE.
- **Main things I need your input on:** (1) Is the narrowed contribution acceptable? (2) How should I handle the smooth-vs-hard deployment question? (3) What should carry the main result: synthetic experiments, X5, or both? (Section 6)

---

## 1. Research question and contribution

**Base paper:** Frauen et al. (NeurIPS 2025) show that when the CATE model is restricted (it can't represent the true effect exactly), the model with the smallest CATE error does not necessarily give the best treatment decisions. Their PT-CATE loss mixes CATE error with policy value through a weight γ. Their decision rule is "treat if the estimated effect > 0", with no budget.

**My extension:** retail promotion targeting (X5 RetailHero SMS campaign) where treatment costs **differ by customer** and there is a **fixed budget**.

> **Working research question:** Does policy-targeted training of a restricted CATE model improve budget-constrained targeting decisions, compared with ordinary CATE estimation followed by *the same* budget allocation step?

**Contribution (narrowed on purpose).** The budget allocation rule itself is not new. Ranking customers by benefit/cost and filling the budget is established (Bhattacharya & Dupas 2012; Zou et al. 2020 at Uber; Sun et al. 2021; Cerulli 2026). My contribution is **combining that allocation rule with the PT-CATE training objective** and testing empirically whether policy-targeting still helps once a budget changes where the decision boundary lies. The contribution is an *empirically validated methodological extension*, not a new optimization theorem. A null or conditional result is still a valid answer.

---

## 2. What changed since the 9 October version and why

| Topic | Old version (9 Oct) | Now | Why |
|---|---|---|---|
| Handling the budget | λ inside the PT-CATE loss, outer dual-ascent loop adjusts λ until spend ≈ B | A separate **allocation step** solves the budget for any score g, and PT-CATE evaluates the policy it produces | Dual ascent has no convergence/optimality guarantee for a non-convex neural problem. It was also unclear whether λ is the budget's shadow price or just a penalty weight once it is mixed with γ. |
| Budget definition | Total budget B for a cohort | **Budget per eligible customer** b = B_N / N | A total budget isn't comparable across train/validation/test splits of different sizes. |
| Binary vs. fractional | Mixed the binary 0–1 knapsack with the fractional relaxation | Main problem is the **population (fractional, randomized) problem**. Finite binary allocation is treated separately. | Greedy ratio ranking is exact only for the fractional problem, not for binary knapsack. |
| Smooth approximation | Frauen's adaptive sigmoid σ(α(x)[g − λc]) with learned α(x) | **Entropy-smoothed allocation** with the threshold re-calibrated so the budget always holds, with exact gradients | The α(x) region detector needs fresh justification once costs are subtracted (noisy DR pseudo-outcome signs). The new operator has a clean derivation. |
| Cost function | c₀ + β·basket (SMS cost + discount rate β) | c_k(x) = c̄·[(1−k) + k·basket/mean basket], **k = cost heterogeneity** | β (discount depth) can't be sourced for X5, so I sweep how heterogeneous costs are while holding the average cost fixed. |
| Model class G | Fixed-architecture neural network | **Linear score primary**, segment table secondary, NN dropped | Matches Frauen's real-data setup and keeps the restriction deliberate and interpretable. |

---

## 3. Current methodology (non-technical overview)

**3.1 Setting.** τ(x) is the true effect of the SMS on purchase probability. A policy π(x) ∈ [0,1] gives the probability of treating customer x. Its value is the expected number of extra purchases, **V(π) = E[π(X)·τ(X)]**. For example, V = 0.01 means 10 extra purchases per 1,000 eligible customers.

**3.2 Costs and budget.**
- Costs are *scenario-defined*, because X5 has no cost data: c_k(x) = c̄·[(1−k) + k·h(x)/h̄], where h = historical average basket value (with a small positive floor).
  - k = 0 means all customers cost the same, so the problem reduces to a "treat the top q%" rule.
  - k = 1 means cost is fully proportional to basket size.
  - The average cost is the same for every k.
- Budget per customer: b = q·c̄. Here q = "what treating a fraction q at average cost would spend". Grid: q ∈ {0.1, 0.3, 0.5}, primary q = 0.3, which matches the original top-30% idea.
- Budget constraint: **E[c(X)·π(X)] ≤ b**.

**3.3 Allocation step A_b(g).** Given any score g:
1. rank customers by g(x)/c(x);
2. treat from the top until the budget is used;
3. randomize only at the boundary;
4. never treat anyone with g ≤ 0;
5. unused budget is allowed.

This is the standard fractional-knapsack solution. Every arm uses the same allocation step, so arms differ only in *how the score was learned*.

**3.4 Budget-constrained PT-CATE objective.**

  J(g) = (1−γ)·E[(τ − g)²] − γ·V(A_b(g)),  g ∈ G (linear)

- γ = 0 gives ordinary CATE regression (Arm 2).
- γ = 1 gives pure policy optimization.
- γ is selected on the validation split, as in Frauen.

**3.5 Making it trainable.** A_b(g) is a hard ranking, so it has no useful gradient. I replace it with a **smoothed allocation**: treatment probability σ((g/c − δ − λ)/T), where the threshold λ is re-solved each step so that the budget holds exactly. Gradients flow through λ by implicit differentiation; Section 4.1 explains why this matters.

**3.6 Comparison arms (decision D3).**

| Group | Policies |
|---|---|
| Baselines | random; cost-only (treat the cheapest customers) |
| Arm 1 | predictive score P(Y=1 \| X, A=1) |
| Arm 2 | linear CATE, γ = 0 |
| Arm 3 | budget PT-CATE (this thesis) |
| Arm 4 | Frauen's original PT-CATE plus the same allocator |
| Reference | flexible (gradient-boosting) DR-CATE |

- Evaluation: held-out doubly-robust (DR) policy value, reported per 1,000 customers, plus spend and treated share.
- Stage-1 nuisances: cross-fitted gradient boosting; propensity fixed at the RCT treated share (0.4998).

---

## 4. What has been tested so far

### 4.1 Theory checks (worked examples with three customer types; algebra verified with exact arithmetic)

| Result | What it shows | Status |
|---|---|---|
| **The problem exists under a budget** | A small population where the best-fitting CATE score gives a *worse* budget policy than another score in the same class. The value gap is 1/30, i.e. 33 purchases per 1,000. | Derived and checked. This is the budget analogue of Frauen's Theorem 4.1. |
| **Basic properties** | γ = 0 recovers ordinary CATE. If the model class contains the true τ, PT-CATE gives no advantage. Decision regret ≤ √(CATE error), so a good fit is sufficient but not necessary. | Derived |
| **Smoothing converges for a fixed score** | As T, δ → 0 (with δ/T → ∞), the smoothed allocation converges to the hard allocation. Weaker schedules can fail. | Derived |
| **Gradients must go through the threshold** | If λ is treated as fixed when taking gradients (the "detached" version), training can get stuck at a worse solution. The correct implicit gradient moves to the better one. | Derived, plus numerical check |
| **⚠ Smooth ≠ hard at deployment** | A three-type example where the hard objective has a unique best solution. Yet **every** exact optimum of the smooth objective, deployed as a hard ranking, loses value: the excess converges to 4/75 and never vanishes. The cause is ties: smoothing breaks near-ties differently from the hard ranking. | Proven analytically and verified numerically (`verify_exact_minimizer_counterexample.py`) |

**Consequence:** I can't claim that training the smooth surrogate solves the hard budget problem in general. **Decision D1** (to confirm with you): the **calibrated smooth (randomized) policy is the primary deployed policy**, and hard ranking of the same score is a secondary analysis. Both are applied identically to all arms, so the comparison stays fair.

### 4.2 Data pipeline audit (30 Oct – 9 Nov)

- **Critical leakage found and fixed.** One feature (`issue_redeem_delay`) used loyalty-card redemption dates recorded *after* the SMS. The SMS itself raised later redemption by 0.39 pp (z ≈ 2.7). This feature would therefore have created fake effect heterogeneity. It is now replaced by a "redeemed before the campaign" flag.
- Also fixed:
  - missing values in interval features;
  - ordinal encoding of gender (now one-hot);
  - no standardization;
  - lost client IDs and raw basket values (both needed to build costs);
  - added recency and card-tenure features.
- Confirmed:
  - all purchase history precedes the campaign;
  - RCT balance is good (all standardized mean differences < 0.02);
  - treated share is 0.4998 in every split;
  - the split is reproducible.

### 4.3 Phase 1: X5 signal and power check (9 Nov; train + validation only, test untouched)

**Purpose:** before building Arms 3/4, check whether X5 has enough detectable heterogeneity, and enough statistical power, to make the comparison meaningful.

**Is there heterogeneity?** Yes. A calibration test of the linear CATE score gives slope t = 7.9. The top score quintile has an effect of about 9.4 pp, versus about 0.7 pp in the bottom quintile. The DR average effect is about 3.1 pp.

**Policy value on the validation split (extra purchases per 1,000 customers), primary budget q = 0.3:**

| Policy | Equal costs (k=0) | Basket-proportional costs (k=1) |
|---|---:|---:|
| Random | 9.3 | 9.3 |
| Cost-only (cheapest first) | 9.3 | 23.3 |
| Arm 1 — predictive | 3.0 | 18.9 |
| Linear DR-CATE (Arm 2, γ=0) | **19.0** | **24.5** |
| Flexible DR-CATE (reference) | 17.9 | 20.9 |

**What this tells us:**
1. **Causal beats predictive, clearly.**
   - Linear CATE beats Arm 1 by 16 per 1,000 at k=0 (z ≈ 5.6) and by 5.6 per 1,000 at k=1 (z ≈ 2.5).
   - The predictive model mostly targets customers who would buy anyway. Its top quintile has the *lowest* effect.
   - The Arm 1 vs Arm 2 question already has a solid answer.
2. **With heterogeneous costs, "treat cheap customers" is already strong.**
   - At k=1, q=0.3, linear CATE beats cost-only by only 1.2 per 1,000 (z ≈ 0.7).
   - Cheap customers let you treat 55% of the population for the same budget.
   - So the cost scenario largely drives the results. This matters for how the cost proxy is interpreted.
3. **⚠ No headroom for the restricted class.**
   - The flexible model does *not* beat the linear one at any budget or cost setting. It is slightly worse everywhere; the best z is −0.5.
   - PT-CATE's advantage relies on the restricted class being misspecified in a way that matters for decisions. On X5 I currently see no evidence of that.
4. **⚠ Underpowered for the main comparison (Gate B).**
   - The smallest arm difference that the 30% test split can detect with 80% power is about **4.6 per 1,000**.
   - The gain I pre-specified as worth reporting is 3 per 1,000 (about 10% of the ATE).
   - So the X5 test set alone probably can't confirm or rule out a realistic PT-CATE gain.

(Exploratory: several budgets and cost settings were inspected without multiplicity adjustment. These numbers inform the design and are not thesis results.)

---

## 5. Remaining open issues

| # | Issue | Current handling |
|---|---|---|
| 1 | Smooth training vs. hard deployment can diverge (Section 4.1) | D1: deploy the smooth policy; report hard ranking as secondary; treat tie/near-tie sensitivity as a core robustness analysis |
| 2 | X5 shows no headroom and limited power (Section 4.3) | See question 3 below |
| 3 | Costs are hypothetical (basket value as a proxy); results depend strongly on k | Report k = 0 and k = 1 side by side; frame cost as a scenario, not measured campaign cost |
| 4 | X5 randomization protocol isn't documented (0.4998 is an *empirical* share) | Treat the "RCT" claim as needing documentation before the causal interpretation is stated |
| 5 | Expected-budget feasibility ≠ realized budget for a finite binary cohort (random draws can overspend) | Report finite-cohort feasibility and value loss separately |
| 6 | γ mixes two quantities on different scales (squared error vs. value) | Keep Frauen's unnormalized form for comparability; normalized version as robustness check |
| 7 | Smoothing hyperparameters: T = κ·spread of γ=0 ratios, δ = 5T | κ = 0.1 dropped (13 Nov) because the budget goes slack there; κ ∈ {0.02, 0.05} remain |

---

## 6. Questions for discussion

1. **Contribution scope.** Is a narrowed, empirically validated extension acceptable: PT-CATE + budget allocation, with an honest null result possible, and no convergence theorem?
2. **Deployment.** Do you agree that the smooth randomized policy should be the primary deployed policy, with hard ranking secondary, given the counterexample?
3. **Where the main evidence comes from.** X5 shows strong causal-vs-predictive results but no evidence that the linear model is too restrictive, and low power for small arm differences. Options:
   - (a) make **synthetic experiments** with controlled misspecification the primary evidence for the PT-CATE mechanism, and use X5 as the realistic application;
   - (b) make the restriction on X5 deliberately coarser (e.g. a small segment table) so there is a real gap to close;
   - (c) accept a likely null result on X5 and frame it as a finding.

   My inclination is (a) plus a short version of (b), but I'd like your view.
4. **Cost scenarios.** Is a scenario-based cost (basket proxy, heterogeneity k swept) acceptable without real campaign cost data?
5. **Comparators.** Is the arm set in Section 3.6 sufficient, or too much for the remaining time? In particular, is Arm 4 (Frauen + allocator) needed?

---

## 7. Next steps (target: finished by end of January 2027; deadline 9 April 2027)

1. Implement the smooth allocation and budget PT-CATE training for the linear class. Verify gradients, budget residuals and tie handling against the exact small-example results.
2. Synthetic experiments with known effects: correct vs. misspecified models, varying budgets and cost–effect correlation.
3. X5: train Arms 2–4 with γ selected on validation, then a **single** final evaluation on the test split with paired standard errors.
4. Robustness: budget grid, cost scenarios k, smooth vs. hard deployment, finite-cohort allocation.

---

### Key references

- Frauen, Melnychuk, Schweisthal, van der Schaar, Feuerriegel (2025). *Treatment Effect Estimation for Optimal Decision-Making.* NeurIPS.
- Bhattacharya & Dupas (2012). *Inferring welfare maximizing treatment assignment under budget constraints.* J. Econometrics.
- Zou, Du, Lee, Pedersen (2020). *Heterogeneous causal learning for effectiveness optimization in user marketing.* arXiv:2004.09702.
- Sun, Munro, Kalashnov, Du, Wager (2021). *Treatment allocation under uncertain costs.* arXiv:2103.11066.
- Cerulli (2026). *Optimal policy learning under budget and coverage constraints.* arXiv:2605.12235.
- Chernozhukov et al. *Generic machine learning inference on heterogeneous treatment effects* (calibration/BLP test used in Phase 1).

*Full derivations: Methodology Memorandum v0.3 (30 Oct 2026).*
