# Research Memorandum
## Budget-Constrained Policy-Targeted CATE Estimation
### Mathematical Foundations, Methodological Decisions, and Open Questions

**Project:** Master's Thesis — Budget-Constrained PT-CATE for Retail Promotional Targeting  
**Status:** Working methodology, version 0.3 — evidence-audited revision with independently checked exact-global-minimizer counterexample  
**Original date:** 26 October 2026  
**Revised:** 30 October 2026 (Europe/Berlin)  
**Scope:** Population decision problem, restricted policy representation, PT-CATE estimand, and smooth budget-allocation operator.  
**Next research task:** Reproduce the original report’s complete baseline package; implement and benchmark the specified smooth method against exact low-dimensional references.

**Contribution decision:** Primarily an empirically validated methodological extension of Frauen et al.; a globally convergent optimizer theorem is not a thesis requirement.

**Evidence convention:** DERIVED = algebra/proof checked in this revision; REPORT-TESTED = numerical result stated in the supplied report, not rerun here; PROPOSED = future protocol; UNVERIFIED = missing evidence or unresolved claim. The report’s executable script and CSV/JSON outputs were not attached to this chat. No report-only numerical diagnostic is presented as independently reproduced.

**Source recovery update:** The user supplied the later counterexample derivation on 30 October 2026. Section 16.6 independently checks and integrates it. Its algebra and a new finite-support diagnostic were verified in this revision; the original REPORT.md experiments remain report-only.

---

# 1. Research Objective and Motivation

## 1.1 Background

Frauen et al. (2025), *Treatment Effect Estimation for Optimal Decision-Making*, investigate the distinction between accurate estimation of conditional average treatment effects (CATEs) and optimal decision-making.

Their central result demonstrates that, when a CATE is estimated within a restricted function class, the function that minimizes squared CATE estimation error need not induce the treatment policy with the highest expected value within the corresponding policy class.

Their proposed Policy-Targeted CATE (PT-CATE) framework addresses this discrepancy by combining squared CATE estimation error with a policy-value objective.

The present thesis investigates an extension of this idea to retail promotional targeting under a monetary budget constraint and heterogeneous treatment costs.

Budget-constrained treatment allocation has already been investigated in the literature, including Bhattacharya and Dupas (2012), Zou et al. (2020), and *Treatment Allocation under Uncertain Costs*. The thesis does not aim to establish the cost-effectiveness priority rule as a new contribution.

Instead, it examines whether policy-targeted re-estimation of a restricted CATE representation improves budget-constrained treatment decisions.

## 1.2 Working research question

> Does adapting policy-targeted CATE estimation to budget-constrained treatment allocation improve the expected incremental outcome achieved by policies induced by restricted CATE representations, compared with conventional CATE estimation followed by the same budget-aware allocation procedure?

The key distinction is between:

1. Estimating treatment effects accurately.
2. Learning a treatment-effect representation that supports high-value budget-constrained decisions.
3. Implementing the resulting allocation for a finite customer cohort.

These are related but distinct problems.

## 1.3 Methodological principle

We will derive the mathematical target before specifying its numerical optimizer.

In particular, convexity or strong-duality results for unrestricted population allocation will not be transferred automatically to restricted, nonconvex neural-network learning.

---

# 2. Causal Setting and Policy Value

## 2.1 Observed data

Let

\[
(X,A,Y)\sim P,
\]

where:

- \(X\in\mathcal X\) denotes observed pre-treatment customer covariates.
- \(A\in\{0,1\}\) denotes treatment assignment.
- \(Y\in\{0,1\}\) denotes purchase response.

Let \(Y(1)\) and \(Y(0)\) be the potential outcomes under treatment and control.

The conditional average treatment effect is

\[
\boxed{
\tau(x)=\mathbb E[Y(1)-Y(0)\mid X=x].
}
\tag{1}
\]

The causal interpretation requires consistency, an appropriate no-interference assumption, treatment exchangeability, and positivity.

The randomized X5 RetailHero setting supports exchangeability under the documented randomization design, but the complete assumptions and assignment probabilities must still be verified.

## 2.2 Treatment policies

A population treatment policy is a measurable function

\[
\pi:\mathcal X\to[0,1].
\]

The value \(\pi(x)\) denotes the probability of treating a customer with characteristics \(x\).

The expected outcome under the policy is

\[
V_{\mathrm{abs}}(\pi)
=
\mathbb E[
\pi(X)Y(1)+(1-\pi(X))Y(0)
].
\]

Rearranging gives

\[
V_{\mathrm{abs}}(\pi)
=
\mathbb E[Y(0)]
+
\mathbb E[\pi(X)\tau(X)].
\]

Because the first term does not depend on the treatment policy, the relevant incremental policy value is

\[
\boxed{
V(\pi)=\mathbb E[\pi(X)\tau(X)].
}
\tag{2}
\]

The objective is to maximize incremental purchase response, not monetary profit.

For example, \(V(\pi)=0.01\) corresponds to ten additional expected purchases per 1,000 eligible customers, relative to treating nobody.

---

# 3. Treatment Costs and Budget Normalization

## 3.1 Expected treatment costs

Let

\[
c:\mathcal X\to(0,\infty)
\]

be a known or specified positive cost function.

The quantity \(c(x)\) represents expected incremental resources required to treat a customer with characteristics \(x\).

For the theoretical development, we assume costs are positive and integrable.

The X5 dataset does not directly provide the treatment-specific cost needed for the proposed monetary allocation analysis. Therefore, the thesis will initially use scenario-defined treatment costs rather than claim to estimate actual incremental campaign costs.

## 3.2 Proposed heterogeneous-cost function

We intend to use

\[
\boxed{
c_k(x_i)=
\bar c\left[
(1-k)+k\frac{h_i}{\bar h}
\right],
}
\tag{3}
\]

where:

- \(\bar c>0\) is the reference mean treatment cost.
- \(h_i\ge0\) is a historical basket-value feature.
- \(\bar h>0\) is the training-reference mean of that feature.
- \(k\) controls cost heterogeneity.

When \(k=0\), treatment costs are homogeneous:

\[
c_0(x)=\bar c.
\]

When \(k=1\),

\[
c_1(x)=\bar c\frac{h(x)}{\bar h}.
\]

If \(\bar h\) is defined using the same reference population, then

\[
\mathbb E_{\mathrm{ref}}[c_k(X)]=\bar c.
\]

The variance satisfies

\[
\operatorname{Var}_{\mathrm{ref}}(c_k(X))
=
\left(\frac{\bar c k}{\bar h}\right)^2
\operatorname{Var}_{\mathrm{ref}}(h(X)).
\]

Thus, \(k\) changes cost heterogeneity while holding the reference mean constant.

**Correction / adopted positivity rule:** Restrict \(0\le k\le1\) and define a pre-specified positive feature \(\widetilde h=\max(h,\epsilon_h)\), \(\epsilon_h>0\), with training-reference mean \(\overline{\widetilde h}\). Replace both numerator and denominator in Eq. (3) by these quantities. This preserves the reference mean exactly and ensures positive costs, including at \(k=1\). Choose the floor using training data only; alternatives and floor sensitivity are robustness checks. Applying a floor to costs after normalization would generally change their mean and require renormalization.

The economic validity of historical basket value as a cost proxy must also be discussed explicitly. The cost scenarios are hypothetical and are not experimentally identified from X5.

## 3.3 Budget normalization

We distinguish:

\[
B_N=\text{total monetary budget for a cohort of }N
\]

and

\[
\boxed{
b=\frac{B_N}{N}.
}
\tag{4}
\]

Here \(b\) denotes budget intensity per eligible customer.

The population budget condition is

\[
\mathbb E[c(X)\pi(X)]\le b.
\]

For an empirical sample of size \(m\), its sample analogue is

\[
\frac1m\sum_{i=1}^m c_i\pi_i\le b.
\]

This avoids assigning the same total monetary budget to datasets of different sizes.

The normalization permits consistent comparison across training, validation, test, and future deployment cohorts.

For example, reference budget intensities may be defined as

\[
b_q=q\bar c_{\mathrm{train}},
\]

where \(q\) is a budget-reference fraction.

Under heterogeneous costs, \(q\) is not generally equal to the fraction of customers treated.

---

# 4. Decision 1 — Population-Level Expected-Budget Allocation

**Decision: Adopted.**

The primary theoretical decision problem is

\[
\boxed{
\begin{aligned}
\text{(P1)}\qquad
\max_{\pi:\mathcal X\to[0,1]}
&\quad\mathbb E[\tau(X)\pi(X)]\\
\text{subject to}
&\quad\mathbb E[c(X)\pi(X)]\le b.
\end{aligned}
}
\tag{5}
\]

The policy is permitted to randomize.

The objective and constraint are linear in \(\pi\), and the unrestricted feasible policy set is convex.

This is the population counterpart of a fractional resource-allocation problem.

It differs from the finite-cohort binary allocation problem:

\[
\max_{z_i\in\{0,1\}}
\sum_{i=1}^N \tau_i z_i
\quad
\text{subject to}
\quad
\sum_{i=1}^N c_i z_i\le B_N.
\]

The latter is a 0–1 knapsack problem. Benefit-cost greedy ranking is not generally an exact solution to finite binary knapsack.

The finite-cohort allocation problem will be treated separately from the central population learning problem.

## 4.1 Lagrangian

For multiplier \(\lambda\ge0\),

\[
\begin{aligned}
\mathscr L(\pi,\lambda)
&=
\mathbb E[\pi(X)\tau(X)]\\
&\quad-\lambda\left(
\mathbb E[\pi(X)c(X)]-b
\right).
\end{aligned}
\]

Equivalently,

\[
\boxed{
\mathscr L(\pi,\lambda)
=
\mathbb E[
\pi(X)(\tau(X)-\lambda c(X))
]
+\lambda b.
}
\tag{6}
\]

For fixed \(\lambda\), unrestricted allocation is pointwise:

\[
\pi_\lambda(x)=
\begin{cases}
1,&\tau(x)-\lambda c(x)>0,\\
a(x)\in[0,1],&\tau(x)-\lambda c(x)=0,\\
0,&\tau(x)-\lambda c(x)<0.
\end{cases}
\]

For \(c(x)>0\), the priority score is

\[
\boxed{
\rho(x)=\frac{\tau(x)}{c(x)}.
}
\tag{7}
\]

The population optimum treats customers with sufficiently high positive benefit per unit cost, with boundary randomization when required.

## 4.2 Complementary slackness

At an appropriate primal-dual optimum,

\[
\lambda^*\ge0,
\]

\[
C(\pi^*)\le b,
\]

and

\[
\boxed{
\lambda^*(C(\pi^*)-b)=0.
}
\tag{8}
\]

The budget need not be exhausted.

If treating all customers with positive treatment effects is affordable, an optimal policy may leave budget unused.

We therefore retain the inequality constraint throughout the methodology.

## 4.3 Economic interpretation

If \(\tau(x)\) is measured in incremental purchase probability and \(c(x)\) in euros, then

\[
[\lambda]
=
\frac{\text{incremental purchase probability}}{\text{euro}}.
\]

Thus,

\[
\tau(x)-\lambda c(x)
\]

is dimensionally coherent.

The multiplier expresses marginal incremental response per unit budget, not monetary profit.

---

# 5. Decision 2 — Restricted Score Representation and Allocation Operator

**Decision: Adopted, with a specified tie-breaking convention.**

Let

\[
g\in\mathcal G
\]

be a restricted CATE-like representation.

Instead of jointly learning \(g\) and an independent decision cutoff, the policy is constructed using an estimated-benefit-optimal allocation operator.

Define

\[
\boxed{
\mathcal A_b(g)
\in
\arg\max_{\substack{\pi:\mathcal X\to[0,1]\\
C(\pi)\le b}}
\mathbb E[g(X)\pi(X)].
}
\tag{9}
\]

This operator maximizes estimated benefit \(g\), not the unknown true CATE \(\tau\).

The corresponding priority score is

\[
r_g(x)=\frac{g(x)}{c(x)}.
\]

## 5.1 Selected allocation convention

The operator is defined to select a policy satisfying:

- Treat all customers with benefit-cost ratios strictly above the positive allocation threshold.
- Do not treat customers below that threshold.
- Apply constant treatment probability on a positive-ratio boundary when fractional allocation is necessary.
- Do not treat customers with \(g(x)=0\) or \(g(x)<0\).
- Permit unused budget when no further positive estimated benefits are available.

The resulting allocation depends on \(g\), costs, budget intensity, and the population distribution.

The tie-breaking convention must be independent of the unknown true treatment effects.

## 5.2 Restricted induced policy class

The restricted policy class is

\[
\boxed{
\Pi_{\mathcal G,b}
=
\{\mathcal A_b(g):g\in\mathcal G\}.
}
\tag{10}
\]

This is not necessarily the same as every feasible threshold policy expressible using \(g\in\mathcal G\) with an arbitrary independent cutoff.

We restrict attention to policies produced by estimated-benefit-optimal budget allocation.

This corresponds to a realistic decision pipeline: learn a treatment-effect score, then allocate the available budget based on that score.

## 5.3 Implication for model restrictions

The function class \(\mathcal G\) restricts the CATE-like score, but not necessarily the final treatment policy to the same degree.

Even a constant positive \(g(x)\) can produce heterogeneous decisions under heterogeneous costs because

\[
r_g(x)=\frac{g(x)}{c(x)}.
\]

Therefore, representation complexity and policy complexity must be distinguished.

For experiments, deliberately restricted linear or segment-based score classes are preferred as primary settings, with a more flexible neural network as a robustness comparison.

---

# 6. Conventional CATE Estimation vs. Restricted Policy Optimization

## 6.1 Conventional CATE projection

Define squared CATE risk:

\[
R(g)=\mathbb E[(\tau(X)-g(X))^2].
\]

The conventional restricted CATE target is

\[
\boxed{
g_0^*
\in
\arg\min_{g\in\mathcal G}R(g).
}
\tag{11}
\]

Its budget-aware treatment policy is

\[
\pi_{0,b}=\mathcal A_b(g_0^*).
\]

## 6.2 Optimal induced policy

The best attainable policy value within the allocation-induced class is

\[
\boxed{
V_{\mathcal G,b}^*
=
\sup_{g\in\mathcal G}
V(\mathcal A_b(g)).
}
\tag{12}
\]

The budget-constrained projection gap is

\[
\boxed{
\Delta_{\mathcal G,b}
=
V_{\mathcal G,b}^*
-
V(\mathcal A_b(g_0^*)).
}
\tag{13}
\]

By definition,

\[
\Delta_{\mathcal G,b}\ge0.
\]

A strictly positive gap is established for the report’s finite-support restricted class: \(\Delta_{\mathcal G,b}=1/30\). See section 16.2. This is a stylized construction, not an X5 estimate.

---

# 7. Decision 3 — Proposed Budget-Constrained PT-CATE Estimand

**Decision: Adopted as the working population estimand; optimization remains unresolved.**

For \(\gamma\in[0,1]\), define

\[
\boxed{
\mathcal J_{\gamma,b}(g)
=
(1-\gamma)R(g)
-
\gamma V(\mathcal A_b(g)).
}
\tag{14}
\]

Equivalently,

\[
\boxed{
\begin{aligned}
\mathcal J_{\gamma,b}(g)
={}&
(1-\gamma)
\mathbb E[(\tau(X)-g(X))^2]\\
&-
\gamma
\mathbb E[
\tau(X)\mathcal A_b(g)(X)
].
\end{aligned}
}
\tag{15}
\]

The proposed population target is

\[
\boxed{
\mathcal J_{\gamma,b}^*
=
\inf_{g\in\mathcal G}
\mathcal J_{\gamma,b}(g).
}
\tag{16}
\]

A minimizer, if attained, is denoted \(g_{\gamma,b}^*\).

With heterogeneous-cost scenario \(k\), use

\[
\boxed{
\mathcal J_{\gamma,b,k}(g)
=
(1-\gamma)R(g)
-
\gamma V(\mathcal A_{b,k}(g)).
}
\tag{17}
\]

This is the proposed budget-constrained PT-CATE extension.

It should not be described as an established Frauen et al. result.

## 7.1 Limiting cases

For \(\gamma=0\),

\[
\mathcal J_{0,b}(g)=R(g).
\]

Thus, the method recovers conventional restricted CATE estimation.

For \(\gamma=1\),

\[
\mathcal J_{1,b}(g)
=
-V(\mathcal A_b(g)).
\]

Thus, it becomes pure policy optimization within the allocation-induced class.

For \(0<\gamma<1\), the objective trades off CATE approximation accuracy and budget-constrained policy value.

## 7.2 Why no additional budget penalty appears

The allocation operator already constructs a population-budget-feasible policy.

Therefore, the outer objective evaluates the original incremental policy value directly.

We do not additionally subtract a learned shadow-price penalty from that policy value.

This avoids conflating:

- The multiplier of the inner economic allocation problem.
- A multiplier for an outer constrained learning problem.
- A free neural decision-threshold parameter.

The resulting mathematical structure is a bilevel problem:

\[
\boxed{
\begin{aligned}
\inf_{g\in\mathcal G}
&\quad
(1-\gamma)R(g)
-\gamma V(\mathcal A_b(g))\\
\text{where }
\mathcal A_b(g)
&\in
\arg\max_{\substack{0\le\pi\le1\\C(\pi)\le b}}
\mathbb E[g(X)\pi(X)].
\end{aligned}
}
\tag{18}
\]

The inner problem is tractable through population priority allocation.

The outer problem is generally nonsmooth and nonconvex.

Strong duality for the inner allocation does not establish global optimality or convergence of a neural algorithm solving the outer problem.

---

# 8. Derived Population Properties

The following statements are mathematical consequences of the proposed definitions, subject to the specified existence and optimality qualifications.

## 8.1 CATE accuracy at the mixed optimum

Because \(g_0^*\) minimizes squared error,

\[
\boxed{
R(g_{\gamma,b}^*)\ge R(g_0^*).
}
\tag{19}
\]

Policy targeting may sacrifice CATE accuracy.

## 8.2 Policy-value improvement at a global optimum

For \(0<\gamma\le1\), if \(g_{\gamma,b}^*\) and \(g_0^*\) are global minimizers of their respective population objectives,

\[
\boxed{
V(\mathcal A_b(g_{\gamma,b}^*))
\ge
V(\mathcal A_b(g_0^*)).
}
\tag{20}
\]

For \(0<\gamma<1\), this follows from

\[
(1-\gamma)
[R(g_{\gamma,b}^*)-R(g_0^*)]
\le
\gamma
[V(\mathcal A_b(g_{\gamma,b}^*))
-
V(\mathcal A_b(g_0^*))].
\]

At \(\gamma=1\), the result follows directly from pure policy-value maximization.

**Qualification:** This is a population global-optimum property. It does not guarantee that estimated neural PT-CATE outperforms conventional CATE out of sample.

## 8.3 Correctly specified model class

If

\[
\tau\in\mathcal G,
\]

then \(\tau\) simultaneously minimizes squared CATE error and produces an unrestricted optimal population allocation.

Consequently, it is a global minimizer of the mixed objective for every \(\gamma\in[0,1]\).

For \(\gamma<1\), it is the unique squared-error solution up to almost-sure equivalence.

At \(\gamma=1\), other functions inducing the same optimal allocation may also minimize the objective.

Thus, the policy-targeting advantage should disappear at the population level when the CATE is correctly represented and conventional CATE estimation is globally optimal.

## 8.4 CATE-error bound on allocation regret

Let

\[
\pi^*=\mathcal A_b(\tau),
\qquad
\pi_g=\mathcal A_b(g).
\]

Then

\[
\boxed{
0\le
V(\pi^*)-V(\pi_g)
\le
\mathbb E[|\tau(X)-g(X)|]
\le
\sqrt{R(g)}.
}
\tag{21}
\]

The first inequality follows from unrestricted optimality of \(\pi^*\).

The second follows because \(\pi_g\) maximizes estimated benefit among budget-feasible policies.

The final inequality follows from Cauchy–Schwarz.

Accurate CATE estimation is therefore sufficient for low allocation regret, but accurate CATE estimation is not necessary for excellent allocation decisions.

This motivates policy-targeted learning.

---

# 9. Scaling of the Mixed Objective

The terms

\[
R(g)=\mathbb E[(\tau-g)^2]
\]

and

\[
V(\mathcal A_b(g))
\]

have different units and numerical scales.

A value \(\gamma=0.8\) does not automatically represent an economically meaningful 80/20 weighting.

A normalized alternative is

\[
\boxed{
\mathcal J_{\gamma,b}^{\mathrm{norm}}(g)
=
(1-\gamma)\frac{R(g)}{s_R}
-\gamma\frac{V(\mathcal A_b(g))}{s_V},
}
\tag{22}
\]

with fixed positive reference scales \(s_R,s_V\).

Fixed positive scaling preserves the attainable scalarized trade-off family after reparameterizing \(\gamma\), but changes its numerical interpretation.

**Working approach:** Retain Frauen's unnormalized mixture for methodological comparability, and investigate normalization as a robustness and numerical-conditioning issue.

Reference scales must be specified independently of the candidate fitted score and without using test outcomes.

---

# 10. The Optimization Challenge

The outer objective contains

\[
V(\mathcal A_b(g)).
\]

Changing \(g\) alters individual priority scores and the allocation threshold selected to satisfy the budget.

The exact allocation operator is generally nonsmooth because of ranking changes, threshold crossings, ties, and binding/slack transitions.

This prevents direct reliance on ordinary neural gradient descent.

We considered three broad approaches:

1. Differentiate through a smoothed budget-calibrated operator.
2. Recalibrate the threshold while omitting its derivative during score updates.
3. Optimize a fixed-shadow-price surrogate without differentiating through budget calibration.

None is established as a generally consistent solver of the original hard population objective. The implicit derivative is the correct reference gradient of the specified smooth objective; detachment is an ablation, with an analytical failure case in section 16.4.

For method development, we selected a differentiable allocation operator as the theoretical reference. This does not commit us to using implicit differentiation in the final algorithm.

---

# 11. Proposed Smoothed Allocation Operator

## 11.1 Motivation

A simple sigmoid,

\[
\sigma\left(
\frac{g(x)-\lambda c(x)}{T}
\right),
\]

does not automatically reproduce our selected allocation convention at zero scores or tied benefit-cost ratios.

We therefore derived a regularized allocation problem with a more explicit mathematical interpretation.

## 11.2 Cost-weighted entropy regularization

Let

\[
H(p)=-p\log p-(1-p)\log(1-p).
\]

For \(T>0\) and \(\delta>0\), define

\[
\boxed{
\begin{aligned}
\mathcal A_{b,T,\delta}(g)
=
\arg\max_{\substack{0\le\pi\le1\\C(\pi)\le b}}
\bigg\{
&\mathbb E[g(X)\pi(X)]\\
&-\delta\mathbb E[c(X)\pi(X)]\\
&+T\mathbb E[c(X)H(\pi(X))]
\bigg\}.
\end{aligned}
}
\tag{23}
\]

The regularizer has two components.

The entropy term encourages smooth interior treatment probabilities.

The positive cost penalty discourages the assignment of treatment when estimated incremental benefit is exactly zero.

The cost weighting in the entropy term is chosen so that customers with identical benefit-cost ratios receive identical treatment probabilities.

## 11.3 Solution of the regularized allocation problem

Introduce a multiplier \(\lambda\ge0\) for the budget constraint.

For fixed \(g\), the pointwise optimal treatment probability is

\[
\boxed{
\pi_{T,\delta,g,\lambda}(x)
=
\sigma\left(
\frac{g(x)/c(x)-\delta-\lambda}{T}
\right).
}
\tag{24}
\]

The calibrated multiplier satisfies

\[
\lambda\ge0,
\]

\[
\mathbb E[c(X)\pi_{T,\delta,g,\lambda}(X)]\le b,
\]

and

\[
\lambda
\left(
\mathbb E[c(X)\pi_{T,\delta,g,\lambda}(X)]-b
\right)=0.
\]

The smoothed allocation is then

\[
\boxed{
\mathcal A_{b,T,\delta}(g)(x)
=
\sigma\left(
\frac{r_g(x)-\delta-\lambda_{T,\delta}(g)}{T}
\right).
}
\tag{25}
\]

The temperature \(T\) and penalty \(\delta\) are measured in benefit-per-cost units in this formulation.

## 11.4 Budget calibration

**Endpoint correction:** Eqs. (24)–(26) with a finite multiplier apply for \(b>0\), finite scores/ratios and integrable positive costs. At \(b=0\), define the allocation to be identically zero; the logistic representation requires \(\lambda=+\infty\). For infinite support assume \(\mathbb E|g|<\infty\), \(\mathbb E[c]<\infty\), and sufficient differentiation regularity. With bounded \(\tau\), finite \(R(g)\) implies score integrability. For \(b\ge\mathbb E[c]\), calibration is slack.


Define

\[
S_{T,\delta,g}(\lambda)
=
\mathbb E\left[
c(X)
\sigma\left(
\frac{r_g(X)-\delta-\lambda}{T}
\right)
\right].
\]

For positive finite costs, \(T>0\), and suitable integrability assumptions, spending is continuous and strictly decreasing in \(\lambda\).

Define

\[
\boxed{
\lambda_{T,\delta}(g)=
\begin{cases}
0,
&S_{T,\delta,g}(0)\le b,\\
\text{unique positive solution of }S_{T,\delta,g}(\lambda)=b,
&S_{T,\delta,g}(0)>b.
\end{cases}
}
\tag{26}
\]

The smooth population allocation satisfies

\[
\boxed{
C(\mathcal A_{b,T,\delta}(g))\le b.
}
\tag{27}
\]

This is an exact population expected-budget feasibility property of the smooth randomized policy.

It does not establish feasibility after converting probabilities to finite binary assignments.

---

# 12. Convergence of the Smoothed Inner Allocation

Define estimated allocation value

\[
Q_g(\pi)=\mathbb E[g(X)\pi(X)].
\]

Let

\[
\pi^*=\mathcal A_b(g),
\qquad
\pi_{T,\delta}=\mathcal A_{b,T,\delta}(g).
\]

Using optimality of the regularized problem and the bound

\[
0\le H(p)\le\log2,
\]

we obtain

\[
\boxed{
0\le
Q_g(\pi^*)-Q_g(\pi_{T,\delta})
\le
(\delta+T\log2)\mathbb E[c(X)].
}
\tag{28}
\]

Therefore,

\[
Q_g(\pi_{T,\delta})
\longrightarrow
Q_g(\pi^*)
\]

as \(T,\delta\downarrow0\).

**Established:** The estimated allocation-value gap vanishes.

**Updated:** Selected-allocation convergence is established for fixed finite support under Eq. (29), as proved in section 16.1. It is not established here for arbitrary covariate distributions or scores varying with temperature.

This is a stronger requirement, especially at ties and zero-benefit regions.

## 12.1 Zero-benefit convention

For \(g(x)=0\) and a slack budget with \(\lambda=0\),

\[
\pi_{T,\delta}(x)
=
\sigma\left(-\frac{\delta}{T}\right).
\]

To obtain zero treatment probability in the limit, a sufficient schedule is

\[
\boxed{
\delta_T\downarrow0,
\qquad
\delta_T/T\to\infty.
}
\tag{29}
\]

The numerical choice of such a schedule has not yet been settled.

## 12.2 Tied priority scores

Cost-weighted entropy produces equal treatment probabilities for customers sharing the same benefit-cost ratio.

This matches the structure of the constant-randomization convention.

However, a full convergence proof must still address:

- Positive-score ties.
- Zero-score mass points.
- Binding and slack budgets.
- Degenerate score distributions.
- Convergence of the selected multiplier.
- Behavior when \(g\) varies with \(T\).

The finite-support cases are now covered by section 16.1. Infinite-support extensions and scores varying with temperature require separate arguments.

---

# 13. Differentiability of the Smoothed Operator

Let \(g=g_\theta\), where \(\theta\) denotes differentiable model parameters, and assume \(c(x)\) does not depend on \(\theta\).

Define

\[
r_\theta(x)=\frac{g_\theta(x)}{c(x)}
\]

and

\[
p_\theta(x)
=
\sigma\left(
\frac{r_\theta(x)-\delta-\lambda(\theta)}{T}
\right).
\]

Also define

\[
w_\theta(x)=p_\theta(x)(1-p_\theta(x)).
\]

## 13.1 Binding-budget multiplier gradient

For a locally binding budget,

\[
\mathbb E[c(X)p_\theta(X)]=b.
\]

Implicit differentiation gives

\[
\boxed{
\nabla_\theta\lambda(\theta)
=
\frac{
\mathbb E[c(X)w_\theta(X)\nabla_\theta r_\theta(X)]
}{
\mathbb E[c(X)w_\theta(X)]
}.
}
\tag{30}
\]

This requires a differentiable score parameterization, appropriate interchange of differentiation and expectation, and a locally regular calibration equation.

## 13.2 Smoothed allocation gradient

The treatment-probability gradient is

\[
\boxed{
\nabla_\theta p_\theta(x)
=
\frac{w_\theta(x)}{T}
\left[
\nabla_\theta r_\theta(x)
-
\nabla_\theta\lambda(\theta)
\right].
}
\tag{31}
\]

Consequently, for

\[
V_{T,\delta}(\theta)
=
\mathbb E[\tau(X)p_\theta(X)],
\]

the correct binding-budget gradient is

\[
\boxed{
\nabla_\theta V_{T,\delta}
=
\frac1T
\mathbb E\left[
\tau(X)w_\theta(X)
\left(
\nabla_\theta r_\theta(X)
-
\frac{
\mathbb E[cw_\theta\nabla_\theta r_\theta]
}{
\mathbb E[cw_\theta]
}
\right)
\right].
}
\tag{32}
\]

This is the gradient of the smoothed policy value, not the gradient of the original hard-allocation objective.

## 13.3 Slack-budget case

When the budget is strictly slack,

\[
\lambda(\theta)=0
\]

locally and

\[
\nabla_\theta\lambda(\theta)=0.
\]

Therefore,

\[
\boxed{
\nabla_\theta V_{T,\delta}
=
\frac1T
\mathbb E[
\tau(X)w_\theta(X)\nabla_\theta r_\theta(X)
].
}
\tag{33}
\]

At binding/slack transitions, the multiplier mapping may be nondifferentiable.

## 13.4 Implications for detached-threshold optimization

If the budget threshold is treated as fixed during differentiation, the resulting gradient omits

\[
\nabla_\theta\lambda(\theta).
\]

It is therefore not generally equal to the gradient of the smoothed budget-calibrated policy value.

One diagnostic example is a constant shift of every priority score under a binding budget:

\[
r_\theta(x)\mapsto r_\theta(x)+a.
\]

The calibrated multiplier shifts accordingly, leaving the allocation unchanged. The correct policy-value gradient in this direction is zero.

A detached-cutoff gradient need not be zero.

**Conclusion:** Differentiating through the cutoff is generally necessary for the exact gradient of the chosen smooth objective. Detachment is not a generally valid replacement: section 16.4 gives a directional allocation failure. Whether a separately specified alternative surrogate can learn effectively under additional conditions remains an empirical/theoretical question.

---

# 14. Smoothed Outer PT-CATE Objective

We define the proposed smooth outer objective:

\[
\boxed{
\mathcal J_{\gamma,b,T,\delta}(g)
=
(1-\gamma)R(g)
-
\gamma V(\mathcal A_{b,T,\delta}(g)).
}
\tag{34}
\]

The desired relationship is

\[
\mathcal J_{\gamma,b,T,\delta}(g)
\longrightarrow
\mathcal J_{\gamma,b}(g)
\]

as \(T,\delta\downarrow0\).

The inner estimated-value convergence bound alone does not establish this.

The reason is that the inner operator optimizes estimated benefit \(g\), while the outer objective evaluates the policy using true benefit \(\tau\).

## 14.1 Sufficient allocation convergence condition

Because \(Y\) is binary, \(|\tau(X)|\le1\).

Therefore,

\[
\boxed{
\begin{aligned}
&
|V(\mathcal A_{b,T,\delta}(g))
-
V(\mathcal A_b(g))|\\
&\quad\le
\mathbb E[
|\mathcal A_{b,T,\delta}(g)(X)
-
\mathcal A_b(g)(X)|
].
\end{aligned}
}
\tag{35}
\]

Convergence of treatment probabilities in \(L^1(P_X)\) is sufficient for convergence of true policy value for fixed \(g\).

## 14.2 Uniform surrogate approximation

A stronger desirable property is

\[
\boxed{
\sup_{g\in\mathcal G}
|\mathcal J_{\gamma,b,T,\delta}(g)
-\mathcal J_{\gamma,b}(g)|
\longrightarrow0.
}
\tag{36}
\]

If the approximation error is uniformly bounded by \(\varepsilon\), then a global minimizer of the surrogate has original objective excess of at most \(2\varepsilon\), assuming the relevant optimization values are finite.

This is an established implication of the *hypothesized uniform approximation condition*.

We have not established that the condition holds for our selected function class.

Potential obstacles include:

- Priority scores close to the budget boundary.
- Large mass points or ties.
- Changing priority rankings.
- Nonunique hard allocations.
- Flexible restricted-score models that approach degenerate score configurations.
- Nonuniformity as \(g\) varies with temperature.

Uniform approximation is false in general for the adopted class: section 16.3 gives a moving-score witness. A general outer-minimizer consistency guarantee is therefore unavailable; the report’s nonattainment example alone does not disprove a theorem assuming an attained hard minimum. Section 16.6 now gives a verified stronger counterexample with an attained, unique hard minimum: even exact global smooth minimizers can retain a nonvanishing hard-objective excess.

---

# 15. Remaining Statistical Methodology

The previous development assumes the true CATE \(\tau\) is known.

In real RCT data, it is not observed directly.

The intended empirical implementation will require estimated nuisance functions and a doubly robust pseudo-outcome.

For estimated response functions \(\widehat\mu_0,\widehat\mu_1\) and propensity \(\widehat e\), the DR pseudo-outcome takes the form

\[
\Gamma^{DR}
=
\widehat\mu_1(X)-\widehat\mu_0(X)
+
\frac{A-\widehat e(X)}
{\widehat e(X)(1-\widehat e(X))}
(Y-\widehat\mu_A(X)).
\]

Under the appropriate nuisance conditions, its conditional mean targets \(\tau(X)\).

However, the sign of an observation-level pseudo-outcome is not necessarily the sign of the conditional CATE.

For a fixed cost multiplier,

\[
Z_\lambda=\Gamma^{DR}-\lambda c(X)
\]

can target conditional cost-adjusted benefit, but does not provide a noise-free observation of it.

Frauen's adaptive-region detector will therefore require separate justification in the budgeted setting.

The empirical protocol is now specified in sections 17–18. Its implementation remains future work; surrogate limitations must be disclosed rather than treated as a reason to require a universal consistency theorem before any empirical investigation.

---

# 16. Finite-Support Progress and Evidence Audit

## 16.1 Fixed-score convergence — DERIVED

For fixed finite positive masses \(p_j\), positive finite costs \(c_j\), finite scores \(g_j\), and \(b\ge0\), if \(T\downarrow0\), \(\delta\downarrow0\), and \(\delta/T\to\infty\), the regularized allocation converges coordinatewise to the selected hard allocation. Both ordinary and cost-weighted L1 convergence follow, and hence true value converges for bounded effects.

Proof: the feasible finite-dimensional polytope is compact. Eq. (28) makes every accumulation point estimated-benefit-optimal. Equal ratios have equal regularized probabilities. At a positive binding cutoff, higher groups must be filled, lower groups excluded, and the budget fixes the common boundary fraction. Exactly affordable positive prefixes have the same limiting primal allocation even when the dual is nonunique. Affordable positive scores are filled; zero scores satisfy \(a_j\le\sigma(-\delta/T)\to0\), and negative scores vanish too. Thus every accumulation point is the same selected allocation. At zero budget the result is immediate by the endpoint extension.

Weaker schedules fail in general. For \(p=(1/3,1/3,1/3)\), \(c=(1,2,3)\), \(g=(1,0,-1)\), \(b=1\), the budget is eventually slack. With \(\tau_2=0.8\), schedules \(\delta=T\) and \(\delta=T^2\) yield limiting zero-score probabilities \(\sigma(-1)\) and \(1/2\), and true-value errors \(0.8\sigma(-1)/3\approx0.0717177\) and \(2/15\). Estimated-value error still vanishes. A fixed positive delta can suppress genuinely positive scores. These are selection failures, not violations of Eq. (28).

## 16.2 Positive projection gap and nonattainment — DERIVED

Use the binary-compatible population

| Type | Mass | Cost | True effect | Restricted score |
|---|---:|---:|---:|---|
| 1 | 1/3 | 1 | 0.6 | \(0.2(1+\theta)\) |
| 2 | 1/3 | 2 | 0.8 | \(0.4(1-\theta)\) |
| 3 | 1/3 | 3 | 0.2 | \(-0.2\) |

Take \(\theta\in[0,0.9]\), \(b=0.5\), \(\gamma=0.8\). Bernoulli conditional means \(\mu_0=(0.1,0.1,0.1)\), \(\mu_1=(0.7,0.9,0.3)\) realize these effects.

At zero the selected allocation is \((1/2,1/2,0)\), with value \(7/30\). For every positive theta it is \((1,1/4,0)\), with value \(4/15\). Direct substitution yields

\[
R(\theta)=\frac{12+4\theta+5\theta^2}{75},\qquad
J(0)=-\frac{58}{375},
\]
\[
J(\theta)=-\frac{68}{375}+\frac{4\theta}{375}+\frac{\theta^2}{75},\quad\theta>0.
\]

The CATE projection is zero, and \(\Delta=1/30\). These identities were independently checked with exact rational arithmetic in this revision. The hard infimum is \(-68/375\) and is not attained. Compactness of the parameter domain is insufficient because the selected hard objective is discontinuous and not lower semicontinuous here.

For \(\delta=\sqrt T\), each smooth objective is continuous and attains a minimum. A competitor \(\theta=\sqrt T\) approaches the better allocation and the hard infimum. On every separated interval \([\epsilon,0.9]\), uniform convergence holds with a positive hard excess. Thus exact smooth global minimizers tend to zero and are positive for small T. Their limiting parameter has worse hard value. This is a limit-point pathology, **not** a counterexample with an attained hard global minimizer.

## 16.3 Nonuniformity and report numerical results

For \(\theta_T=T^2>0\) with \(\delta=\sqrt T\), the ratio gap is \(0.4T^2=o(T)\). The smooth allocation tends to \((1/2,1/2,0)\); the hard allocation at every member of the sequence is \((1,1/4,0)\). Consequently

\[
\lim\max_j|a_{T,j}-a_{\mathrm{hard},j}|=1/2,\qquad
\lim|J_T(\theta_T)-J(\theta_T)|=2/75.
\]

This DERIVED witness disproves general uniform convergence despite fixed-score convergence. Ties must be represented as separate points in plots; connecting across them as if the objective were continuous is misleading.

The following are REPORT-TESTED, not rerun in this revision:

| Report diagnostic | Reported outcome | Interpretation |
|---|---|---|
| 300 hard allocation vs enumerated fractional-LP checks | maximum value discrepancy 4.44e-16 | implementation evidence |
| 900 smooth allocations | bounds, budget and complementarity checked | finite numerical checks |
| 300 regret checks; 528 allocation rows | Eqs. (21), (28), (35) checked | not universal proofs |
| Implicit allocation / outer derivatives | errors 4.07e-10 / relative 2.38e-8 | finite-difference agreement |
| Common-ratio shift | allocation change below 1e-15 | invariance diagnostic |
| Continuous search, T=1e-6 | theta approximately 3.902e-5; smooth J approximately -0.18133289045 | grid plus basin refinement, no global certificate |
| Same score, hard J | approximately -0.18133291713 | compare with hard J(0) approximately -0.15466666667 |

Positive controls in the report: restricting theta to [0.1,0.9] supplies separation and alignment; restricting to {0,0.1,0.4,0.9} permits exact enumeration. Neither implies consistency for a flexible neural class.

## 16.4 Threshold derivatives and comparator failures — DERIVED / REPORT-TESTED

Eqs. (30)–(33) remain correct under their regularity assumptions. At theta=0 in section 16.2 and small T, \(d\lambda/d\theta\to-1/15\). Leading loss derivatives are

\[
J'_{T,\mathrm{implicit}}(0)=4/375-4/(1125T)+\text{exponentially small terms},
\]
\[
J'_{T,\mathrm{detached}}(0)=4/375+1/(375T)+\text{exponentially small terms}.
\]

These coefficients follow directly from \(a_1,a_2\to1/2\), \(r'_1=0.2\), \(r'_2=-0.2\), and Eq. (30), and were checked algebraically. Descent with the implicit derivative moves toward positive theta; projected detached descent at zero stays at zero. The report’s 8,000-step diagnostic confirms this for its tested starts and temperatures; it does not certify optimizer convergence.

The report specifies a fixed effective ratio cutoff comparator \(a_i=\sigma((g_i/c_i-0.2)/T)\), evaluated using true welfare and the same risk mixture. Its first two probabilities sum to one, increasing theta reduces its welfare and increases its risk, so its minimizer is zero. Hard deployment then yields 7/30 rather than 4/15. This one comparator is not every fixed-price method. It need not satisfy the training budget. The envelope theorem for estimated inner benefit does not cancel the multiplier derivative in the distinct outer true-welfare objective.

## 16.5 Corrections and numerical cautions

- The report supersedes an earlier numerical population with true effects (3,4,1), which is incompatible with binary outcomes. The valid population is (0.6,0.8,0.2). Rescaling effects changes squared-risk/value balance; keep neither old losses nor minimizers without recomputation.
- No finite logistic multiplier yields exact zero spending: use the zero-budget extension in section 11.4.
- Saturated probabilities can hide exponentially small residual signs. The report describes a stable log-space gain/loss calibration fix. Its code is not available here; implementation verification remains pending.
- In the exactly affordable case p=(1/3,1/3,1/3), c=(1,2,3), g=(0.2,-0.2,-0.6), b=1/3, hard dual multipliers form [0,0.2]. For delta=sqrt(T), balancing leading tails yields \(\lambda=0.05-\sqrt T+(T/2)\log2+o(T)\). This algebra checks out, but numerical agreement is report-only. Primal convergence does not require selecting an arbitrary hard dual representative.
- Ties mean exact equal ratios in the report’s numerical implementation; tolerance-based grouping changes the operator. Document grouping rules explicitly and test near ties separately.
- A naive formula with no cost division in the sigmoid corresponds to a different entropy regularizer. Retain the ratio-based Eq. (24) for cost-weighted entropy.

## 16.6 Exact-global-minimizer counterexample — DERIVED and independently CHECKED

**Source:** User-supplied continuation of the conversation, received 30 October 2026. This is a different population from section 16.2. Independent exact-rational algebra and finite-support diagnostics are saved in `verify_exact_minimizer_counterexample.py`, `exact_minimizer_verification.json` and `exact_minimizer_checks.csv`. The numerical searches are diagnostics; the global-minimizer conclusion follows from the proof below, not a numerical global certificate.

Take three equal masses, unit costs, true effects \(\tau=(1/5,4/5,1/5)\), budget \(b=1/3\), \(\gamma=4/5\), and

\[
g_\theta=(1/2,1/2-\theta,1/2-2\theta),\qquad
\theta\in[0,1/20],\qquad\delta_T=\sqrt T.
\]

Binary potential-outcome compatibility follows from control means (0.1,0.1,0.1) and treatment means (0.3,0.9,0.3). The class is a compact continuous affine family and all scores are positive. The schedule satisfies the fixed-score convergence conditions.

**Hard target.** At theta=0 the selected tie policy is (1/3,1/3,1/3), with true value 2/15. For every theta>0 the strict ordering induces (1,0,0), with value 1/15. Expanding the squared risk gives

\[
R(\theta)=\frac9{100}-\frac\theta5+\frac53\theta^2,
\qquad J(0)=-\frac{133}{1500},
\]
\[
J(\theta)-J(0)=\frac4{75}-\frac\theta{25}+\frac{\theta^2}{3},\qquad\theta>0.
\]

Its derivative on the positive interval is \(-1/25+2\theta/3<0\). Therefore the **tight** lower bound across that interval is

\[
J(\theta)-J(0)\ge\frac{313}{6000}\approx0.0521667>0.
\]

The supplied bound 0.051 is valid but conservative. Thus the hard objective attains a unique global minimum at theta=0; nonattainment cannot explain this example. The conventional risk minimizer on the specified interval is theta=1/20 (the unconstrained risk vertex is 3/50), whose hard policy is inferior. This additional projection check does not enter the consistency disproof.

**Smooth target and exact global solutions.** For every \(0<T<4/25\), \(g_j-\sqrt T>0\) for all types and all theta in the domain, so uncalibrated probabilities exceed 1/2 and the budget binds. Hence \(p_1+p_2+p_3=1\). At theta=0 calibration gives \(p_j=1/3\) exactly, with

\[
\lambda_T(0)=1/2-\sqrt T+T\log2>0.
\]

Using the implicit derivative and \(g'=(0,-1,-2)\),

\[
\lambda_T'(0)=-1,\quad
(p_1',p_2',p_3')=\left(\frac2{9T},0,-\frac2{9T}\right),\quad V_T'(0)=0.
\]

These are exact identities in this binding regime, not merely leading asymptotics. The equal effects of types 1 and 3 cancel their welfare derivatives. Therefore the right derivative is

\[
J_T'(0)=(1-\gamma)R'(0)-\gamma V_T'(0)=-1/25<0.
\]

For each positive T in this regime the calibrated smooth objective is continuous on the compact parameter interval, so an exact global minimum exists. Its negative right derivative at zero excludes zero from the argmin. **Every exact global smooth minimizer** \(\theta_T\) is positive. Consequently, independently of which smooth minimizer is chosen,

\[
J(\theta_T)-\min_\theta J(\theta)\ge313/6000,
\qquad
V(\mathcal A_b(g_0))-V(\mathcal A_b(g_{\theta_T}))=1/15.
\]

Thus global optimality for the smooth objective does **not** guarantee hard-objective consistency, even with an attained unique hard optimum, a valid temperature/penalty schedule, exact implicit gradients, no statistical noise and no neural optimization difficulty. This refutes a general claim that exact global smooth solutions must have vanishing excess under hard deployment. It also defeats an unrestricted guarantee covering all approximate minimizers, because exact minimizers are included in that set. It does not imply every approximate solution or every alternative surrogate must fail.

**Parameter convergence is different.** In this construction all exact smooth global minimizers nevertheless satisfy \(\theta_T\to0\). For any fixed epsilon>0, the interval [epsilon,1/20] has a uniformly separated top score. Under binding calibration the common delta is absorbed into lambda; lower-type probabilities vanish uniformly as T decreases. Thus \(J_T\to J\) uniformly there. Its hard loss is separated above J(0) by at least 313/6000, whereas \(J_T(0)=J(0)\) exactly. No global smooth minimizer can remain in that interval. This proves convergence of all minimizing parameters to the unique hard-optimal parameter, **despite** persistent hard loss at every finite-T minimizing parameter. In fact

\[
\lim_{T\downarrow0}[J(\theta_T)-J(0)]=4/75.
\]

Therefore this example disproves objective consistency after hard deployment; it does **not** disprove convergence of minimizing parameters to the hard argmin. Evaluating the hard policy at the limiting parameter and taking the limit of hard policies at positive parameters are different operations.

**New numerical diagnostic, independently RUN 30 October 2026.** Solve calibration using centered logits \((a+\theta/T,a,a-\theta/T)\), with their sigmoid sum equal to one. This avoids subtracting nearly equal raw scores. A golden-section search on the stated small diagnostic interval finds:

| T | Positive numerical theta | Smooth loss | Hard excess |
|---|---:|---:|---:|
| 0.01 | 5.02085e-4 | -0.08867670765 | 0.05331333398 |
| 0.001 | 5.06211e-6 | -0.08866676791 | 0.05333313086 |
| 0.0001 | 5.06284e-8 | -0.08866666768 | 0.05333333131 |

The zero-parameter smooth loss is -0.08866666667. The positive smooth improvement shrinks rapidly, while the hard excess approaches 4/75. Double precision eventually cannot reliably distinguish the tiny smooth improvement, so the analytic derivative/existence proof is essential. The search is not a uniqueness or global numerical certificate. The script also checks risk/loss identities with rational arithmetic, budget residuals and positive multipliers. This new run is separate from the original report’s unrerun numerical results.

**Implication for the thesis:** keep the empirically tested soft method as a precisely declared surrogate; evaluate the intended hard deployment directly and treat tie/near-tie sensitivity as a core robustness analysis. Rounding scores, imposing margins, or deploying probabilities can alter this example but change the procedure and require explicit specification and evaluation. Correct implicit differentiation fixes the smooth gradient; it cannot repair this hard-target selection instability by itself.

---

# 17. Contribution, Research Decisions, and Statistical Target

The thesis contribution is an **empirically validated methodological extension of Frauen et al.** to a budget-aware score-to-policy pipeline with heterogeneous scenario costs. The methodological contribution must include a specified target, a valid and reproducible training objective, and honest evaluation against matched alternatives. A theorem establishing global neural optimization is not required. A negative or conditional empirical result can answer the research question, provided the method and comparisons are valid; improvement is a hypothesis, not a promised outcome.

Retain the exact hard target and tie convention for diagnosis and deployment comparisons. Use the calibrated entropy surrogate with implicit differentiation as the reference **smooth method**, without claiming it solves the hard target consistently. Detached and fixed-cutoff variants are explicit ablations. A target that changes tie behavior, imposes margins, or uses soft deployment is a different declared research choice; do not silently substitute it.

For a fixed policy, estimate incremental value on held-out randomized data by

\[
\widehat V(\pi)=\frac1{n_{test}}\sum_i\pi(X_i)\widehat\Gamma_i^{DR},
\quad
\widehat C(\pi)=\frac1{n_{test}}\sum_i c(X_i)\pi(X_i).
\]

Use the DR pseudo-outcome from section 15, with nuisance predictions generated without the evaluated observation’s outcome. If the randomization propensity is known, use the documented design probability (including any strata), not a guessed 0.5. Under conditional unbiasedness of the pseudo-outcome,

\[
\mathbb E[(\Gamma-g(X))^2]
=R(g)+\mathbb E[\operatorname{Var}(\Gamma\mid X)],
\]

for candidate-independent nuisance construction and finite second moments. Thus pseudo-outcome regression targets the risk up to a constant; noisy pseudo-outcome signs do not identify individual CATE signs. Cross-fitting mitigates nuisance overfitting but does not make model selection on test outcomes valid. Estimated nuisances can leave bias; no finite-sample unbiasedness is asserted automatically.

**Experimental selection caveat:** the X5 eligible customer sample, campaign period, communication channel and outcome definition define the evaluation population. RCT assignment within it does not identify effects for excluded customers, other periods, or different offers. Confirm assignment mechanism, eligibility, compliance, attrition, missing outcomes, customer duplication, feature timestamps and any interference. The accessible organizer page confirms communication/purchase columns and pre-communication histories; it does not establish the full randomization protocol or known assignment probabilities. Treat “X5 RCT” as a design claim awaiting documentation before causal interpretation.

**Finite-cohort caveat:** expected feasibility under a population distribution is not realized binary budget feasibility. Independent Bernoulli implementation can overspend. A deterministic binary cohort requires a separate knapsack/rounding rule; ratio greedy is not generally exact. Report its feasibility and welfare loss separately. Recalibrating a cutoff on held-out covariates only yields a cohort-dependent allocation; distinguish that estimand and its uncertainty from a frozen population policy.

---

# 18. Prioritized Stage-Gated Research Roadmap — PROPOSED

Proceed in order. Save configurations, splits, seeds, source hashes, environment details, failures and decision records at every gate. Set practical tolerances and primary comparisons before observing final test outcomes.

| Priority / stage | Work and required evidence | Gate / decision |
|---|---|---|
| P0 / A: reproducible exact baseline | Recover experiment.py, verification.json and CSVs; rerun report. Implement an independent fractional-LP/enumeration oracle with explicit ties and slack/zero budgets. For low-dimensional linear/segment classes partition score-order/sign cells and inspect boundaries and one-sided infima, using analytic minima or certified bounds where possible. | Reproduce tables within declared precision, independently confirm allocations, distinguish attained minima from infima and numerical searches. Do not label dense grids “exact global.” |
| P0 / B: stronger counterexample — completed | Section 16.6 independently checks the supplied derivation and runs a new finite-support diagnostic. Hard argmin exists uniquely; every exact smooth minimizer has persistent hard excess, while parameters converge to zero. | Verified disproof of general hard-deployment objective consistency. Preserve as a regression diagnostic; extend margin/tie robustness without blocking the empirical stages. |
| P1 / C: valid soft method | Implement Eqs. (23)–(34), inequality calibration and implicit derivatives, stable residuals and transition handling. Verify finite differences away from transitions, shift invariance, ties, gradients and budget residuals. Warm starts/multistart/annealing are numerical strategies, not global guarantees. | Freeze a precisely specified smooth loss and training protocol. Compare achieved smooth and hard loss against the exact low-dimensional baseline; disclose optimization gaps. |
| P1 / C ablations | Same model with gamma=0; implicit vs detached; explicitly defined fixed cutoff; optional unweighted entropy as a different operator; temperature/delta schedules and score margins. | Confirm which change explains any gain. Reject invalid gradients as reference implementations while retaining labeled diagnostics. |
| P1 / D: synthetic experiments | Known Bernoulli response DGPs with correct specification and controlled misspecification, positive/negative/zero effects, ties/near ties, cost-effect correlation, varying budgets, sample size and nuisance difficulty. Separate population-oracle, estimated-nuisance and finite-sample optimization errors. | Show correct controls, reliable budget behavior and interpretable comparisons across independent replicates. Tune on validation only; failure to improve is a reportable result. |
| P2 / E: X5 design/data gate | Verify trial documentation, assignment probabilities and unit; create only pre-treatment features; audit joins/eligibility/missingness and freeze customer-level split. Use the labeled sample for held-out evaluation; do not assume competition test labels are available. | Proceed with causal claims only if design/identification assumptions are supportable. Otherwise restrict claims and seek an appropriate documented RCT. |
| P2 / F: cross-fitting and held-out DR OPE | Fit nuisances out of fold on development data; train second-stage scores using those pseudo-outcomes. Select gamma, temperatures, floors, complexity and early stopping on validation folds. Freeze policy and calibration before final labeled test outcomes are used. Fit evaluation nuisances without test outcomes, or use a properly nested out-of-fold evaluation protocol. | Compare paired held-out incremental DR value and spend. Keep validation-selected estimates separate from final estimates; no test-driven retuning. |
| P2 / G: robustness and uncertainty | Repeated training splits/seeds, cost floors/scenarios and scale normalization, budget grid, misspecification/capacity, nuisance alternatives, calibration sample size, near ties and finite-cohort implementation. | Report effect sizes and uncertainty, including failures and unused budget; distinguish conditional policy uncertainty from retraining uncertainty. |

## 18.1 Model matching and comparators

The primary contrast is conventional restricted CATE followed by the **same** hard allocator versus budget PT-CATE with the same score class, features, nuisance estimates, splits, initialization opportunities, training/tuning resources and deployment rule. Use linear or segment models first, then a matched neural model. Include a strong flexible CATE-plus-allocator benchmark to test whether representation flexibility removes the need for targeting. Include a direct policy learner if feasible, reporting differences in expressiveness rather than implying equal model classes. Preserve the original Frauen method as a comparator only after verifying its implemented objective; adding budget allocation changes its pipeline. Audit the supplied source archive before reusing it; no repository-level audit was completed in this revision.

## 18.2 Budget and cost scenario normalization

Use \(b_q=q\bar c_{train}\), \(B_N=Nb_q\), with a predeclared budget grid and the positive normalized cost construction in section 3.2. Freeze reference transforms, mean and floors from training data. Do not normalize each validation/test set to its own observed mean for the primary analysis: that changes the scenario. Report actual split means, spend, unused budget, treated fraction, and incremental purchases per 1,000 eligible customers. q is a budget-reference fraction, not a treated fraction.

Compare homogeneous and heterogeneous costs at equal reference mean; vary cost-effect association separately from heterogeneity in synthetic DGPs. Basket value is a scenario proxy, not measured communication cost or profit. If changing monetary units by a factor a, transform c and b by a and ratio temperatures, delta and multipliers by 1/a; otherwise the soft operator changes. Risk/value scaling also changes the effective gamma: report raw and fixed-reference-normalized objectives, with scales selected without test outcomes.

## 18.3 Robustness and uncertainty protocol

Primary endpoint: paired difference in held-out DR incremental value at predeclared budget/cost settings. Also report risk/PEHE only where true tau is known (synthetic), budget residual, treated fraction, regret to an appropriate oracle, runtime and optimization variability. X5 cannot supply observed individual CATE error. AUUC/Qini can be secondary ranking summaries; they do not replace monetary-budget value.

For a fixed learned policy with frozen calibration and independent evaluation observations, use the paired contributions \((\pi_1(X_i)-\pi_0(X_i))\widehat\Gamma_i\) for standard errors under justified sampling/nuisance conditions. Respect stratification or clustering in the trial. If calibration uses the evaluation cohort, account for that dependence with a justified cohort-level procedure rather than claiming the ordinary fixed-policy formula applies. For uncertainty in the full learning procedure, resample/refit the whole training, tuning and calibration pipeline or use repeated independent synthetic datasets; merely resampling final predictions does not capture training uncertainty. Predeclare primary settings and handle multiple budgets/scenarios with simultaneous inference or an explicitly exploratory presentation. Report intervals and practical magnitudes rather than only significance.

---

# 19. Sources, Version History, and Decision Log

## 19.1 Source register

0. **New independent check:** `verify_exact_minimizer_counterexample.py`, standard-library Python; exact rational algebra plus numeric diagnostics saved in `exact_minimizer_verification.json` and `exact_minimizer_checks.csv`, run 30 October 2026.
1. **Original project memorandum:** `../sources/Budget_Constrained_PTCATE_Memorandum.md`, v0.1, 26 October 2026. Read-only; sections 1–15 preserved except explicit corrections/status updates. Its earlier literature references are retained as inherited background, not newly audited theorem attributions.
2. **Finite-support report:** [reference/REPORT.md](reference/REPORT.md), “Recomputed 26 October 2026,” retrieved from the referenced chat attachment on 30 October 2026. Report source code and supporting numeric artifacts unavailable in this chat. Numerical outcomes are report-tested, not independently reproduced.
3. **Conversation:** [Branch · Rewrite budget constrained methodology](chatgpt-conversation://6adf811f-35d8-83eb-b9dc-12cf44e057b3), retrieved 30 October 2026. Assistant mathematical replies initially returned only as content-reference markers. The user subsequently pasted the full stronger counterexample into this chat on 30 October 2026; that supplied text is the source for section 16.6 and was independently checked.
4. Frauen, D., Melnychuk, V., Schweisthal, J., van der Schaar, M., and Feuerriegel, S. (2025), [Treatment Effect Estimation for Optimal Decision-Making, arXiv v2](https://arxiv.org/abs/2505.13092v2). Metadata/abstract checked 30 October 2026; supports the CATE/decision mismatch and adaptively smoothed PT-CATE context, not the new budgeted consistency claims.
5. X5 organizer, [RetailHero task data](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data), accessed 30 October 2026. Confirms pre-communication purchases, communication flag and subsequent purchase target; complete randomization protocol remains unverified.
6. [scikit-uplift X5 dataset documentation, v0.4.1](https://www.uplift-modeling.com/en/v0.4.1/api/datasets/fetch_x5.html), accessed 30 October 2026, corroborates schema and feature timing. This is a versioned schema reference, not advice to install that package version.
7. Inherited references: Bhattacharya & Dupas (2012), *Inferring Welfare Maximizing Treatment Assignment under Budget Constraints*; Zou et al. (2020), *Heterogeneous Causal Learning for Effectiveness Optimization in User Marketing*; supplied *Treatment Allocation under Uncertain Costs* and Cerulli (2026), *Optimal Policy Learning under Budget and Coverage Constraints*. Project PDFs remain available as reference material; their results have not been extended to this bilevel objective by attribution.

## 19.2 Change log

| Version / date | Changes |
|---|---|
| 0.1 / 26 October 2026 | Original population target, selected allocation, entropy operator and gradient derivations. |
| 0.3 / 30 October 2026 | Recovered user-supplied exact-global-minimizer construction; verified all core algebra and global argument; sharpened bound to 313/6000 and proved limiting excess 4/75; distinguished parameter convergence from hard-deployment objective consistency; added independently run finite-support diagnostic and retained v0.2 snapshot. |
| 0.2 / 30 October 2026 | Preserved foundational sections; integrated finite-support proof, projection gap, nonattainment, nonuniformity and comparator failures; separated report numerical evidence from independent algebra; fixed zero-budget endpoint, integrability and cost positivity; adopted empirical contribution and stage-gated roadmap; documented missing later counterexample source. |

## 19.3 Concise decision log

- **Adopted:** empirical methodological extension of Frauen et al.; positive findings are hypotheses to test.
- **Retained:** hard expected-budget target, common-probability positive ties, exclusion of zero scores, unused-budget permission, and finite-cohort distinction.
- **Reference implementation:** calibrated cost-weighted entropy surrogate with implicit gradient; no general hard-target consistency or global optimizer guarantee.
- **Verified analytically:** finite-support fixed-score convergence; projection gap 1/30; report hard-loss formulas; nonattainment; moving-score nonuniformity; leading implicit/detached derivatives; dual tail formula.
- **Report-only:** numerical diagnostic counts, errors, search results and gradient-training trajectories; reproduce after recovering code and outputs.
- **Verified stronger counterexample:** attained unique hard optimum; every exact smooth global minimizer incurs hard excess at least 313/6000, tending to 4/75, even though its parameter converges to the hard optimum.
- **Next gate:** reproduce the original report’s complete low-dimensional baseline, then freeze and benchmark the smooth implementation before synthetic and X5 evaluation. The stronger construction’s verification gate is complete.
