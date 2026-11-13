"""Stage-1 nuisance models and DR pseudo-outcomes (memo section 15 / 17).

* mu_a(x) = P(Y=1 | X=x, A=a), fitted separately per arm with gradient boosting.
* Training pseudo-outcomes are CROSS-FITTED (K folds): each row's Gamma uses
  nuisances that never saw that row's outcome.
* Validation/test pseudo-outcomes use nuisances fitted on the full training split
  only, so no evaluation outcome enters any fitted model.
* The propensity is the RCT assignment probability, not estimated. The value
  0.4998 is the empirical treated share; the documented design probability is
  still unverified (memo section 17), so it is passed explicitly.

mu_1 doubles as the Arm 1 "purely predictive" score P(Y=1 | X, A=1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

DEFAULT_LEARNER = HistGradientBoostingClassifier(
    learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
    max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
    random_state=0,
)
CLIP = 1e-3  # keeps log-loss finite; irrelevant for DR with known propensity


def dr_pseudo_outcome(y, a, mu1, mu0, e):
    """Gamma = mu1 - mu0 + (A - e) / (e (1 - e)) * (Y - mu_A)."""
    mu_a = np.where(a == 1, mu1, mu0)
    return mu1 - mu0 + (a - e) / (e * (1 - e)) * (y - mu_a)


@dataclass
class NuisanceFit:
    propensity: float
    learner: object = field(default_factory=lambda: clone(DEFAULT_LEARNER))
    n_folds: int = 5
    seed: int = 0

    def _fit_arm(self, X, y, a, arm):
        m = clone(self.learner)
        m.fit(X[a == arm], y[a == arm])
        return m

    @staticmethod
    def _predict(m, X):
        return np.clip(m.predict_proba(X)[:, 1], CLIP, 1 - CLIP)

    def fit_crossfit(self, X, y, a):
        """Out-of-fold mu0, mu1 and Gamma on the training split; then refit on all of it."""
        X, y, a = np.asarray(X), np.asarray(y), np.asarray(a)
        mu0, mu1 = np.empty(len(y)), np.empty(len(y))
        strata = 2 * a + y
        for tr, te in StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed).split(X, strata):
            mu1[te] = self._predict(self._fit_arm(X[tr], y[tr], a[tr], 1), X[te])
            mu0[te] = self._predict(self._fit_arm(X[tr], y[tr], a[tr], 0), X[te])
        self.m1_, self.m0_ = self._fit_arm(X, y, a, 1), self._fit_arm(X, y, a, 0)
        return dict(mu1=mu1, mu0=mu0, gamma=dr_pseudo_outcome(y, a, mu1, mu0, self.propensity))

    def evaluate(self, X, y, a):
        """Nuisances fitted on train only -> pseudo-outcomes on a held-out split."""
        X, y, a = np.asarray(X), np.asarray(y), np.asarray(a)
        mu1, mu0 = self._predict(self.m1_, X), self._predict(self.m0_, X)
        return dict(mu1=mu1, mu0=mu0, gamma=dr_pseudo_outcome(y, a, mu1, mu0, self.propensity))
