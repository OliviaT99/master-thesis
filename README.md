# Policy-Targeted CATE Estimation for Retail Promotional Targeting

Master's thesis implementation.

## Research question

Among promotional-targeting decisions under a fixed treatment budget,
does a purely predictive response model recover the same targeting
decisions as causal and policy-targeted CATE estimators?

## Dataset

X5 RetailHero uplift-modelling dataset.

## Experimental arms

1. Purely predictive response model
2. Restricted DR-CATE, gamma = 0
3. Restricted PT-CATE, gamma > 0

## Main extension

The original PT-CATE decision boundary is replaced by a $x fixed treatment budget.

## Environment

Python virtual environment: `.venv`
Jupyter kernel: `Master Thesis (.venv)`