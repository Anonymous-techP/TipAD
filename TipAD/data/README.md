# Data

TipAD is evaluated on **TSB-AD-M**, the multivariate track of the TSB-AD benchmark.

```
data/
  File_List/
    TSB-AD-M-Tuning.csv           # the tuning split (hyperparameter selection only)
    TSB-AD-M-Eva.csv              # the eval split
  benchmark_results/
    multi_mergedTable_VUS-PR.csv  # official VUS-PR for the 30 classical baselines
    leaderboard/
      Multi_*.csv                 # official VUS-PR for recent SOTA baselines
  TSB-AD-M/                       # <- you add this, see below
```

The baseline results are consistent with the official TSB-AD results.

## Get the raw series

Download: [TSB-AD-M.zip](https://www.thedatum.org/datasets/TSB-AD-M.zip)

Unzip it into `data/`. That gives you `data/TSB-AD-M/` with 200 CSV files exactly what the experiments need.
