# TipAD

Code and data for reproducing the main results of **TipAD**.

## Repository Layout

```
src/tipad/
├── predictor.py              
├── criticality_estimator.py  
├── fusion.py                 
├── model.py                  
├── config.py                 
└── data.py                   

experiments/
├── run_tipad.py          train + evaluate on TSB-AD-M
└── compare_baselines.py  rank against official baselines

data/
├── File_List/                        
│   ├── TSB-AD-M-Tuning.csv           tuning — hyperparameter selection only
│   └── TSB-AD-M-Eva.csv              eval
└── benchmark_results/                
    ├── multi_mergedTable_VUS-PR.csv  official VUS-PR, 30 classical baselines
    └── leaderboard/
        └── Multi_*.csv               official VUS-PR, recent SOTA baselines

figures/    architecture + results figures
```

## Architecture

![TipAD architecture](figures/Overview.png)

We reformulate an anomaly as a crossing, the system’s departure from its stable state. Inspired by this, we propose **TipAD**, a multivariate anomaly detection method. 

## Overall Comparison

![TipAD Main_experiment result](figures/Main_experiment.png)

## Data

TipAD is evaluated on **TSB-AD-M**, the multivariate track of TSB-AD. The raw dataset file is too large for GitHub and must be downloaded separately.

Download [TSB-AD-M.zip](https://www.thedatum.org/datasets/TSB-AD-M.zip) and unzip it into `data/`, giving `data/TSB-AD-M/`.

> If this link doesn't work, search for the TSB-AD benchmark (NeurIPS 2024) for the current download location.

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing the Main Results

```bash
cd experiments
python3 run_tipad.py --phase train
python3 run_tipad.py --phase eval
python3 compare_baselines.py
```

| step | what it does |
|---|---|
| `train` | selects the fusion mode and softmax on TSB-AD-M-Tuning |
| `eval` | freezes those and scores every series in TSB-AD-M-Eva |
| `compare_baselines` | reproduces the paper's Avg.RANK table against the official baselines |



