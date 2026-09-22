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

We ran the experiments reported in the paper on an Apple M3 (8-core CPU), 24GB RAM, CPU only, no GPU (see Appendix B).

### Choosing the number of workers

`run_all.sh` takes the number of parallel workers as its only argument (default 4). Pick the largest row your machine can satisfy:

| Workers | RAM needed | Runtime | Suitable for |
|--------:|-----------:|--------:|--------------|
| 1       | ~3 GB      | ~XX h   | any machine  |
| 2       | ~5 GB      | ~XX h   | 8 GB laptops |
| **4**   | **~10 GB** | **XX h XX min** | **16 GB and up** |
| 8       | ~20 GB     | ~XX h   | 32 GB |


### Quick Start

To run the whole pipeline in one go — training, both seeds, averaging, and the final comparison table — run:

```bash
bash run_all.sh
```

This uses all available CPU cores. On the full 200-series evaluation set it can still take a while. See below for what each step does and how to run them manually.

### 1. Train

```bash
cd experiments
python3 run_tipad.py --phase train
```

### 2. Evaluate 

#### Option A — single process (simplest, can take several hours):

```text
python3 run_tipad.py --phase eval
```

#### Option B — parallel (faster): open N terminals in experiments/ and run

one shard in each (N = number of CPU cores is a reasonable choice), e.g. for N=4:
```text
python3 run_tipad.py --phase eval --shard 0 --nshards 4   # terminal 1
python3 run_tipad.py --phase eval --shard 1 --nshards 4   # terminal 2
python3 run_tipad.py --phase eval --shard 2 --nshards 4   # terminal 3
python3 run_tipad.py --phase eval --shard 3 --nshards 4   # terminal 4
```
#### Once all N finish, combine them:

```text
python3 run_tipad.py --phase merge
```
### 3. Compare against the baselines

```text
python3 compare_baselines.py
```



