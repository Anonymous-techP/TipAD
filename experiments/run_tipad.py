import os, sys, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import pandas as pd
import torch; torch.set_num_threads(1)
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

from tipad.config import TipADConfig
from tipad import model as M
from tipad import data as D

DATA_DIR = os.path.join(HERE, "..", "data")
POOL = os.path.join(DATA_DIR, "TSB-AD-M")
TUNING_LIST = os.path.join(DATA_DIR, "File_List", "TSB-AD-M-Tuning.csv")
EVAL_LIST = os.path.join(DATA_DIR, "File_List", "TSB-AD-M-Eva.csv")

BEST = os.path.join(HERE, "tipad_best.json")
ARR = os.path.join(HERE, "eval_arrays")
DIM_CAP, HPO_MAXLEN = 60, 20000
GRID_MODE = ["softmax", "resid", "nis_kf2"]
GRID_KAPPA = [2.0, 3.0]


def load_list(split):
    return pd.read_csv(EVAL_LIST if split == "eval" else TUNING_LIST)["file_name"].values


def load_series(fn, cap=None):
    return D.load_series(os.path.join(POOL, fn), cap=cap)


def vus(s, y, sw):
    return float(get_metrics(np.nan_to_num(s), y, slidingWindow=sw)["VUS-PR"])


def train():
    cfg = TipADConfig()
    cache = []
    for fn in load_list("tuning"):
        try:
            data, label, tr = load_series(fn, cap=HPO_MAXLEN)
            if data.shape[1] > DIM_CAP or label.sum() == 0 or tr < 5 or tr >= len(data) - 100:
                continue
            Z = D.preprocess(data, tr, "zscore")
            s = M.signals(Z, tr, cfg)
            if s is None:
                continue
            sw = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
            cache.append({"file": fn, "sw": sw, "label": label,
                          "resid": s["resid"], "nis_kf2": s["nis_kf2"]})
            print(f"  {fn[:30]:30s} nis_kf2={vus(s['nis_kf2'], label, sw):.3f} "
                  f"R={vus(s['resid'], label, sw):.3f}", flush=True)
        except Exception as e:
            print(f"  skip {fn[:28]}: {e}", flush=True)

    res = {}
    for mode in GRID_MODE:
        kappas = [2.0] if mode in ("resid", "nis_kf2") else GRID_KAPPA
        for kappa in kappas:
            vs = [vus(M.fuse(c["resid"], c["nis_kf2"], mode, kappa), c["label"], c["sw"])
                  for c in cache]
            res[(mode, kappa)] = float(np.mean(vs))
    best = max(res, key=res.get); bmode, bkappa = best
    print("\n=== mean VUS-PR on Tuning (fusion selection) ===", flush=True)
    for (m, kappa), v in sorted(res.items(), key=lambda x: -x[1]):
        print(f"  {m:8s} kappa={kappa:g}  {v:.4f}{'  *' if (m, kappa) == best else ''}", flush=True)

    frozen = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__ if k not in ("fuse", "fusion_kappa")}
    frozen["fuse"] = bmode; frozen["fusion_kappa"] = bkappa
    json.dump({"cfg": frozen, "tuning_vus": res[best]}, open(BEST, "w"), indent=2, default=str)
    print(f"\nfrozen -> fuse={bmode}, kappa={bkappa}, Tuning VUS-PR={res[best]:.4f}\nDONE_TRAIN", flush=True)


def evaluate(shard=0, nshards=1):
    fz = json.load(open(BEST))["cfg"]
    cfg = TipADConfig(**{k: v for k, v in fz.items() if k in TipADConfig.__dataclass_fields__})
    mode, kappa = fz["fuse"], float(fz["fusion_kappa"])

    os.makedirs(ARR, exist_ok=True)
    csv = os.path.join(HERE, f"eval_shard{shard}.csv") if nshards > 1 else os.path.join(HERE, "eval.csv")
    done = set(pd.read_csv(csv)["file"]) if os.path.exists(csv) else set()
    rows = pd.read_csv(csv).to_dict("records") if os.path.exists(csv) else []
    files = [f for i, f in enumerate(load_list("eval")) if i % nshards == shard and f not in done]
    print(f"[shard {shard}] fuse={mode} kappa={kappa} | {len(files)} series to go", flush=True)

    for fn in files:
        try:
            pth = os.path.join(ARR, fn.replace(".csv", "") + ".npz"); sig = None
            if os.path.exists(pth):
                z = np.load(pth, allow_pickle=True)
                sig = {"resid": z["resid"], "nis_kf2": z["nis_kf2"]}
                label, sw = z["label"], int(z["sw"])
            else:
                data, label, tr = load_series(fn)
                if label.sum() == 0 or tr < 5:
                    continue
                Z = D.preprocess(data, tr, "zscore")
                s = M.signals(Z, tr, cfg)
                if s is None:
                    print(f"[shard {shard}] skip {fn[:26]}: too short", flush=True)
                    continue
                sig = {"resid": s["resid"], "nis_kf2": s["nis_kf2"]}
                sw = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
                np.savez(pth, label=label, sw=sw, **sig)
            row = {"file": fn, "fam": fn.split("_")[1],
                   "fused": vus(M.fuse(sig["resid"], sig["nis_kf2"], mode, kappa), label, sw),
                   "nis_kf2": vus(sig["nis_kf2"], label, sw),
                   "R": vus(sig["resid"], label, sw)}
            rows.append(row); pd.DataFrame(rows).to_csv(csv, index=False)
            print(f"[shard {shard}] {fn[:26]:26s} fused={row['fused']:.3f} "
                  f"(nis={row['nis_kf2']:.3f} R={row['R']:.3f})", flush=True)
        except Exception as e:
            print(f"[shard {shard}] skip {fn[:26]}: {e}", flush=True)
    print(f"[shard {shard}] DONE_SHARD", flush=True)
    if nshards == 1:
        merge()


def merge():
    parts = [os.path.join(HERE, f) for f in os.listdir(HERE) if f.startswith("eval_shard") and f.endswith(".csv")]
    if not parts:
        parts = [os.path.join(HERE, "eval.csv")]
    df = pd.concat([pd.read_csv(p) for p in parts if os.path.exists(p)], ignore_index=True).drop_duplicates("file")
    df.to_csv(os.path.join(HERE, "eval_merged.csv"), index=False)
    print(f"merged {len(df)} series -> eval_merged.csv")
    for col in ("fused", "nis_kf2", "R"):
        print(f"  mean VUS-PR[{col}] = {df[col].mean():.4f}")
    print("DONE_MERGE")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--phase", choices=["train", "eval", "merge"], required=True)
    a.add_argument("--shard", type=int, default=0)
    a.add_argument("--nshards", type=int, default=1)
    args = a.parse_args()
    if args.phase == "train":
        train()
    elif args.phase == "merge":
        merge()
    else:
        evaluate(args.shard, args.nshards)
