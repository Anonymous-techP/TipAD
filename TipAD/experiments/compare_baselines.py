import os, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(HERE, "..", "data", "benchmark_results")
META = {"file", "ts_len", "anomaly_len", "num_anomaly", "avg_anomaly_len",
        "anomaly_ratio", "point_anomaly", "seq_anomaly"}


def load_official_methods(results_dir):
    """{method_name: Series(index=file, values=VUS-PR)} from the official
    30-baseline table and the official per-year leaderboard CSVs."""
    baseline_csv = os.path.join(results_dir, "multi_mergedTable_VUS-PR.csv")
    leaderboard_dir = os.path.join(results_dir, "leaderboard")

    methods = {}
    base = pd.read_csv(baseline_csv).set_index("file")
    for c in base.columns:
        if c not in META:
            methods[c] = base[c]

    for f in sorted(os.listdir(leaderboard_dir)):
        if not f.startswith("Multi_") or not f.endswith(".csv"):
            continue
        d = pd.read_csv(os.path.join(leaderboard_dir, f))
        idx = "file" if "file" in d.columns else ("filename" if "filename" in d.columns else None)
        if idx is None:
            continue
        d = d.set_index(idx)
        name = f[len("Multi_"):-len(".csv")]
        if "VUS-PR" in d.columns:
            methods[name] = pd.to_numeric(d["VUS-PR"], errors="coerce")
        else:
            for c in d.columns:
                if c not in META:
                    methods[c] = pd.to_numeric(d[c], errors="coerce")
    return methods


def main(results_dir):
    df = pd.read_csv(os.path.join(HERE, "eval_merged.csv"))
    methods = load_official_methods(results_dir)
    idx = df["file"].values
    M = pd.DataFrame({m: s.reindex(idx) for m, s in methods.items()})
    M["TipAD"] = df.set_index("file")["fused"].reindex(idx).values
    keep = [m for m in M.columns if m == "TipAD" or M[m].notna().mean() >= 0.5]
    dropped = [m for m in M.columns if m not in keep]
    if dropped:
        print(f"[note] dropped for insufficient coverage on this file set: {dropped}")
    M = M[keep].copy()
    M["fam"] = [f.split("_")[1] for f in M.index]

    per_dataset = M.groupby("fam").mean(numeric_only=True)
    avg_rank = per_dataset.rank(axis=1, ascending=False, method="average").mean(axis=0).sort_values()
    print("\n=== Avg.RANK on TSB-AD-M (lower is better) ===")
    for i, (m, r) in enumerate(avg_rank.items(), 1):
        vus = M[m].mean()
        flag = "  <- TipAD" if m == "TipAD" else ""
        print(f"  {i:2d}. {m:20s} Avg.RANK={r:.2f}  mean VUS-PR={vus:.4f}{flag}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                   help="directory holding multi_mergedTable_VUS-PR.csv and leaderboard/ "
                        "(default: ../data/benchmark_results, bundled in this repo)")
    args = a.parse_args()
    main(args.results_dir)
