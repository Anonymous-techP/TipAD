import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(seeds):
    dfs = [pd.read_csv(os.path.join(HERE, f"eval_merged_s{s}.csv")).set_index("file") for s in seeds]
    common = dfs[0].index
    for d in dfs[1:]:
        common = common.intersection(d.index)

    avg = sum(d.loc[common, ["fused", "nis_kf2", "R"]] for d in dfs) / len(dfs)
    avg["fam"] = dfs[0].loc[common, "fam"]
    avg = avg.reset_index()[["file", "fam", "fused", "nis_kf2", "R"]]

    out = os.path.join(HERE, "eval_merged.csv")
    avg.to_csv(out, index=False)
    print(f"averaged {len(avg)} series across seeds {seeds} -> eval_merged.csv")
    for col in ("fused", "nis_kf2", "R"):
        print(f"  mean VUS-PR[{col}] = {avg[col].mean():.4f}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [2023, 2024]
    main(seeds)
