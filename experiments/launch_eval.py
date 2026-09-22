#!/usr/bin/env python3
"""Resource-aware launcher for the TipAD evaluation phase.

Replaces the fixed `-P $(nproc)` fan-out in run_all.sh. It discovers the real
CPU/memory limits (cgroup-aware, so it is correct inside containers), measures
the actual per-series memory cost on this machine, then schedules shards under
a memory budget instead of a fixed process count. Killed shards are retried at
lower concurrency; completed work is never redone (it is cached in eval_arrays*).

Usage:
    python3 launch_eval.py --seed 2023
    python3 launch_eval.py --seed 2023 --quick        # 6-series smoke test
    python3 launch_eval.py --seed 2023 --dry-run      # print the plan only
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "..", "data", "TSB-AD-M")
EVAL_LIST = os.path.join(HERE, "..", "data", "File_List", "TSB-AD-M-Eva.csv")
GB = 1 << 30


# ---------------------------------------------------------------- detection
def _read_int(path):
    try:
        v = open(path).read().strip()
        return None if v == "max" else int(v)
    except Exception:
        return None


def memory_limit_bytes():
    """Smallest of: cgroup v2 limit, cgroup v1 limit, physical RAM."""
    cands = []
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read_int(p)
        if v and v < (1 << 62):          # v1 uses a huge sentinel for unlimited
            cands.append(v)
    try:
        cands.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError):
        pass
    return min(cands) if cands else 4 * GB


def memory_in_use_bytes():
    for p in ("/sys/fs/cgroup/memory.current",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        v = _read_int(p)
        if v is not None:
            return v
    return 0


def cpu_limit():
    """cgroup quota if set, else the CPUs we are actually allowed to run on."""
    v = _read_int("/sys/fs/cgroup/cpu.max")          # v2 stores "quota period"
    if v is None:
        try:
            q, p = open("/sys/fs/cgroup/cpu.max").read().split()
            if q != "max":
                return max(1, int(int(q) / int(p)))
        except Exception:
            pass
    q = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")   # v1
    p = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if q and q > 0 and p:
        return max(1, int(q / p))
    try:
        return len(os.sched_getaffinity(0))          # respects taskset/cpuset
    except AttributeError:
        return os.cpu_count() or 1


# ------------------------------------------------------------------- repair
def repair_cache(seed=None):
    """Drop cache files a killed process may have left half-written.

    Writes are atomic in run_tipad.py, so this only matters for caches produced
    before that fix. A truncated .npz otherwise survives forever: os.path.exists
    is true, np.load raises, the exception is swallowed, and the series is
    skipped on every future run.
    """
    import glob
    pat = f"eval_arrays_s{seed}" if seed is not None else "eval_arrays_s*"
    removed = 0
    for d in glob.glob(os.path.join(HERE, pat)):
        for f in glob.glob(os.path.join(d, "*.part*")):      # interrupted writes
            os.remove(f); removed += 1
        for f in glob.glob(os.path.join(d, "*.npz")):
            try:
                import numpy as np
                np.load(f).files                              # reads the zip index only
            except Exception:
                os.remove(f); removed += 1
    for f in glob.glob(os.path.join(HERE, "eval_shard*.csv.part")):
        os.remove(f); removed += 1
    if removed:
        print(f"  repaired cache: removed {removed} incomplete file(s); "
              f"those series will be recomputed", flush=True)
    return removed


# ------------------------------------------------------------------ costing
def series_bytes(fn):
    try:
        return os.path.getsize(os.path.join(POOL, fn))
    except OSError:
        return 0


def load_eval_files():
    import csv
    with open(EVAL_LIST) as fh:
        return [r["file_name"] for r in csv.DictReader(fh)]


def calibrate(seed, files, biggest):
    """Run exactly the heaviest series and report its real peak RSS.

    The child's output is NOT captured: this series is the largest in the set
    and can take tens of minutes, so its epoch progress must stay visible or
    the run looks hung. The peak is returned via a temp file instead of stdout.

    Trick: with nshards == len(files), the selector `i % nshards == shard`
    picks exactly one file, so we can isolate one series without touching
    run_tipad.py. The result is a legitimate eval row and is kept.
    """
    idx, n = files.index(biggest), len(files)
    out = os.path.join(HERE, ".calib_peak")
    code = ("import resource,sys,runpy,os;"
            "sys.argv=['run_tipad.py','--phase','eval','--seed',sys.argv[1],"
            "'--shard',sys.argv[2],'--nshards',sys.argv[3]];"
            "runpy.run_path('run_tipad.py',run_name='__main__');"
            "open(sys.argv[4],'w').write("
            "str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))")
    t0 = time.time()
    subprocess.run([sys.executable, "-c", code,
                    str(seed), str(idx), str(n), out], cwd=HERE)
    if os.path.exists(out):
        kb = int(open(out).read().strip())
        os.remove(out)
        return kb * (1024 if sys.platform != "darwin" else 1), time.time() - t0
    return None, time.time() - t0


# ---------------------------------------------------------------- scheduling
def run(seed, shard_ids, nshards, budget_bytes, per_proc_bytes, dry=False):
    """Bounded pool: never keep more than `slots` shards alive at once."""
    slots = max(1, int(budget_bytes // per_proc_bytes))
    print(f"  -> admitting at most {slots} concurrent shards "
          f"({per_proc_bytes/GB:.1f} GB each, budget {budget_bytes/GB:.1f} GB) "
          f"for {len(list(shard_ids))} shards", flush=True)
    if dry:
        return 0

    todo, running, failed = list(shard_ids), {}, []
    while todo or running:
        while todo and len(running) < slots:
            i = todo.pop(0)
            running[subprocess.Popen(
                [sys.executable, "run_tipad.py", "--phase", "eval",
                 "--seed", str(seed), "--shard", str(i),
                 "--nshards", str(nshards)], cwd=HERE)] = i
        time.sleep(1.0)
        for p in [p for p in running if p.poll() is not None]:
            i = running.pop(p)
            if p.returncode == -9:                   # SIGKILL == OOM killer
                print(f"  !! shard {i} OOM-killed; requeuing at lower concurrency",
                      flush=True)
                failed.append(i)
            elif p.returncode != 0:
                print(f"  !! shard {i} exited {p.returncode}", flush=True)
                failed.append(i)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--check", type=int, metavar="N",
                    help="preflight only: verify N workers fit in memory, then exit")
    ap.add_argument("--reserve-gb", type=float, default=2.0,
                    help="headroom left to the OS / page cache")
    ap.add_argument("--max-procs", type=int, default=8,
                    help="hard ceiling on concurrency (default 8; beyond this "
                         "the returns are small and the OOM risk grows)")
    ap.add_argument("--procs", type=int, default=None,
                    help="force an exact concurrency, bypassing auto-detection")
    ap.add_argument("--quick", action="store_true", help="6-series smoke test")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.check:
        repair_cache()
        lim, use = memory_limit_bytes(), memory_in_use_bytes()
        av = lim - use - int(a.reserve_gb * GB)
        need = a.check * 2.5 * GB          # conservative per-worker estimate
        print(f"  memory limit (cgroup-aware) : {lim/GB:6.1f} GB")
        print(f"  available for this run      : {av/GB:6.1f} GB")
        print(f"  usable CPUs                 : {cpu_limit():6d}")
        print(f"  requested workers           : {a.check:6d}  (~{need/GB:.1f} GB)")
        if need > av:
            fewer = max(1, int(av // (2.5 * GB)))
            sys.exit(f"\nERROR: {a.check} workers need about {need/GB:.1f} GB but only "
                     f"{av/GB:.1f} GB is available.\n"
                     f"       Re-run with fewer workers:   bash run_all.sh {fewer}")
        if a.check > cpu_limit():
            print(f"  NOTE: more workers ({a.check}) than usable CPUs ({cpu_limit()}); "
                  f"consider 'bash run_all.sh {cpu_limit()}'")
        print("  OK\n")
        return
    if a.seed is None:
        sys.exit("--seed is required unless --check is given")

    limit, used = memory_limit_bytes(), memory_in_use_bytes()
    avail = max(GB, limit - used - int(a.reserve_gb * GB))
    cpus = cpu_limit()
    files = load_eval_files()
    biggest = max(files, key=series_bytes)

    print("=" * 66)
    print(f"  memory limit (cgroup-aware) : {limit/GB:8.1f} GB")
    print(f"  currently in use            : {used/GB:8.1f} GB")
    print(f"  usable for this run         : {avail/GB:8.1f} GB")
    print(f"  usable CPUs                 : {cpus:8d}")
    print(f"  heaviest series             : {biggest[:40]} "
          f"({series_bytes(biggest)/1048576:.0f} MB on disk)")
    print("=" * 66, flush=True)

    if a.procs:
        # Concurrency was chosen by the caller, so the measurement is only
        # needed for the ETA -- not worth tens of minutes on the largest series.
        peak, secs = 2.5 * GB, None
        print("concurrency given with --procs; skipping calibration", flush=True)
    else:
        print(f"calibrating on the heaviest series ({biggest[:40]}) -- this is the "
              f"largest series in the set and can take tens of minutes;\n"
              f"its training progress is shown below. Pass --procs N to skip it.",
              flush=True)
        peak, secs = calibrate(a.seed, files, biggest)
    if peak is None:
        peak = 3 * GB
        print(f"  calibration failed; assuming {peak/GB:.1f} GB/process", flush=True)
    elif secs is not None:
        print(f"  measured peak RSS = {peak/GB:.2f} GB  ({secs:.0f}s)", flush=True)
    else:
        print(f"  assuming {peak/GB:.1f} GB per worker for the safety check",
              flush=True)
    per_proc = peak * 1.3                                     # safety factor

    if per_proc > avail:
        sys.exit(f"\nFATAL: one series alone needs ~{per_proc/GB:.1f} GB but only "
                 f"{avail/GB:.1f} GB is usable.\nFree memory or raise the container's "
                 f"limit; this dataset cannot be evaluated on this machine.")

    if a.quick:
        # nshards == len(files) makes `i % nshards == shard` select exactly one
        # file per shard, so launching 6 shards evaluates exactly 6 series.
        nshards, shard_ids = len(files), list(range(6))
    else:
        nshards = max(cpus, 60)
        shard_ids = list(range(nshards))
    if a.procs:
        slots = a.procs
        print(f"\nconcurrency forced to {slots} by --procs", flush=True)
    else:
        slots = min(int(avail // per_proc), cpus, a.max_procs)
        slots = max(1, slots)
        print(f"\nconcurrency = min(memory {int(avail//per_proc)}, "
              f"cpus {cpus}, cap {a.max_procs}) = {slots}", flush=True)

    # --- ETA: extrapolate from the calibration run, weighted by file size ---
    if secs is None:
        print("(no ETA without calibration)", flush=True)
        total_bytes = 0
    else:
        total_bytes = sum(series_bytes(f) for f in files)
    big_bytes = max(series_bytes(biggest), 1)
    serial_secs = (secs or 0) * (total_bytes / big_bytes)
    eta_h = serial_secs * (2 if not a.quick else 6 / len(files) * 2) / slots / 3600
    if secs is not None:
        print(f"estimated wall-clock for the remaining work: ~{eta_h:.1f} h "
              f"({slots} procs); cached series are skipped, so the real time "
              f"is lower", flush=True)
    print(f"planning {len(shard_ids)} shards (nshards={nshards})", flush=True)

    budget = slots * per_proc
    for attempt in range(4):
        failed = run(a.seed, shard_ids, nshards, budget, per_proc, dry=a.dry_run)
        if a.dry_run or not failed:
            break
        shard_ids, budget = failed, budget / 2
        print(f"\nretry {attempt+1}: halving budget to {budget/GB:.1f} GB "
              f"({len(failed)} shards to redo; finished work is cached)", flush=True)
    else:
        sys.exit("FATAL: shards still failing after 4 attempts.")
    print("\nall shards complete.", flush=True)


if __name__ == "__main__":
    main()
