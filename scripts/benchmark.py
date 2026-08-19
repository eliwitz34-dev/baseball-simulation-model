#!/usr/bin/env python3
"""
benchmark.py — measure simulation throughput on your own machine.

The README quotes a speedup for the compiled kernel over the vectorized NumPy
implementation. That number is measured here rather than asserted, because the
figure is easy to get wrong in a flattering direction. Three specific traps this
script avoids:

  1. THE BASELINE. Comparing against naive Python loops inflates the multiple by
     roughly another order of magnitude and means nothing, because nobody would
     write the reference that way. The baseline here is `reference_numpy.py`, a
     genuinely vectorized implementation written to be fast.

  2. COMPILATION TIME. The first call to a Numba function compiles it. Folding
     that into a timed run understates the kernel badly; hiding it entirely
     overstates what a user feels on a cold start. It is measured separately and
     reported on its own line.

  3. WARM-UP AND VARIANCE. The first timed call in a process pays thread-pool
     start-up and cache effects. Every configuration is warmed before timing,
     and the median of several repeats is reported rather than the best.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --n-sims 50000 --repeats 7
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))

import core  # noqa: E402
from reference_numpy import simulate_numpy  # noqa: E402
from tables import load_tables_with_provenance  # noqa: E402


def _median_time(fn, repeats):
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        fn(seed=1000 + i)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=None,
                    help="single simulation count (default: sweep 10k/50k/200k)")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    sizes = [args.n_sims] if args.n_sims else [10_000, 50_000, 200_000]

    print("=" * 74)
    print("MLB game-simulation kernel — throughput benchmark")
    print("=" * 74)
    print(f"  machine   : {platform.machine()} ({os.cpu_count()} logical cores)")
    print(f"  platform  : {platform.platform()}")
    print(f"  python    : {sys.version.split()[0]}    numpy: {np.__version__}", end="")
    try:
        import numba
        print(f"    numba: {numba.__version__}")
        try:
            print(f"  threads   : {numba.get_num_threads()} (numba threading layer)")
        except Exception:
            pass
    except ImportError:
        print("\n  numba     : NOT INSTALLED — nothing to compare")
        return 1
    print()

    tables, empirical = load_tables_with_provenance(quiet=True)
    print(f"  base-running tables: {'empirical' if empirical else 'deterministic fallback'}")
    print("  (throughput is unaffected by which tables are used; both are the same shape)")
    print()

    # ── Cold start ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    core.simulate(n_sims=64, tables=tables, seed=0)
    t_compile = time.perf_counter() - t0
    print(f"Cold start (JIT compile + first call): {t_compile:.1f}s")
    if t_compile < 1.0:
        print("  (fast — a compiled kernel was reused from NUMBA_CACHE_DIR)")
    print()

    print(f"{'n_sims':>9}  {'numba (ms)':>11}  {'numpy (ms)':>11}  {'speedup':>8}  "
          f"{'numba games/s':>14}")
    print("-" * 74)

    speedups = []
    for n in sizes:
        # Warm each configuration at this size before timing it.
        core.simulate(n_sims=n, tables=tables, seed=0)
        simulate_numpy(n_sims=n, tables=tables, seed=0)

        t_numba = _median_time(
            lambda seed: core.simulate(n_sims=n, tables=tables, seed=seed),
            args.repeats)
        t_numpy = _median_time(
            lambda seed: simulate_numpy(n_sims=n, tables=tables, seed=seed),
            max(1, args.repeats // 2))

        speedups.append(t_numpy / t_numba)
        print(f"{n:>9,}  {t_numba * 1e3:>11.1f}  {t_numpy * 1e3:>11.1f}  "
              f"{t_numpy / t_numba:>7.2f}x  {n / t_numba:>14,.0f}")

    print()
    print(f"Speedup over vectorized NumPy: {min(speedups):.1f}-{max(speedups):.1f}x "
          f"on this machine, over the sizes tested.")
    print()
    print("Note on what is being compared: the compiled kernel reseeds its generator")
    print("once per simulation so that results are independent of thread count. That")
    print("costs throughput and buys exact reproducibility. Run with")
    print("--n-sims large to see the compiled kernel's advantage grow, since the")
    print("NumPy implementation's masked gathers do progressively more wasted work as")
    print("finished games accumulate in the lockstep loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
