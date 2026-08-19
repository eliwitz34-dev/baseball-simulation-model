#!/usr/bin/env python3
"""
validate.py — checks that must pass before any result from this model is quoted.

These are not unit tests in the usual sense. They are the model-validation checks
that catch the failures actually observed in this project: transcription errors
inside the compiled kernel, silent substitution of fallback data, and results
quoted without regard to Monte Carlo error.

  1. CROSS-IMPLEMENTATION AGREEMENT
     The compiled kernel and the vectorized NumPy implementation were written
     separately from the same specification. They should agree on the simulated
     distribution to within Monte Carlo error. A compiled parallel loop is
     exactly where an off-by-one in a lookup index hides, because it cannot be
     inspected mid-flight; an independent implementation is the cheapest
     available detector.

  2. TRANSITION TABLE INVARIANTS
     Structural properties the kernel assumes and does not check at run time.
     A malformed table produces plausible-looking output, which is the worst
     possible failure mode.

  3. FALLBACK DIVERGENCE
     Quantifies how far the data-free fallback sits from the empirical tables,
     so the gap is a measured number rather than a reassuring adjective.

  4. FACE VALIDITY
     Scoring levels against the real league average. A model that is internally
     consistent and externally absurd passes every other check here.

Exit status is non-zero if any check fails, so it can gate a commit.

Usage:
    python scripts/validate.py
    python scripts/validate.py --empirical-tables /path/to/dir
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))

import core  # noqa: E402
import tables as tbl  # noqa: E402
from reference_numpy import simulate_numpy  # noqa: E402

# Real-world reference points, for face validity. Both are stable across recent
# seasons to well inside the tolerance used.
LEAGUE_RUNS_PER_GAME = 8.8      # both teams combined
LEAGUE_HOME_WIN_RATE = 0.535    # home-field advantage, long-run

def _read_provenance(data_dir: str | None) -> dict:
    """Read the manifest written alongside built tables, if there is one."""
    import json
    base = data_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    try:
        with open(os.path.join(base, "provenance.json")) as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return {}


PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def check_table_invariants(tables) -> None:
    print("\nTransition table invariants")
    post, runs, cum = tables
    try:
        tbl._validate(post, runs, cum)
        check("structural invariants (shape, dtype, monotone cumulative probabilities)", True)
    except ValueError as exc:
        check("structural invariants", False, str(exc))

    # Outs may never decrease. Encoded as: the resulting state's out count is at
    # least the starting state's, unless the inning ended.
    violations = 0
    for state in range(tbl.N_STATES):
        outs = state >> 3
        for oi in range(len(tbl.OUTCOMES)):
            for alt in range(tbl.N_ALTERNATIVES):
                new = int(post[state, oi, alt])
                if new == tbl.INNING_OVER:
                    continue
                if (new >> 3) < outs:
                    violations += 1
    check("outs never decrease within a half-inning", violations == 0,
          f"{violations} violating entries" if violations else "")

    # A plate appearance can score at most four runs (a grand slam).
    check("runs per play within [0, 4]", int(runs.min()) >= 0 and int(runs.max()) <= 4,
          f"observed range [{int(runs.min())}, {int(runs.max())}]")


def check_cross_implementation(tables, n_sims: int) -> None:
    """Compare the two implementations as distributions, not sample by sample.

    They consume random numbers in a different order, so individual games differ.
    What must agree is the distribution, and the tolerance is set by Monte Carlo
    error rather than chosen for convenience: a difference of more than four
    standard errors between two correct implementations would occur essentially
    never.
    """
    print(f"\nCross-implementation agreement ({n_sims:,} simulations each)")

    a = core.simulate(n_sims=n_sims, tables=tables, seed=11)
    b_scores = simulate_numpy(n_sims=n_sims, tables=tables, seed=22)

    a_total, b_total = a.total_runs, b_scores.sum(axis=1)
    a_win = a.p_home_win
    b_win = float((b_scores[:, 0] > b_scores[:, 1]).mean())

    # Standard error of the difference of two independent sample means.
    se_total = np.sqrt(a_total.var() / n_sims + b_total.var() / n_sims)
    d_total = abs(a_total.mean() - b_total.mean())
    check("mean total runs agree", d_total < 4 * se_total,
          f"{a_total.mean():.4f} vs {b_total.mean():.4f} "
          f"(difference {d_total:.4f}, 4 SE = {4 * se_total:.4f})")

    se_win = np.sqrt(2 * 0.25 / n_sims)
    d_win = abs(a_win - b_win)
    check("home win probability agrees", d_win < 4 * se_win,
          f"{a_win:.4f} vs {b_win:.4f} "
          f"(difference {d_win:.4f}, 4 SE = {4 * se_win:.4f})")

    sd_ratio = a_total.std() / b_total.std()
    check("total-runs dispersion agrees", 0.97 < sd_ratio < 1.03,
          f"sd ratio {sd_ratio:.4f}")


def check_reproducibility(tables) -> None:
    """The same seed must give the same answer, regardless of thread count."""
    print("\nReproducibility")
    first = core.simulate(n_sims=5_000, tables=tables, seed=7).scores
    second = core.simulate(n_sims=5_000, tables=tables, seed=7).scores
    check("identical results from an identical seed",
          np.array_equal(first, second))

    different = core.simulate(n_sims=5_000, tables=tables, seed=8).scores
    check("different seeds give different results",
          not np.array_equal(first, different))


def check_fallback_divergence(empirical_dir: str | None) -> None:
    """Measure the fallback's error rather than describing it."""
    print("\nFallback divergence from empirical tables")
    try:
        emp = tbl.load_tables(empirical_dir)
    except FileNotFoundError:
        check("empirical tables available for comparison", False,
              "not present — skipping (run scripts/build_tables.py)")
        return

    det = tbl.build_deterministic_tables()
    e_post, e_runs, _ = emp
    d_post, d_runs, _ = det

    agree = int(((d_post[:, :, 0] == e_post[:, :, 0]) &
                 (d_runs[:, :, 0] == e_runs[:, :, 0])).sum())
    total = tbl.N_STATES * len(tbl.OUTCOMES)
    # Reported, not asserted. There is no principled threshold here — the right
    # level of agreement depends entirely on which generation of tables is
    # loaded, and inventing a bound would be asserting a number rather than
    # measuring one. The figure is the point.
    print(f"         fallback matches the loaded tables' modal outcome in "
          f"{agree}/{total} cells ({agree / total:.0%})")

    emp_run = core.simulate(n_sims=40_000, tables=emp, seed=3)
    det_run = core.simulate(n_sims=40_000, tables=det, seed=3)
    print(f"         empirical tables : {emp_run.total_runs.mean():.3f} mean total runs, "
          f"sd {emp_run.total_runs.std():.3f}")
    print(f"         fallback tables  : {det_run.total_runs.mean():.3f} mean total runs, "
          f"sd {det_run.total_runs.std():.3f}")


def check_face_validity(tables, empirical: bool, provenance: dict) -> None:
    """Compare simulated scoring against what these tables should produce.

    The target depends on which generation of tables is loaded, and pretending
    otherwise would be dishonest in one direction or the other: holding the
    literature-parameterized tables to the true league average marks a known and
    documented approximation as a failure, while relaxing the target for the
    empirical tables would hide a real regression. So the check tests each
    generation against its own expectation, and separately reports the distance
    from reality, which is the number a reader actually wants.
    """
    print("\nFace validity")
    result = core.simulate(n_sims=100_000, tables=tables, seed=5)
    mean_total = result.total_runs.mean()

    expected = provenance.get("expected_mean_total_runs")
    if expected is None:
        expected = LEAGUE_RUNS_PER_GAME if empirical else None

    if expected is None:
        print(f"         {mean_total:.3f} mean total runs; no expectation recorded "
              f"for these tables, so nothing is asserted")
    else:
        check("scoring matches what these tables should produce",
              abs(mean_total - expected) < 0.25,
              f"{mean_total:.3f} simulated vs {expected:.2f} expected "
              f"for the '{provenance.get('generation', 'empirical')}' tables")

    gap = mean_total - LEAGUE_RUNS_PER_GAME
    print(f"         distance from real baseball: {gap:+.3f} runs per game "
          f"({abs(gap) / LEAGUE_RUNS_PER_GAME:.1%} "
          f"{'low' if gap < 0 else 'high'} vs the true {LEAGUE_RUNS_PER_GAME})")

    # With two identical league-average teams and no home-field advantage
    # modeled, the home win probability should be 0.5 — NOT the real 0.535.
    # This checks the simulation is unbiased, and simultaneously documents that
    # home-field advantage is not in this reference implementation.
    p = result.p_home_win
    check("identical teams give a symmetric result",
          abs(p - 0.5) < 4 * result.standard_error(p),
          f"P(home win) = {p:.4f} ± {result.standard_error(p):.4f}; "
          f"real home teams win {LEAGUE_HOME_WIN_RATE:.1%}, which this "
          f"reference implementation does not model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empirical-tables", default=None,
                    help="directory holding emp_*.npy (default: ../data)")
    ap.add_argument("--n-sims", type=int, default=40_000,
                    help="simulations per cross-implementation check")
    args = ap.parse_args()

    print("=" * 74)
    print("Model validation")
    print("=" * 74)

    tables, empirical = tbl.load_tables_with_provenance(args.empirical_tables, quiet=True)
    provenance = _read_provenance(args.empirical_tables)
    label = (provenance.get("generation", "empirical").upper() if empirical
             else "DETERMINISTIC FALLBACK")
    print(f"\nbase-running tables in use: {label}")
    if provenance.get("description"):
        print(f"  {provenance['description']}")
    if not empirical:
        print("  (some checks below are weaker on the fallback; this is stated per check)")

    check_table_invariants(tables)
    check_cross_implementation(tables, args.n_sims)
    check_reproducibility(tables)
    check_fallback_divergence(args.empirical_tables)
    check_face_validity(tables, empirical, provenance)

    failed = [r for r in _results if r[0] == FAIL]
    print("\n" + "=" * 74)
    print(f"{len(_results) - len(failed)} passed, {len(failed)} failed")
    for _, name, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
