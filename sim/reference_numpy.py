"""
reference_numpy.py — an independent NumPy implementation of the same simulation.

PRIMARY PURPOSE: A CROSS-CHECK ON THE COMPILED KERNEL.
    A compiled parallel loop is precisely where a transcription error hides
    quietly. An off-by-one in a lookup index or a state written back to the
    wrong variable produces output that still looks like baseball, and the
    intermediate steps cannot be inspected mid-flight. This file was written
    separately from the same specification, so two implementations agreeing on a
    distribution to within Monte Carlo error is meaningful evidence that neither
    contains such an error. `scripts/validate.py` runs that comparison and takes
    its tolerances from Monte Carlo error rather than from a number chosen to
    pass.

    The two implementations do not produce identical draws, because they consume
    random numbers in a different order. They are compared as distributions, not
    sample by sample.

SECONDARY: IT IS ALSO THE HONEST BASELINE IF THROUGHPUT IS BEING MEASURED.
    Benchmarking a compiled kernel against naive Python loops reports an
    enormous and meaningless multiple, because nobody would write the reference
    that way. This is a competent implementation in the language's normal idiom,
    written to be fast rather than to lose gracefully.

    The vectorization is over *simulations*, not over innings: all N games
    advance in lockstep, one plate appearance at a time, with a boolean mask
    marking which games still have a live half-inning. That is the only axis
    with enough width to amortize NumPy's per-operation overhead, since innings
    are short and their length is itself random.
"""
from __future__ import annotations

import numpy as np

from core import LEAGUE_AVERAGE_PA, N_OUTCOMES, INNING_OVER, _to_cumulative


def _draw_rows(cum_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one index per row of a (k, m) cumulative-probability array.

    `(u[:, None] < cum_rows).argmax(1)` finds, for each row, the first column
    whose cumulative probability exceeds the uniform draw. Because the final
    column is always exactly 1.0, at least one comparison is always True, so
    argmax never silently returns 0 from an all-False row.
    """
    u = rng.random(cum_rows.shape[0])
    return (u[:, None] < cum_rows).argmax(axis=1)


def simulate_numpy(
    home_lineup=None,
    away_lineup=None,
    home_bullpen=None,
    away_bullpen=None,
    tables=None,
    n_sims: int = 50_000,
    n_innings: int = 9,
    stop_on_tie: bool = False,
    ghost_runner: bool = True,
    exit_bf: tuple[float, float] = (21.0, 3.0),
    exit_er: tuple[float, float] = (4.0, 1.5),
    seed: int = 0,
) -> np.ndarray:
    """Simulate `n_sims` games; return an (n_sims, 2) int32 array [home, away].

    Signature mirrors `core.simulate` so the two can be swapped in tests.
    """
    from tables import load_or_build_tables

    if tables is None:
        tables = load_or_build_tables(quiet=True)
    br_post, br_runs, br_cum = tables

    rng = np.random.default_rng(seed)

    home_cum = _to_cumulative(LEAGUE_AVERAGE_PA if home_lineup is None else home_lineup)
    away_cum = _to_cumulative(LEAGUE_AVERAGE_PA if away_lineup is None else away_lineup)
    if home_cum.shape[0] == 1:
        home_cum = np.tile(home_cum, (9, 1))
    if away_cum.shape[0] == 1:
        away_cum = np.tile(away_cum, (9, 1))
    home_pen = _to_cumulative(LEAGUE_AVERAGE_PA if home_bullpen is None else home_bullpen)[0]
    away_pen = _to_cumulative(LEAGUE_AVERAGE_PA if away_bullpen is None else away_bullpen)[0]

    n = n_sims
    home_runs = np.zeros(n, dtype=np.int32)
    away_runs = np.zeros(n, dtype=np.int32)
    home_bat = np.zeros(n, dtype=np.int64)
    away_bat = np.zeros(n, dtype=np.int64)

    # Per-simulation starter removal thresholds, drawn once per game.
    h_exit_bf = np.clip(rng.normal(exit_bf[0], exit_bf[1], n), 15, 30)
    a_exit_bf = np.clip(rng.normal(exit_bf[0], exit_bf[1], n), 15, 30)
    h_exit_er = np.clip(rng.normal(exit_er[0], exit_er[1], n), 2, 8)
    a_exit_er = np.clip(rng.normal(exit_er[0], exit_er[1], n), 2, 8)
    h_on = np.ones(n, dtype=bool)   # home starter still pitching
    a_on = np.ones(n, dtype=bool)
    h_bf = np.zeros(n, dtype=np.int32)
    a_bf = np.zeros(n, dtype=np.int32)
    h_er = np.zeros(n, dtype=np.int32)
    a_er = np.zeros(n, dtype=np.int32)

    game_live = np.ones(n, dtype=bool)

    def half_inning(live, lineup_cum, pen_cum, bat_idx, starter_on,
                    bf, er, exit_bf_v, exit_er_v, ghost, walkoff):
        """Advance every live game through one half-inning. Returns runs scored."""
        runs = np.zeros(n, dtype=np.int32)
        state = np.where(ghost, 2, 0).astype(np.int64) if np.ndim(ghost) else \
            np.full(n, 2 if ghost else 0, dtype=np.int64)
        alive = live.copy()

        guard = 0
        while alive.any():
            idx = np.flatnonzero(alive)

            # Assemble the matchup distribution for each live game: the batter's
            # row while the starter is in, the aggregate relief row afterwards.
            rows = np.where(
                starter_on[idx, None],
                lineup_cum[bat_idx[idx]],
                pen_cum[None, :],
            )
            outcome = _draw_rows(rows, rng)

            alt = _draw_rows(br_cum[state[idx], outcome], rng)
            new_state = br_post[state[idx], outcome, alt].astype(np.int64)
            scored = br_runs[state[idx], outcome, alt].astype(np.int32)

            runs[idx] += scored

            # Charge the plate appearance to the starter where he is still in,
            # then apply the removal rule.
            st = idx[starter_on[idx]]
            if st.size:
                bf[st] += 1
                er[st] += scored[starter_on[idx]]
                starter_on[st] &= ~((bf[st] >= exit_bf_v[st]) | (er[st] >= exit_er_v[st]))

            bat_idx[idx] = (bat_idx[idx] + 1) % 9
            state[idx] = new_state

            done = new_state == INNING_OVER
            if walkoff is not None:
                done |= runs[idx] >= walkoff[idx]
            alive[idx[done]] = False

            guard += 1
            if guard > 200:   # pathological-input guard, matching the kernel
                break

        return runs

    inning = 0
    while game_live.any():
        inning += 1
        ghost = ghost_runner and inning > n_innings

        away_runs += half_inning(game_live, away_cum, home_pen, away_bat,
                                 h_on, h_bf, h_er, h_exit_bf, h_exit_er,
                                 ghost, None)

        # The home team does not bat in the bottom of the final or an extra
        # inning while it already leads.
        bats = game_live.copy()
        if not stop_on_tie and inning >= n_innings:
            bats &= ~(home_runs > away_runs)
            walkoff = np.maximum(away_runs - home_runs + 1, 1)
        else:
            walkoff = None

        home_runs += half_inning(bats, home_cum, away_pen, home_bat,
                                 a_on, a_bf, a_er, a_exit_bf, a_exit_er,
                                 ghost, walkoff)

        if inning >= n_innings:
            if stop_on_tie:
                break
            game_live &= home_runs == away_runs
        if inning > 30:
            break

    return np.column_stack([home_runs, away_runs]).astype(np.int32)
