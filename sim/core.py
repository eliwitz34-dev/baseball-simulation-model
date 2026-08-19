"""
core.py — a readable reference implementation of the game simulation engine.

This file is the one to read first. It implements the same algorithm as the
production kernel (`production/game_simulation.py`) with the same state space and
the same base-running tables, but stripped to the mechanism: roughly five hundred
lines instead of nearly four thousand, no environment-variable switches, no
fitted artifacts required, and no dependency beyond NumPy and Numba.

WHAT IT MODELS
    A baseball game is a Markov chain over 24 base-out states, driven by a
    sequence of plate appearances whose outcome probabilities depend on the
    batter and the pitcher he faces. Simulating the chain many times gives a
    joint distribution over (home runs scored, away runs scored), and every
    quantity of interest is a functional of that distribution: the probability
    the home team wins, the distribution of total runs, the margin of victory.

    Structurally this is a compound distribution. Each half-inning aggregates a
    random number of plate appearances, each contributing a random number of
    runs, and the object of interest is the aggregate distribution rather than
    its mean. The mean is the easy part and almost never the part that matters;
    the tails carry the information, and the reason to simulate rather than to
    derive is that the summands are neither independent nor identically
    distributed — the base-out state couples them.

WHAT IT DELIBERATELY OMITS
    The production system layers eleven further effects on top of this core:
    a gradient-boosted starter-removal hazard, a conditional-logit reliever
    choice model, times-through-the-order penalties, park factors, umpire and
    carry effects, platoon splits, the running game, and others. Each requires
    fitted parameters estimated from data that is not redistributed here.

    They are not omitted because they do not matter — several move the run
    distribution materially, and METHODOLOGY.md describes them and reports which
    ones survived validation. They are omitted so this file stays legible and
    runnable. Numbers produced here are therefore illustrative of the mechanism,
    not the production model's forecasts, and should not be read as such.

STATE ENCODING
    state = runners_bitmask + 8 * outs

    where bit 0 of the bitmask is a runner on first, bit 1 on second, bit 2 on
    third, and outs is 0, 1 or 2. That gives 8 * 3 = 24 reachable states. The
    value 24 is a sentinel meaning the third out was recorded and the half-inning
    is over.

REPRODUCIBILITY
    The parallel loop reseeds the generator per simulation from a base seed, so
    results depend on the seed but not on the number of threads. That costs a
    little throughput and buys exact reproducibility across machines, which for
    a model whose output is a distribution is worth far more than the cycles.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op decorator so the module imports and runs without Numba."""
        def wrap(fn):
            return fn
        return wrap if not args else args[0]

    prange = range
    warnings.warn(
        "Numba is not installed; the simulation will run as interpreted Python "
        "and will be orders of magnitude slower. Install numba to use the "
        "compiled kernel.",
        ImportWarning,
    )

OUTCOMES = ["K", "BB", "HBP", "GO", "FO", "1B", "2B", "3B", "HR"]
N_OUTCOMES = len(OUTCOMES)
N_STATES = 24
INNING_OVER = 24

# League-average plate-appearance outcome distribution, used when no player
# rates are supplied. Ordered to match OUTCOMES.
LEAGUE_AVERAGE_PA = np.array(
    [0.2242, 0.0850, 0.0110, 0.2798, 0.1812, 0.1416, 0.0430, 0.0036, 0.0306],
    dtype=np.float64,
)


# ═════════════════════════════════════════════════════════════════════════════
# Compiled kernel
# ═════════════════════════════════════════════════════════════════════════════

@njit(cache=True, inline="always")
def _draw(cumulative, rand):
    """Return the index of the first entry of `cumulative` exceeding `rand`.

    A linear scan beats binary search here: these arrays have nine entries (or
    twenty for the base-running alternatives), and the common outcomes sit near
    the front once the array is ordered, so the scan usually terminates in a
    couple of comparisons with no branch misprediction.
    """
    n = cumulative.shape[0]
    for i in range(n):
        if rand < cumulative[i]:
            return i
    return n - 1


@njit(cache=True)
def _play_half_inning(
    lineup_cum,        # (9, 9)  cumulative PA probs, one row per batting slot
    bullpen_cum,       # (9,)    cumulative PA probs for the relief pitcher
    br_post, br_runs, br_cum,   # base-running tables (24, 9, 20)
    batter_index,      # batting slot due up
    starter_active,    # 1 while the starting pitcher is still in the game
    batters_faced,     # starter's batters faced so far
    earned_runs,       # runs charged to the starter so far
    exit_bf, exit_er,  # this simulation's removal thresholds for the starter
    ghost_runner,      # 1 to start with a runner on second (extra-inning rule)
    walkoff_runs,      # >0: end the half-inning as soon as this many runs score
    max_runs,          # safety valve; see note below
):
    """Simulate one half-inning. Returns (runs, next_batter_index, ...state)."""
    state = 2 if ghost_runner else 0   # bitmask 0b010 == runner on second
    runs = 0

    while True:
        # Pick the matchup distribution for this plate appearance. `lineup_cum`
        # holds one row per batting slot, already conditioned on the opposing
        # STARTER; once he is removed, every batter faces the same aggregate
        # relief distribution. Production replaces that aggregate with an
        # individual reliever chosen by a conditional-logit model.
        if starter_active == 1:
            pitcher_cum = lineup_cum[batter_index]
        else:
            pitcher_cum = bullpen_cum

        outcome = _draw(pitcher_cum, np.random.random())

        # Look up where the runners end up. The table is stochastic: a single
        # with a runner on first sends him to third some of the time and
        # occasionally gets him thrown out, so we draw an alternative too.
        alt = _draw(br_cum[state, outcome], np.random.random())
        new_state = br_post[state, outcome, alt]
        scored = br_runs[state, outcome, alt]

        runs += scored
        if starter_active == 1:
            batters_faced += 1
            earned_runs += scored
            # The starter is removed once either threshold is crossed. This is
            # the simple rule; production replaces it with a discrete-time
            # hazard over pitch count, runs allowed, score margin and base-out
            # state, which is the single largest modeling difference between
            # this file and the production kernel.
            if batters_faced >= exit_bf or earned_runs >= exit_er:
                starter_active = 0

        batter_index += 1
        if batter_index == 9:
            batter_index = 0

        if new_state == INNING_OVER:
            break
        state = new_state

        # Walk-off: in the bottom of the final or any extra inning, play stops
        # the instant the batting team goes ahead. Without this the half-inning
        # is played to three outs and simulated home run totals are biased
        # upward. The winning run is credited along with anything else scoring
        # on the same play, which is exactly right for a walk-off home run and
        # slightly generous for other walk-off hits, where the trailing runners
        # are held.
        if walkoff_runs > 0 and runs >= walkoff_runs:
            break

        # A half-inning cannot run forever in reality, but a badly specified
        # probability vector (one with no way to record an out) would loop
        # indefinitely. Bounding it turns an infinite loop into a wrong answer,
        # which is far easier to notice and diagnose.
        if runs > max_runs:
            break

    return runs, batter_index, starter_active, batters_faced, earned_runs


@njit(parallel=True, cache=True)
def _simulate(
    home_lineup_cum, away_lineup_cum,
    home_bullpen_cum, away_bullpen_cum,
    br_post, br_runs, br_cum,
    n_sims, n_innings, stop_on_tie, use_ghost_runner,
    exit_bf_mean, exit_bf_sd, exit_er_mean, exit_er_sd,
    base_seed, max_runs,
):
    """Run `n_sims` independent games. Returns an (n_sims, 2) array [home, away]."""
    scores = np.zeros((n_sims, 2), dtype=np.int32)

    for sim in prange(n_sims):
        # Reseed per simulation so the result is independent of how the work is
        # divided across threads. See the note on reproducibility above.
        np.random.seed(base_seed + sim)

        # Each starter draws his own removal thresholds. Sharing them would tie
        # the two starters' exits together and bias every game toward mirrored
        # scripts — a subtle correlation bug that is invisible in the mean and
        # obvious in the margin distribution.
        h_exit_bf = max(15.0, min(30.0, exit_bf_mean + exit_bf_sd * np.random.normal()))
        a_exit_bf = max(15.0, min(30.0, exit_bf_mean + exit_bf_sd * np.random.normal()))
        h_exit_er = max(2.0, min(8.0, exit_er_mean + exit_er_sd * np.random.normal()))
        a_exit_er = max(2.0, min(8.0, exit_er_mean + exit_er_sd * np.random.normal()))

        home_runs = 0
        away_runs = 0
        home_bat = 0
        away_bat = 0
        h_start_on, a_start_on = 1, 1
        h_bf, a_bf = 0, 0
        h_er, a_er = 0, 0

        inning = 0
        while True:
            inning += 1
            ghost = 1 if (use_ghost_runner and inning > n_innings) else 0

            # ── Top of the inning: the away team bats against the home pitcher.
            r, away_bat, h_start_on, h_bf, h_er = _play_half_inning(
                away_lineup_cum, home_bullpen_cum, br_post, br_runs, br_cum,
                away_bat, h_start_on, h_bf, h_er, h_exit_bf, h_exit_er,
                ghost, 0, max_runs,
            )
            away_runs += r

            # ── Bottom of the inning: the home team bats.
            # The home team does not bat in the bottom of the final regulation
            # inning (or any extra inning) if it is already ahead. Omitting this
            # rule inflates simulated home run totals and distorts every
            # total-runs quantity, while leaving the win probability untouched —
            # a good example of a bug that a mean-based check would never catch.
            skip_bottom = (
                not stop_on_tie
                and inning >= n_innings
                and home_runs > away_runs
            )
            if not skip_bottom:
                # From the bottom of the final regulation inning onward, the
                # home team stops batting the moment it leads.
                walkoff = 0
                if not stop_on_tie and inning >= n_innings:
                    walkoff = away_runs - home_runs + 1
                    if walkoff < 1:
                        walkoff = 1
                r, home_bat, a_start_on, a_bf, a_er = _play_half_inning(
                    home_lineup_cum, away_bullpen_cum, br_post, br_runs, br_cum,
                    home_bat, a_start_on, a_bf, a_er, a_exit_bf, a_exit_er,
                    ghost, walkoff, max_runs,
                )
                home_runs += r

            if inning >= n_innings:
                if stop_on_tie:
                    # First-five-innings markets settle on the score after
                    # exactly `n_innings`, ties included, so stop here.
                    break
                if home_runs != away_runs:
                    break
                # Tied after regulation: play another inning.

            if inning > 30:  # pathological-input guard, as above
                break

        scores[sim, 0] = home_runs
        scores[sim, 1] = away_runs

    return scores


# ═════════════════════════════════════════════════════════════════════════════
# Python interface
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Output of a simulation run, plus enough provenance to interpret it."""

    scores: np.ndarray            # (n_sims, 2) int32, columns [home, away]
    n_sims: int
    empirical_tables: bool        # False => deterministic fallback tables were used
    seconds: float

    @property
    def p_home_win(self) -> float:
        """Probability the home team wins.

        Ties are impossible in a completed game, but are possible when
        `stop_on_tie` is set for a partial-game market, so they are excluded
        from the denominator rather than silently split.
        """
        home, away = self.scores[:, 0], self.scores[:, 1]
        decided = home != away
        if decided.sum() == 0:
            return float("nan")
        return float((home[decided] > away[decided]).mean())

    @property
    def total_runs(self) -> np.ndarray:
        return self.scores.sum(axis=1)

    @property
    def margin(self) -> np.ndarray:
        """Home runs minus away runs."""
        return self.scores[:, 0] - self.scores[:, 1]

    def p_total_over(self, line: float) -> float:
        """P(total runs > line). Lines are half-integers, so no push is possible."""
        return float((self.total_runs > line).mean())

    def p_home_cover(self, line: float) -> float:
        """P(home margin > line), the run-line probability."""
        return float((self.margin > line).mean())

    def standard_error(self, p: float) -> float:
        """Monte Carlo standard error of a simulated probability estimate.

        Worth quoting alongside any simulated probability. With 50,000
        simulations the standard error near p = 0.5 is about 0.22 percentage
        points, which sets the floor on how finely any two forecasts from this
        engine can be distinguished.
        """
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / self.n_sims))

    def summary(self) -> str:
        total = self.total_runs
        return (
            f"{self.n_sims:,} simulations in {self.seconds:.3f}s "
            f"({self.n_sims / self.seconds:,.0f} games/sec)\n"
            f"  P(home win)     {self.p_home_win:.4f} "
            f"± {self.standard_error(self.p_home_win):.4f} (Monte Carlo SE)\n"
            f"  mean total runs {total.mean():.3f}\n"
            f"  sd total runs   {total.std():.3f}\n"
            f"  P(total > 8.5)  {self.p_total_over(8.5):.4f}\n"
            f"  base-running    {'empirical' if self.empirical_tables else 'DETERMINISTIC FALLBACK'}"
        )


def _to_cumulative(probs: np.ndarray) -> np.ndarray:
    """Normalize rows to sum to one and convert to cumulative form."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if probs.shape[-1] != N_OUTCOMES:
        raise ValueError(
            f"expected {N_OUTCOMES} outcome probabilities ({', '.join(OUTCOMES)}), "
            f"got {probs.shape[-1]}"
        )
    if np.any(probs < 0):
        raise ValueError("outcome probabilities must be non-negative")
    totals = probs.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every row of outcome probabilities must sum to a positive number")
    return np.ascontiguousarray(np.cumsum(probs / totals, axis=-1))


def simulate(
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
) -> SimulationResult:
    """Simulate `n_sims` games and return the joint score distribution.

    Args:
        home_lineup:  (9, 9) array of plate-appearance outcome probabilities,
                      one row per batting slot, columns ordered as OUTCOMES.
                      Defaults to nine league-average batters.
        away_lineup:  same, for the away team.
        home_bullpen: (9,) outcome probabilities for the home relief corps
                      treated as one aggregate pitcher. Defaults to league average.
        away_bullpen: same, for the away team.
        tables:       3-tuple of base-running tables from `tables.load_or_build_tables()`.
                      Loaded automatically if omitted.
        n_sims:       number of independent games to simulate.
        n_innings:    regulation length; 5 with `stop_on_tie=True` prices the
                      first-five-innings market.
        stop_on_tie:  end after exactly `n_innings` even if the game is tied,
                      rather than continuing into extra innings.
        ghost_runner: start each extra inning with a runner on second, per the
                      current rule.
        exit_bf:      (mean, sd) of the starter's batters-faced removal threshold,
                      truncated to [15, 30].
        exit_er:      (mean, sd) of the starter's earned-runs removal threshold,
                      truncated to [2, 8].
        seed:         base seed; results are reproducible and thread-count independent.

    Returns:
        SimulationResult holding the (n_sims, 2) score array and derived markets.
    """
    import time

    from tables import load_tables_with_provenance

    # `empirical` is reported on the result so no number produced here can be
    # mistaken for one produced on the real tables.
    empirical = True
    if tables is None:
        tables, empirical = load_tables_with_provenance(quiet=True)

    br_post, br_runs, br_cum = tables

    home_lineup = LEAGUE_AVERAGE_PA if home_lineup is None else home_lineup
    away_lineup = LEAGUE_AVERAGE_PA if away_lineup is None else away_lineup
    home_bullpen = LEAGUE_AVERAGE_PA if home_bullpen is None else home_bullpen
    away_bullpen = LEAGUE_AVERAGE_PA if away_bullpen is None else away_bullpen

    home_cum = _to_cumulative(home_lineup)
    away_cum = _to_cumulative(away_lineup)
    if home_cum.shape[0] == 1:
        home_cum = np.ascontiguousarray(np.tile(home_cum, (9, 1)))
    if away_cum.shape[0] == 1:
        away_cum = np.ascontiguousarray(np.tile(away_cum, (9, 1)))
    home_pen_cum = _to_cumulative(home_bullpen)[0]
    away_pen_cum = _to_cumulative(away_bullpen)[0]

    t0 = time.perf_counter()
    scores = _simulate(
        home_cum, away_cum, home_pen_cum, away_pen_cum,
        np.ascontiguousarray(br_post), np.ascontiguousarray(br_runs),
        np.ascontiguousarray(br_cum),
        n_sims, n_innings, stop_on_tie, ghost_runner,
        exit_bf[0], exit_bf[1], exit_er[0], exit_er[1],
        seed, 50,
    )
    elapsed = time.perf_counter() - t0

    return SimulationResult(
        scores=scores,
        n_sims=n_sims,
        empirical_tables=empirical,
        seconds=elapsed,
    )


if __name__ == "__main__":
    print("Simulating a league-average matchup...\n")
    result = simulate(n_sims=50_000, seed=1)
    print(result.summary())
