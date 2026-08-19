#!/usr/bin/env python3
"""
build_tables.py — build base-running transition tables from published constants.

The simulation needs, for every combination of base-out state and plate-appearance
outcome, the distribution over (resulting state, runs scored). There are three
ways to get it, and this script implements the middle one:

  DETERMINISTIC (sim/tables.py fallback)
      Textbook advancement, one outcome per cell. Runs anywhere with nothing
      installed, and understates scoring by about 21% because it cannot
      represent a runner taking an extra base, a sacrifice fly, or a double play.

  LITERATURE-PARAMETERISED (this script)
      The same state machine, with the branch probabilities set to published
      empirical rates: how often a runner scores from second on a single, how
      often a ground ball with a runner on first becomes a double play, and so
      on. Every constant below is sourced. No data download, no redistribution
      question, and the result is close to the full empirical tables.

  FULLY EMPIRICAL (not included)
      Every one of the 216 cells estimated directly from play-by-play, giving a
      median of four distinct outcomes per cell including rare events no
      parameterization anticipates. This is what the production system used. It
      requires play-by-play data that is not redistributable here.

The gap between the second and third is the price of not shipping data, and
`scripts/validate.py` measures it rather than assuming it is small.

Usage:
    python scripts/build_tables.py              # writes to data/
    python scripts/build_tables.py --out /tmp   # elsewhere
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))

from tables import (INNING_OVER, N_ALTERNATIVES, N_STATES, OUTCOMES,  # noqa: E402
                    _validate)

# ─────────────────────────────────────────────────────────────────────────────
# Published advancement rates
#
# These are league-average rates from the public baseball-research literature.
# They are the entire empirical content of this script; everything else is the
# rules of the game.
# ─────────────────────────────────────────────────────────────────────────────

# Every rate below is keyed by the number of outs BEFORE the play, and the
# out-dependence is the important part. With two outs runners advance on
# contact, because there is no risk of being doubled off and no reason to wait
# and see where the ball goes. A single league-average constant averages over
# that and is wrong in both directions: it understates advancement with two outs
# and overstates it with none.
#
# This matters more than it sounds. An earlier version of this file used the
# widely quoted "a runner scores from second on a single about 63% of the time",
# which is an average over out states. Using it uniformly overstated scoring with
# nobody out by more than twenty percentage points, and the resulting tables
# missed league-average run scoring badly.

P_SINGLE_SCORES_FROM_SECOND = {0: 0.394, 1: 0.533, 2: 0.803}
"""Runner on second scores on a single, by out count."""

P_SINGLE_FIRST_TO_THIRD = {0: 0.281, 1: 0.308, 2: 0.374}
"""Runner on first reaches third on a single, by out count."""

P_DOUBLE_SCORES_FROM_FIRST = {0: 0.290, 1: 0.333, 2: 0.511}
"""Runner on first scores on a double, by out count."""

P_GROUNDOUT_SCORES_FROM_THIRD = {0: 0.293, 1: 0.389}
"""Runner on third scores on a ground out. Higher with one out: with nobody out
the infield more often concedes the out at first only if the runner holds, and
with one out the defense is likelier to take the sure out and let him go."""

P_GROUNDOUT_SECOND_TO_THIRD = {0: 0.595, 1: 0.547}
"""Runner on second advances to third on a ground out when not forced. The
single largest omission in the deterministic fallback: it happens on most
ground outs and the fallback never allows it."""

P_FLYOUT_SCORES_FROM_THIRD = {0: 0.624, 1: 0.653}
"""Runner on third scores on a fly out with fewer than two outs — the sacrifice
fly, averaged over fly-ball depth."""

P_FLYOUT_SECOND_TO_THIRD = {0: 0.321, 1: 0.231}
"""Runner on second tags and reaches third on a fly out."""

P_DOUBLE_PLAY = {0: 0.376, 1: 0.393}
"""Ground ball with a runner on first and third base empty becomes a double
play, by out count."""

# Provenance: these are league-average marginal advancement rates over recent
# seasons of play-by-play. Published tabulations of the same quantities — Tango,
# Lichtman & Dolphin, *The Book* (2006); the Retrosheet and Hardball Times
# first-to-third studies — agree with them to within a few points wherever they
# report the same out state. They are constants, not a dataset: eight numbers
# summarizing hundreds of thousands of plays.


def _bit(state, base):    # base 0 = first, 1 = second, 2 = third
    return (state >> base) & 1


def _outs(state):
    return state >> 3


def _resolve(state, outcome):
    """Return a list of (probability, post_state, runs) for one cell."""
    on1, on2, on3 = _bit(state, 0), _bit(state, 1), _bit(state, 2)
    outs = _outs(state)

    def finish(new_runners, outs_made, runs):
        """Assemble a post_state, collapsing to the inning-over sentinel.

        No branch below both scores a run and records the third out: every
        scoring-on-an-out branch requires fewer than two outs already and adds
        exactly one. So runs never need to be suppressed here, and the caller
        does not have to reason about whether the run beat the out.
        """
        total = outs + outs_made
        if total >= 3:
            assert runs == 0 or outcome in ("1B", "2B", "3B", "HR", "BB", "HBP"), (
                f"a scoring {outcome} recorded the third out in state {state}"
            )
            return (INNING_OVER, runs)
        return (new_runners + 8 * total, runs)

    if outcome == "K":
        return [(1.0, *finish(state & 0b111, 1, 0))]

    if outcome in ("BB", "HBP"):
        if not on1:
            return [(1.0, *finish((state & 0b111) | 0b001, 0, 0))]
        if not on2:
            return [(1.0, *finish(0b011 | (on3 << 2), 0, 0))]
        if not on3:
            return [(1.0, *finish(0b111, 0, 0))]
        return [(1.0, *finish(0b111, 0, 1))]

    def branch(p, condition):
        """Yield (probability, flag) pairs, collapsing to certainty when the
        branch cannot arise — e.g. a runner on second cannot advance if there is
        no runner on second."""
        if not condition:
            return [(1.0, False)]
        return [(p, True), (1 - p, False)]

    if outcome in ("GO", "FO"):
        if outs == 2:
            return [(1.0, INNING_OVER, 0)]

        is_go = outcome == "GO"
        p_third = (P_GROUNDOUT_SCORES_FROM_THIRD if is_go
                   else P_FLYOUT_SCORES_FROM_THIRD)[outs]
        p_second = (P_GROUNDOUT_SECOND_TO_THIRD if is_go
                    else P_FLYOUT_SECOND_TO_THIRD)[outs]
        # A double play is only considered with a runner on first and third base
        # empty. With a runner on third the defense is conceding the run or
        # playing it at the plate, and the source tabulations do not separate
        # those cases.
        dp_possible = is_go and on1 and not on3

        results = []
        for p_dp, double_play in branch(P_DOUBLE_PLAY[outs], dp_possible):
            if double_play:
                # The batter and the runner from first are both retired; a
                # runner on second takes third on the throw.
                results.append((p_dp, *finish(0b100 if on2 else 0b000, 2, 0)))
                continue
            for p3, third_scores in branch(p_third, bool(on3)):
                # Second may only advance to third once third is vacated.
                can_advance = bool(on2) and (not on3 or third_scores)
                for p2, second_advances in branch(p_second, can_advance):
                    runners = int(on1)                    # runner on first holds
                    if on2 and not second_advances:
                        runners |= 0b010
                    if (on2 and second_advances) or (on3 and not third_scores):
                        runners |= 0b100
                    results.append(
                        (p_dp * p3 * p2, *finish(runners, 1, 1 if third_scores else 0))
                    )
        return results

    if outcome == "1B":
        # The batter takes first and the runner on third scores. The runner on
        # second scores or stops at third; the runner on first reaches third or
        # stops at second, but only if third is free once the runner ahead of
        # him has been resolved.
        results = []
        for p2, second_scores in branch(P_SINGLE_SCORES_FROM_SECOND[outs], bool(on2)):
            third_occupied = on2 and not second_scores
            can_reach_third = bool(on1) and not third_occupied
            for p1, first_to_third in branch(P_SINGLE_FIRST_TO_THIRD[outs],
                                             can_reach_third):
                runners = 0b001                            # the batter
                if third_occupied:
                    runners |= 0b100
                if on1:
                    runners |= 0b100 if first_to_third else 0b010
                runs = on3 + (1 if (on2 and second_scores) else 0)
                results.append((p2 * p1, *finish(runners, 0, runs)))
        return results

    if outcome == "2B":
        # The batter takes second; runners on second and third score; the runner
        # on first scores or stops at third.
        results = []
        for p, scores in branch(P_DOUBLE_SCORES_FROM_FIRST[outs], bool(on1)):
            runners = 0b010 | (0b100 if (on1 and not scores) else 0)
            runs = on3 + on2 + (1 if (on1 and scores) else 0)
            results.append((p, *finish(runners, 0, runs)))
        return results

    if outcome == "3B":
        return [(1.0, *finish(0b100, 0, on1 + on2 + on3))]

    if outcome == "HR":
        return [(1.0, *finish(0b000, 0, on1 + on2 + on3 + 1))]

    raise ValueError(f"unknown outcome {outcome!r}")


def build() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    post = np.zeros((N_STATES, len(OUTCOMES), N_ALTERNATIVES), dtype=np.int8)
    runs = np.zeros((N_STATES, len(OUTCOMES), N_ALTERNATIVES), dtype=np.int8)
    cum = np.ones((N_STATES, len(OUTCOMES), N_ALTERNATIVES), dtype=np.float32)

    for state in range(N_STATES):
        for oi, outcome in enumerate(OUTCOMES):
            branches = [(p, s, r) for (p, s, r) in _resolve(state, outcome) if p > 0]
            total = sum(p for p, _, _ in branches)
            if abs(total - 1.0) > 1e-6:
                raise AssertionError(
                    f"branch probabilities for state {state} outcome {outcome} "
                    f"sum to {total:.6f}, not 1"
                )
            acc = 0.0
            for i, (p, s, r) in enumerate(branches):
                acc += p
                post[state, oi, i] = s
                runs[state, oi, i] = r
                cum[state, oi, i] = acc
            # Pad the remaining slots with the final branch so the array is
            # well-formed; the cumulative probability has already reached 1.
            for i in range(len(branches), N_ALTERNATIVES):
                post[state, oi, i] = branches[-1][1]
                runs[state, oi, i] = branches[-1][2]
                cum[state, oi, i] = 1.0

    _validate(post, runs, cum)
    return post, runs, cum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    args = ap.parse_args()

    post, runs, cum = build()
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "emp_post_state.npy"), post)
    np.save(os.path.join(args.out, "emp_runs.npy"), runs)
    np.save(os.path.join(args.out, "emp_cumprob.npy"), cum)

    # Record how these tables were made. Which generation of tables produced a
    # number changes how that number should be read, so provenance travels with
    # the data rather than being remembered.
    import json
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump({
            "generation": "literature",
            "description": (
                "Base-out transition tables built from published league-average "
                "advancement rates, out-dependent. Reproduces about 8.3 mean "
                "total runs against a true league average near 8.8; the residual "
                "is rare events (errors, wild pitches, dropped third strikes) "
                "that a compact parameterization cannot represent."
            ),
            "expected_mean_total_runs": 8.28,
            "built_by": "scripts/build_tables.py",
        }, fh, indent=2)

    n_branches = (np.diff(np.concatenate(
        [np.zeros(cum.shape[:2] + (1,), cum.dtype), cum], axis=-1), axis=-1) > 1e-9).sum(-1)
    print(f"Wrote transition tables to {args.out}")
    print(f"  cells with a stochastic branch: "
          f"{int((n_branches > 1).sum())} of {N_STATES * len(OUTCOMES)}")

    import core
    result = core.simulate(n_sims=100_000, tables=(post, runs, cum), seed=1)
    print(f"  league-average matchup: {result.total_runs.mean():.3f} mean total runs "
          f"(real league average is about 8.8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
