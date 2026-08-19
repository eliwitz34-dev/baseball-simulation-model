"""
tables.py — base-running transition tables for the simulation kernel.

The kernel advances a half-inning through the 24 base-out states. For every
(state, plate-appearance outcome) pair it needs to know where the runners end up
and how many runs score. That mapping is the base-running transition table.

Two versions exist, and the difference between them is the point of this module.

EMPIRICAL (preferred)
    Estimated from play-by-play data: for each of the 24 x 9 cells, the observed
    distribution over (resulting state, runs scored). These are genuinely
    stochastic — a single with a runner on first sends him to third about a third
    of the time, and occasionally he is thrown out trying. The median cell has
    four distinct outcomes. Build them with `scripts/build_tables.py`, which
    fetches public play-by-play data; they are not distributed with this
    repository (see the data note in the README).

DETERMINISTIC (fallback, built here)
    Textbook advancement rules, one outcome per cell with probability 1. Every
    runner advances the number of bases the batter did; nobody takes an extra
    base, nobody is thrown out, no ball is misplayed.

The fallback exists so that the kernel is runnable immediately by someone who
has just cloned the repository. It is NOT the model. Collapsing a four-outcome
empirical distribution to its modal outcome removes real variance from the run
distribution, and a simulation's whole job here is to get that distribution
right, so results from the fallback are illustrative only. The kernel prints a
warning when it is used, and `SimulationResult` records which table it ran on.

TABLE FORMAT
    Three arrays, each of shape (24, 9, 20), matching the kernel's expectations:

      post_state  int8    resulting base-out state; the sentinel 24 means the
                          half-inning ended (the third out was recorded)
      runs        int8    runs scoring on the play (0-4)
      cumprob     float32 cumulative probability across the 20 alternative
                          outcomes; the final slot is always exactly 1.0

    State encoding: state = runners_bitmask + 8 * outs, where the bitmask uses
    bit 0 for a runner on first, bit 1 for second, bit 2 for third. So state 7
    is bases loaded with nobody out, state 16 is bases empty with two outs.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

N_STATES = 24
N_ALTERNATIVES = 20
INNING_OVER = 24  # sentinel post_state: the third out was recorded

# Must match game_simulation.OUTCOMES.
OUTCOMES = ["K", "BB", "HBP", "GO", "FO", "1B", "2B", "3B", "HR"]

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic advancement rules
# ─────────────────────────────────────────────────────────────────────────────

def _advance(runners: int, outcome: str) -> tuple[int, int, int]:
    """Apply textbook advancement to a runner bitmask.

    Returns (new_runners_bitmask, runs_scored, outs_recorded).

    The rules encoded here are the ones every introduction to the game states,
    and they are deliberately the *simple* reading of each:

      K            batter out, runners hold.
      BB / HBP     batter to first; a runner advances only if forced behind him.
                   A run scores only with the bases loaded.
      GO / FO      batter out, runners hold. Real ground and fly outs advance
                   runners often (the sacrifice fly alone scores a runner from
                   third whenever there are fewer than two outs), so this is the
                   single largest simplification in the fallback.
      1B / 2B / 3B batter takes one, two or three bases; every runner advances
                   the same number. Nobody takes an extra base and nobody is
                   thrown out, both of which happen constantly in real games.
      HR           everyone scores.
    """
    on1 = runners & 1
    on2 = (runners >> 1) & 1
    on3 = (runners >> 2) & 1

    if outcome == "K":
        return runners, 0, 1

    if outcome in ("GO", "FO"):
        return runners, 0, 1

    if outcome in ("BB", "HBP"):
        # A walk pushes a runner only if every base behind him is occupied.
        if not on1:
            # First base open: the batter takes it and nobody else is forced.
            return runners | 0b001, 0, 0
        if not on2:
            # Runner on first is forced to second; a runner on third holds.
            return 0b011 | (on3 << 2), 0, 0
        if not on3:
            # First and second forced to second and third; bases end up loaded.
            return 0b111, 0, 0
        # Bases loaded: everyone is forced and the runner on third walks home.
        return 0b111, 1, 0

    bases = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}[outcome]
    runs = 0
    new = 0
    # Advance each existing runner by `bases`; anyone reaching home scores.
    for base_idx, occupied in ((0, on1), (1, on2), (2, on3)):
        if not occupied:
            continue
        dest = base_idx + bases          # 0-indexed: 0=first, so dest 3 == home
        if dest >= 3:
            runs += 1
        else:
            new |= 1 << dest
    # Place the batter.
    if bases == 4:
        runs += 1
    else:
        new |= 1 << (bases - 1)
    return new, runs, 0


def build_deterministic_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct (24, 9, 20) tables from the textbook advancement rules.

    Every cell gets a single alternative with cumulative probability 1.0; the
    remaining 19 slots repeat it so the arrays are well-formed for the kernel,
    which reads until the cumulative probability is reached.
    """
    post = np.zeros((N_STATES, len(OUTCOMES), N_ALTERNATIVES), dtype=np.int8)
    runs = np.zeros((N_STATES, len(OUTCOMES), N_ALTERNATIVES), dtype=np.int8)
    cum = np.ones((N_STATES, len(OUTCOMES), N_ALTERNATIVES), dtype=np.float32)

    for state in range(N_STATES):
        runners = state & 0b111
        outs = state >> 3
        for oi, outcome in enumerate(OUTCOMES):
            new_runners, scored, outs_made = _advance(runners, outcome)
            total_outs = outs + outs_made
            if total_outs >= 3:
                new_state = INNING_OVER
                # Runs on a play that records the third out are credited by the
                # kernel before the inning closes, which matches how the
                # empirical tables are built.
            else:
                new_state = new_runners + 8 * total_outs
            post[state, oi, :] = new_state
            runs[state, oi, :] = scored

    return post, runs, cum


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tables(data_dir: str | None = None):
    """Load empirical tables from disk. Raises FileNotFoundError if absent."""
    data_dir = data_dir or _DEFAULT_DATA_DIR
    post = np.load(os.path.join(data_dir, "emp_post_state.npy"))
    runs = np.load(os.path.join(data_dir, "emp_runs.npy"))
    cum = np.load(os.path.join(data_dir, "emp_cumprob.npy"))
    _validate(post, runs, cum)
    return post, runs, cum


def load_or_build_tables(data_dir: str | None = None, quiet: bool = False):
    """Return empirical tables if present, otherwise the deterministic fallback.

    Callers that need to know which one they got should use
    `load_tables_with_provenance`; inferring it from whether a warning fired is
    fragile, because silencing the warning silently changes the answer.
    """
    return load_tables_with_provenance(data_dir, quiet)[0]


def load_tables_with_provenance(data_dir: str | None = None, quiet: bool = False):
    """Return (tables, is_empirical).

    The flag travels with the data so that any result computed from these tables
    can state which version produced it. A run on the deterministic fallback is
    not comparable to a run on the empirical tables, and the difference is large
    enough to change conclusions, so it should never have to be inferred.
    """
    try:
        tables = load_tables(data_dir)
        if not quiet:
            print("  base-running tables: empirical (estimated from play-by-play)")
        return tables, True
    except FileNotFoundError:
        if not quiet:
            warnings.warn(
                "Empirical base-running tables not found; falling back to "
                "deterministic textbook advancement. The kernel will run, but the "
                "run distribution will be too narrow because real base-running "
                "variance has been removed. Run scripts/build_tables.py to "
                "estimate the empirical tables from public play-by-play data.",
                RuntimeWarning,
                stacklevel=2,
            )
        return build_deterministic_tables(), False


def _validate(post, runs, cum) -> None:
    """Check the structural invariants the kernel relies on."""
    expected = (N_STATES, len(OUTCOMES), N_ALTERNATIVES)
    for name, arr, dtype in (("post_state", post, np.int8),
                             ("runs", runs, np.int8),
                             ("cumprob", cum, np.float32)):
        if arr.shape != expected:
            raise ValueError(f"{name}: expected shape {expected}, got {arr.shape}")
        if arr.dtype != dtype:
            raise ValueError(f"{name}: expected dtype {dtype}, got {arr.dtype}")
    if post.min() < 0 or post.max() > INNING_OVER:
        raise ValueError(f"post_state out of range [0, {INNING_OVER}]")
    if runs.min() < 0 or runs.max() > 4:
        raise ValueError("runs outside the plausible range [0, 4]")
    if not np.allclose(cum[..., -1], 1.0):
        raise ValueError("cumulative probabilities do not reach 1.0 in the final slot")
    if np.any(np.diff(cum, axis=-1) < -1e-6):
        raise ValueError("cumulative probabilities are not monotonically increasing")


if __name__ == "__main__":
    post, runs, cum = build_deterministic_tables()
    _validate(post, runs, cum)
    print("Deterministic base-running tables built and validated.")
    print(f"  shape {post.shape}")
    # A couple of spot checks a reader can verify against the rules above.
    loaded_0out = 0b111
    for outcome in ("HR", "BB", "1B"):
        oi = OUTCOMES.index(outcome)
        print(f"  bases loaded, 0 out, {outcome:>3}: "
              f"post_state={post[loaded_0out, oi, 0]:>2}  runs={runs[loaded_0out, oi, 0]}")
