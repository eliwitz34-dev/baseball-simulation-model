"""
game_simulation.py
==================
Monte Carlo game simulation engine for the MLB win probability model.

ARCHITECTURE
------------
Inputs (per game):
  home_probs:  dict[batter_id → dict[h → PA outcome probs]]  — from level3_predictor.precompute_game()
  away_probs:  same for away team
  home_lineup: list[{"batter_id": int, "handedness": int, "batting_order": int}]
  away_lineup: same for away team
  home_pitcher_id: int
  away_pitcher_id: int
  n_sims:      int — number of Monte Carlo game simulations (default 50_000)

Outputs:
  SimulationResult:
    p_home_win:    float — P(home team wins)
    score_dist:    (n_sims, 2) int32 array — [home_runs, away_runs] per sim
    markets:       dict — {moneyline, run_total, spread, first_five, ...}
    ci90:          dict — 90% confidence intervals per market

STATE REPRESENTATION
--------------------
The 24 base-out states are the standard in baseball analytics (Lindsey 1963, Tango 2007,
confirmed in academic simulation literature: Bukiet et al. 1997 OR, Tallavarjula 2026 SAGE).

State encoding: integer 0-23
  bits 0-2: runners on base (bitmask: bit 0=1B, bit 1=2B, bit 2=3B)
  bit  3-4: outs (0, 1, or 2)
  
  state = runners_bitmask + outs * 8
  
  runners_bitmask ∈ {0..7}: 0=empty, 1=1B only, 2=2B only, 3=1B+2B, 4=3B, 5=1B+3B, 6=2B+3B, 7=loaded
  outs ∈ {0,1,2}
  state ∈ {0..23}

Terminal state: outs == 3 → inning ends, reset to state 0 for next half-inning.

BASE RUNNING TRANSITIONS
------------------------
Per Mott et al. (2025) Section 2.4, base running uses league-average empirical transitions.
For each (pre_state, PA_outcome) pair, the transition gives:
  (post_state, runs_scored)

These are computed from Statcast play-by-play data (2021-2025) and stored as lookup tables.
The transitions are computed by compute_baserunning_tables() and cached to disk.

Key design choices:
  - 9 PA outcomes (K, BB, HBP, GO, FO, 1B, 2B, 3B, HR) × 24 states = 216 (state, outcome) pairs
  - Each maps to (new_state, runs_scored) — both integers, fits in int8/int16
  - Stored as two int8 arrays of shape (24, 9): new_state_table, runs_table
  - At lookup time: O(1) per PA — ideal for numba inner loop

STARTER EXIT MODEL
------------------
Modern starters average ~86 pitches and 5.24 IP per start (FanGraphs 2024).
Per SABR (2023), 71% of starters face the batting order twice, 56% face it three times.

We model starter exit as a function of:
  1. Batters faced (primary signal: "times through the order" penalty)
  2. Simulated performance (ER allowed in the sim)

A starter exits when EITHER:
  - batters_faced >= exit_bf_threshold (drawn per-sim from empirical distribution)
  - earned_runs >= exit_er_threshold (drawn per-sim from empirical distribution)

exit_bf_threshold ~ TruncNormal(mean=21, std=3, min=15, max=30)  [~3rd time through order]
exit_er_threshold ~ TruncNormal(mean=4, std=1.5, min=2, max=8)

Each starter draws its own independent threshold pair. Sharing thresholds
would force both pitchers to exit at the same BF/ER, which biases every
simulation toward mirrored game scripts.

After starter exit: aggregate bullpen pool (single entity with usage-weighted rates).
Bullpen rates are loaded from Statcast reliever data and stored alongside game inputs.

NUMBA JIT STRATEGY
------------------
The inner simulation loop must be numba-compiled for speed:
  - @njit(parallel=True) over n_sims — embarrassingly parallel
  - prange for the simulation index (each sim independent)
  - Pure integer arithmetic inside the loop (no Python objects)
  - All lookup tables passed as numpy int8 arrays

Expected performance:
  - Pure Python:  50k sims ≈ 2-5 seconds
  - Numba JIT:    50k sims ≈ 50-150ms (20-40x speedup)
  - First call:   +5-10s compilation overhead (cached to disk)

REFERENCES
----------
- Mott et al. (2025): PBRB model, base running transitions (Section 2.4)
- Bukiet, Harold, Palacios (1997): Markov window baseball simulation (Operations Research)
- Lindsey (1963): Original 24 base-out state framework
- Tango, Lichtman, Dolphin (2006): The Book — Playing the Percentages in Baseball
- Tallavarjula (2026 SAGE): Speed-stratified Monte Carlo baseball simulation
- FanGraphs (2024): Modern starter usage trends (~86 pitches, 5.24 IP avg)
- SABR (2023): Starter/bullpen workload distribution in postseason
"""

from __future__ import annotations

import os
import json
import warnings
import time
import pickle

# ── Calibrated wrapper (module-level for pickle compatibility) ────────────────

class _CalibratedWrapper:
    """Isotonic-calibrated classifier wrapper. Defined at module level for pickling."""
    def __init__(self, base, calibrator, feature_names=None):
        self.base          = base
        self.calibrator    = calibrator
        self.feature_names = feature_names  # converts numpy to named DataFrame for LightGBM

    def predict_proba(self, X):
        import numpy as _np, pandas as _pd
        if self.feature_names is not None and not isinstance(X, _pd.DataFrame):
            X = _pd.DataFrame(X, columns=self.feature_names)
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return _np.column_stack([1 - cal, cal])

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Numba import — optional at module level for servers without numba installed.
# Compile errors at import time are caught; simulation falls back to numpy.
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    warnings.warn(
        "numba not installed — simulation will run in numpy fallback mode (~20x slower). "
        "Install with: pip install numba",
        ImportWarning,
    )

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

OUTCOMES       = ["K", "BB", "HBP", "GO", "FO", "1B", "2B", "3B", "HR"]
N_OUTCOMES     = len(OUTCOMES)
OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}

# ── Pitches per PA outcome (empirical 2023+, pa_level_dataset) ───────────────
# Pitch count is the DOMINANT driver of starter removal (LightGBM hazard: at a
# fixed BF, varying pitch count 70→110 swings P(removal) 0.22→0.99). The sim
# accumulates the LITERAL pitch count from the OUTCOMES drawn (K/BB cost ~1.5–2
# more pitches than balls in play) into home_pitches/away_pitches, and the
# starter-removal hazard LUT (exit_model.bake_start_lut) is baked over a literal
# pitch_count axis (index 0..130 = actual pitches), so the kernel indexes it with
# _pc = int(home_pitches) directly — NO "effective-BF" rescale. (HISTORICAL NOTE:
# the OLD GAM hazard baked pitch_count = BF×3.8 and looked it up at an
# eff_bf = cum_pitches / LEAGUE_PITCHES_PER_BF surrogate; that mechanism is GONE
# with the exit-model rebuild — the two constants below are now VESTIGIAL, used
# nowhere in the live hazard path.) A high-K / high-walk (laboring) outcome mix
# therefore raises the literal pitch count → earlier hook, emergent from the
# simulated matchup. Order matches OUTCOMES.
PITCHES_PER_OUTCOME   = np.array([4.85, 5.71, 3.11, 3.36, 3.38, 3.36, 3.35, 3.47, 3.34],
                                 dtype=np.float64)
LEAGUE_PITCHES_PER_BF = 3.8                          # vestigial (old GAM eff-BF surrogate; not used in the live hazard)
INV_PITCHES_PER_BF    = np.float64(1.0 / LEAGUE_PITCHES_PER_BF)   # vestigial

# ── Pitcher-specific pitch economy (within-outcome pitches/PA) ───────────────
# build_pitch_economy.py estimates a per-pitcher (9,) pitches-per-outcome vector
# with empirical-Bayes shrinkage toward the league vector above (K/BB carry real
# signal and shrink lightly; balls-in-play and rare outcomes shrink hard to
# league). Passing a pitcher's vector into the sim makes the in-game pitch
# trajectory — and therefore the effective-BF hook lookup — reflect that
# pitcher's true count economy (nibbler exits earlier, attacker goes deeper),
# on top of the outcome-mix effect the sim already simulates. A league-average
# pitcher's vector ≈ PITCHES_PER_OUTCOME, so calibration is preserved on average.
# Stochastic per-PA pitch count: draw each PA's pitch count from the empirical
# per-outcome distribution (centered on the pitcher's mean) instead of adding the
# deterministic mean. Restores the per-PA dispersion the mean removes — the
# deterministic version under-disperses exit-BF (std ~3.6 vs empirical ~4.9);
# stochastic recovers ~4.1 while leaving the MEAN exit unchanged (+0.05 BF), so
# it sharpens hook-timing variance at no calibration cost. Toggle for A/B.
STOCHASTIC_PITCHES = os.environ.get("STOCHASTIC_PITCHES", "1") not in ("0", "false", "False")

_PITCH_ECONOMY_CACHE: dict | None = None   # pitcher_id → (9,) economy vector
_PITCH_PROFILE_CACHE: dict | None = None   # pitcher_id → full profile dict
_PITCH_DIST_CACHE: tuple | None = None     # (resid_vals(9,M), resid_cum(9,M))


_REL_HAZARD_CACHE = None


def load_reliever_hazard(path: str | None = None) -> np.ndarray:
    """Reliever removal-hazard table (REL_HAZARD_SHAPE) from train_reliever_removal_model.py. Cached.
    DEFAULT ON + artifact REQUIRED: when REL_HAZARD != 0 the reliever_removal_hazard.npz artifact is
    HARD-REQUIRED (missing/wrong-shape RAISES — no silent zero-table). Explicit REL_HAZARD=0 = golden-master
    escape hatch → an all-zero table ('never pull on hazard', so reliever changes fall back to the
    closer/fireman overrides only)."""
    global _REL_HAZARD_CACHE
    if _REL_HAZARD_CACHE is not None:
        return _REL_HAZARD_CACHE
    if os.environ.get("REL_HAZARD", "1") == "0":
        _REL_HAZARD_CACHE = np.ascontiguousarray(np.zeros(REL_HAZARD_SHAPE, dtype=np.float32))
        return _REL_HAZARD_CACHE
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "processed", "reliever_removal_hazard.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"REL_HAZARD is ON (default) but {path} is missing. Run train_reliever_removal_model.py "
            "(wired in finalize_fit) or set REL_HAZARD=0 to disable (zero-table = never pull on hazard).")
    tbl = np.load(path)["table"].astype(np.float32)
    if tbl.shape != REL_HAZARD_SHAPE:
        raise ValueError(f"reliever hazard shape {tbl.shape} != {REL_HAZARD_SHAPE}")
    _REL_HAZARD_CACHE = np.ascontiguousarray(tbl)
    return _REL_HAZARD_CACHE


def load_pitch_count_dist(path: str | None = None) -> tuple:
    """Load the per-outcome zero-mean pitch-count residual distribution. Cached.

    Returns (resid_vals, resid_cum), each (N_OUTCOMES, MAXP) float64. Missing
    file → a degenerate single-point distribution at 0 (i.e. no dispersion, so
    stochastic accumulation reduces exactly to the deterministic mean).
    """
    global _PITCH_DIST_CACHE
    if _PITCH_DIST_CACHE is not None:
        return _PITCH_DIST_CACHE
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "processed", "pitch_count_dist.npz",
        )
    if os.path.exists(path):
        z = np.load(path)
        vals = np.ascontiguousarray(z["resid_vals"], dtype=np.float64)
        cum  = np.ascontiguousarray(z["resid_cum"],  dtype=np.float64)
    else:
        vals = np.zeros((N_OUTCOMES, 1), dtype=np.float64)
        cum  = np.ones((N_OUTCOMES, 1),  dtype=np.float64)
    _PITCH_DIST_CACHE = (vals, cum)
    return _PITCH_DIST_CACHE


def _load_pitch_profile(path: str | None = None) -> dict:
    """Load the full per-pitcher hook profile (workload + economy). Cached.

    Returns dict[pitcher_id → {avg_bf, avg_pitches, k_pct, ppo(9,)}]. Missing
    file → empty dict (every pitcher then falls back to league defaults).
    """
    global _PITCH_PROFILE_CACHE
    if _PITCH_PROFILE_CACHE is not None:
        return _PITCH_PROFILE_CACHE
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "processed", "pitch_economy.parquet",
        )
    table: dict = {}
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_parquet(path)
        cols = [f"ppo_{o}" for o in OUTCOMES]
        has_wl = {"avg_bf", "avg_pitches", "k_pct"}.issubset(df.columns)
        for _, row in df.iterrows():
            rec = {"ppo": np.ascontiguousarray(row[cols].to_numpy(dtype=np.float64))}
            if has_wl:
                rec["avg_bf"]      = float(row["avg_bf"])
                rec["avg_pitches"] = float(row["avg_pitches"])
                rec["k_pct"]       = float(row["k_pct"])
            table[int(row["pitcher"])] = rec
    _PITCH_PROFILE_CACHE = table
    return table


def load_pitch_economy(path: str | None = None) -> dict:
    """Load pitcher → (9,) pitches-per-outcome vector (float64). Cached.

    Returns a dict keyed by integer pitcher id. Missing file → empty dict (every
    pitcher then falls back to the league PITCHES_PER_OUTCOME vector).
    """
    global _PITCH_ECONOMY_CACHE
    if _PITCH_ECONOMY_CACHE is not None:
        return _PITCH_ECONOMY_CACHE
    _PITCH_ECONOMY_CACHE = {pid: rec["ppo"]
                            for pid, rec in _load_pitch_profile(path).items()}
    return _PITCH_ECONOMY_CACHE


def pitcher_hook_params(pitcher_id: int | None) -> dict | None:
    """Build the full simulate_game pitcher-params dict for a starter.

    Combines the per-pitcher workload features (drive the hazard table) with the
    pitch-economy vector (drives in-game pitch accumulation). Returns None when
    the pitcher is unknown or has no profile → caller leaves params None and the
    sim falls back to the league-average hook (unchanged behavior).
    """
    if pitcher_id is None or int(pitcher_id) <= 0:
        return None
    rec = _load_pitch_profile().get(int(pitcher_id))
    if rec is None:
        return None
    params = {"pitcher_id": int(pitcher_id), "pitches_per_outcome": rec["ppo"]}
    if "avg_bf" in rec:
        params["pitcher_avg_bf"]      = rec["avg_bf"]
        params["pitcher_avg_pitches"] = rec["avg_pitches"]
        params["pitcher_k_pct"]       = rec["k_pct"]
    return params


def _ppo_canon(v) -> np.ndarray:
    """ONE canonical numba type for the pitches-per-outcome vector: fresh, WRITABLE, C-contiguous
    float64 (9,).

    WHY A COPY IS DELIBERATE HERE (2026-07-22 — this is a ~15-minute-per-restart bug, not a nicety).
    numba keys its compiled overloads on the full array type, and that type includes the NumPy
    ``writeable`` flag: ``array(float64, 1d, C)`` and ``readonly array(float64, 1d, C)`` are DIFFERENT
    signatures needing separate ~5-minute compiles on the 4-core server. resolve_pitch_economy used to
    return the parquet-backed cache vector directly for a KNOWN starter (read-only, because pandas
    hands back a read-only buffer) but a module constant for an UNKNOWN one (writable). With a home
    and an away starter that is 2x2 = 4 combinations, and a normal slate hits all four — so the maker
    recompiled `simulate_games` FOUR times on every cold start. Measured on the live server: 54 cached
    overload entries resolving to exactly 4 distinct signatures, flag combos RW/RW 19, RO/RO 18,
    RO/RW 9, RW/RO 8, and four fresh compiles between 19:44 and 19:55 on 2026-07-22.

    ``np.array(..., copy=True)`` (the default) always yields a fresh writable buffer, so every caller
    now presents the SAME type and one cached compile serves everything. The copy is 9 float64 = 72
    bytes per sim call — free. The kernel only READS this vector (indexing at the pitch-accumulation
    sites in both the main and resume loops), so making it writable changes no behavior and the
    values are untouched ⇒ sim output stays bit-identical.

    DO NOT apply the same copy to the exit LUT: that array is 33 MB, its type is already consistent,
    and copying it per call would cost far more than it saves.
    """
    return np.array(v, dtype=np.float64, order="C")


def resolve_pitch_economy(params: dict | None) -> np.ndarray:
    """Resolve a starter's (9,) pitches-per-outcome vector from sim params.

    Priority: explicit ``pitches_per_outcome`` vector → ``pitcher_id`` lookup in
    pitch_economy.parquet → league PITCHES_PER_OUTCOME fallback. Always returns a
    fresh writable contiguous float64 (9,) array — ONE numba type (see _ppo_canon).
    """
    if params is not None:
        ppo = params.get("pitches_per_outcome")
        if ppo is not None:
            return _ppo_canon(ppo)
        pid = params.get("pitcher_id")
        if pid is not None:
            vec = load_pitch_economy().get(int(pid))
            if vec is not None:
                return _ppo_canon(vec)
    return _ppo_canon(PITCHES_PER_OUTCOME)

# League-average PA outcome probabilities (2023–2026, post-rule-change era).
# Computed from pa_level_dataset.parquet filtered to game_date >= 2023-01-01,
# weighted by actual opp-handed / same-handed PA mix (54 % / 46 %).
# 2023 rule changes (pitch clock, shift ban, larger bases) make this era
# structurally different from 2019–2022; using earlier data inflates FO and
# understates GO by ~5 pp each.
# outcomes: K      BB     HBP    GO     FO     1B     2B     3B     HR
LEAGUE_AVG_PA = np.array(
    [0.2242, 0.0850, 0.0110, 0.2798, 0.1812, 0.1416, 0.0430, 0.0036, 0.0306],
    dtype=np.float32,
)

# 24 base-out states: state = runners_bitmask + outs * 8
N_STATES   = 24
INNINGS    = 9
N_SIMS_DEFAULT = 50_000
N_SIMS_DYNAMIC = 5_000    # for in-game dynamic baseline (speed-optimized)

# Bullpen leverage-deployment thresholds. A relief PA is "high leverage" when it
# is late (inning index >= HIGH_LEV_INNING_IDX; 6 == the 7th inning, 0-based) AND
# the score is close (|run differential| <= HIGH_LEV_MARGIN). In those spots the
# simulator deploys the team's high-leverage tier (closer/setup) instead of the
# general bullpen blend — capturing the run-suppression that protects late leads.
HIGH_LEV_INNING_IDX = 6
HIGH_LEV_MARGIN     = 3
# Blowout: when the score gap exceeds this, the manager deploys mop-up arms
# (and the pen-exhausted fallback recycles the worst-quality used arm).
BLOWOUT_MARGIN      = 6

# ── Individual-reliever bullpen manager ─────────────────────────────────────
# Role codes (match build_bullpen_profiles): the depth chart is ordered best→worst
# by quality, so index 0 = best arm. The manager deploys middle relief early,
# setup in the 7th–8th, the closer in 9th+ save situations, and mop-up in
# blowouts — reserving the closer/setup for late and respecting per-reliever
# stamina + the 3-batter minimum (stamina is clamped ≥3, so the minimum is
# automatically satisfied). Each arm is used once; a depleted pen falls back to
# the blended general vector.
ROLE_CLOSER, ROLE_SETUP, ROLE_MIDDLE, ROLE_LONG = 0, 1, 2, 3
SAVE_INNING_IDX = 8       # 0-based: inning index 8 == the 9th
SAVE_LEAD_MAX   = 3       # a save situation is a lead of 1..3 in the 9th+

# ── Leverage Index table (Tango), built by build_leverage_index.py from our own statcast.
# Module-level so the njit kernel reads it as a compile-time constant — no signature plumbing.
# Keyed IDENTICALLY to training via leverage_index.li_key (pitching-team frame).
from leverage_index import (load_li, N_INN as LI_N_INN, SD_OFF as LI_SD_OFF,
                            SD_MAX as LI_SD_MAX)  # noqa: E402
LI_TABLE = load_li()
# Innings 6-8 tiering for the deterministic path: how far ahead still counts as "protecting".
PROTECT_MIN_LEAD = 1      # leading by >=1 → escalate; trailing by >=2 → conserve
CONSERVE_MIN_DEF = 2


# ── MLB 5.10(g): when may a POSITION PLAYER pitch? ─────────────────────────────────────────
# Mirrored from build_position_player_arm.PP_* (the source of truth); a guard in build_bullpen_
# profiles asserts they agree. Verified against 2023-26 statcast entry states, which satisfy this
# rule 99-100% of the time (2023 99.1%, 2024 98.7%, 2025 100.0%, 2026 98.6%). Mirrored rather than
# imported because njit bakes module globals in at compile time.
PP_LOSING_BY  = 8      # trailing by at least this many
PP_WINNING_BY = 10     # leading by at least this many


def _pick_reliever(used_mask, n, rel_role, closer_idx, inning, save_sit, blowout, hi_lev,
                   lead=0, pp_block=-1):
    """Return the index of the next reliever to deploy, or −1 if the pen is spent.

    Arms are ordered best→worst quality (index 0 = best). njit-compatible: only
    ints, array indexing, and loops. inning is 0-based. hi_lev=1 marks a very
    high-leverage jam (runners on, late, close) → the best available "fireman" is
    summoned regardless of the inning-based role reservation (leverage-optimal).

    `lead` is the PITCHING team's SIGNED run differential. Before 2026-07-23 this path
    tiered purely by INNING, so in innings 6-8 a team protecting a lead and a team losing
    by the same margin deployed identical arms — no leader/trailer asymmetry, which is what
    reality expresses (leading team's mean arm quality 0.05232 vs trailing 0.04365). A
    TRAILING team now steps DOWN a tier in the 7th-8th (conserve its best arms) while a
    LEADING team keeps the setup escalation.
    """
    # MLB 5.10(g): pp_block >= 0 means the position-player arm is currently ILLEGAL. Marking it
    # "used" is exactly the semantics we want and needs no change to the eligibility loops below.
    # `used_mask` is an int (by value), so the caller's real used-arm state is unaffected.
    if pp_block >= 0:
        used_mask = used_mask | (1 << pp_block)
    if blowout == 1:                                   # mop-up: worst available
        for i in range(n - 1, -1, -1):
            if (used_mask >> i) & 1 == 0:
                return i
        return -1
    if save_sit == 1 and closer_idx >= 0 and ((used_mask >> closer_idx) & 1) == 0:
        return closer_idx                              # save → the closer
    if hi_lev == 1:                                    # high-LI jam → best available
        for i in range(n):
            if (used_mask >> i) & 1 == 0:
                return i
        return -1
    if inning >= SAVE_INNING_IDX:                      # 9th+: closer, else best
        if closer_idx >= 0 and ((used_mask >> closer_idx) & 1) == 0:
            return closer_idx
        for i in range(n):
            if (used_mask >> i) & 1 == 0:
                return i
        return -1
    if inning >= HIGH_LEV_INNING_IDX:                  # 7th–8th
        if lead <= -CONSERVE_MIN_DEF:
            # TRAILING by 2+: conserve the high-leverage arms — deploy the best MIDDLE/LONG
            # tier arm instead of the setup man (matches real managers, who do not burn
            # setup/closer chasing a deficit).
            for i in range(n):
                if ((used_mask >> i) & 1) == 0 and i != closer_idx and rel_role[i] >= ROLE_MIDDLE:
                    return i
        for i in range(n):                             # protecting / close: best non-closer
            if ((used_mask >> i) & 1) == 0 and i != closer_idx:
                return i
        if closer_idx >= 0 and ((used_mask >> closer_idx) & 1) == 0:
            return closer_idx
        return -1
    for i in range(n):                                 # early: middle/long, reserve late arms
        if ((used_mask >> i) & 1) == 0 and rel_role[i] >= ROLE_MIDDLE:
            return i
    for i in range(n):                                 # fallback: best available
        if (used_mask >> i) & 1 == 0:
            return i
    return -1


def _platoon_pick(base, used_mask, n, rel_role, closer_idx, inning, save_sit,
                  blowout, hi_lev, rel_throws, want_hand, pp_block=-1):
    """Handedness override (validated +2.0pt vs real managers): in NON-priority spots
    only, among eligible non-closer arms prefer one whose throwing hand matches the
    majority hand of the upcoming ≤3 batters. Priority spots (save/blowout/fireman/9th+)
    and want_hand<0 (unknown) defer to `base`. njit-compatible: ints + loops only."""
    # MLB 5.10(g): pp_block >= 0 means the position-player arm is currently ILLEGAL. Marking it
    # "used" is exactly the semantics we want and needs no change to the eligibility loops below.
    # `used_mask` is an int (by value), so the caller's real used-arm state is unaffected.
    if pp_block >= 0:
        used_mask = used_mask | (1 << pp_block)
    if (base < 0 or save_sit == 1 or blowout == 1 or hi_lev == 1
            or inning >= SAVE_INNING_IDX or want_hand < 0):
        return base
    for i in range(n):                                 # best-quality eligible platoon match
        if ((used_mask >> i) & 1) == 0 and i != closer_idx and rel_throws[i] == want_hand:
            return i
    return base


# ── CLF CONTRACT (2026-08-07) ────────────────────────────────────────────────────────────────
# _clf_choose below indexes clf_w[0..CLF_N_INT-1] and rel_clf[:, 0..CLF_N_RELCLF_COLS-1]. @njit
# runs with bounds-checking OFF, so a builder emitting NARROWER arrays is a SILENT out-of-bounds
# read — garbage weights on every reliever pick, no exception, no log line, and a successful sim
# proves nothing. These constants are the kernel's DECLARED requirement; build_bullpen_profiles
# (_CLF_INT_ORDER length, rel_clf column count) must satisfy them, and the gate in
# _make_simulate_games_numba's caller checks it before the learned picker is ever enabled.
# They live HERE, beside the indexing they describe, because the dangerous direction is
# new-kernel/old-builder — a check living in the builder could never fire for it.
CLF_N_INT = 17          # == len(_CLF_INT_ORDER) in build_bullpen_profiles
CLF_N_RELCLF_COLS = 11  # == rel_clf column count in build_bullpen_profiles


def _clf_choose(used_mask, n, rel_clf, rel_throws, clf_w,
                cur_inn, save_sit, blowout, late9, close_late,
                li, protect, deficit, jam, want, udraw, pp_block=-1):
    """LEARNED reliever pick (conditional logit, see build_reliever_choice_model.py):
    sample an eligible arm from softmax_j(u_j), where
        u_j = clf_base_j + Σ_k clf_w[k]·(arm_j × state interaction_k).
    rel_clf[i] = [clf_base, quality, avg_entry, late_entries, stamina, is_closer, krate]
    (clf_base already folds the state-independent main effects). `udraw` is a
    uniform(0,1) drawn by the caller's RNG (xorshift in numba, rng.random in numpy)
    so the outcome stream stays seed-reproducible. Returns the chosen arm index, or
    −1 if no arm is eligible. njit-compatible: only loops, np.empty, np.exp.

    STATE INPUTS (order mirrors _CLF_INT_ORDER in build_bullpen_profiles.py):
      li               Leverage Index (Tango) — situational IMPORTANCE, carries base-out.
      protect/deficit  SIGNED lead split: max(0,lead) / max(0,−lead). These replace the old
                       symmetric `abslead`, which made a team protecting a 3-run lead and a
                       team losing by 3 indistinguishable — zeroing the leader/trailer
                       asymmetry that reality shows (+0.0087 runs/PA) and costing ~2/3 of the
                       missing late-margin amplification.
      jam              base-out pressure (≥2 on, ≤1 out) → pairs with the arm's K rate.
    LI is applied DIRECTIONALLY: it is near-symmetric in the lead's sign, so one shared q×li term
    escalated BOTH teams in a close game and partly cancelled q_x_deficit — measurably leaving the
    trailing team un-suppressed. Partitioned on the sign (protect / tied / trail) the model can
    spend leverage when defending and hold arms back when chasing. Exactly one is non-zero."""
    # MLB 5.10(g): pp_block >= 0 means the position-player arm is currently ILLEGAL. Marking it
    # "used" is exactly the semantics we want and needs no change to the eligibility loops below.
    # `used_mask` is an int (by value), so the caller's real used-arm state is unaffected.
    if pp_block >= 0:
        used_mask = used_mask | (1 << pp_block)
    li_p = li if protect > 0.0 else 0.0
    li_t = li if (protect <= 0.0 and deficit <= 0.0) else 0.0
    li_d = li if deficit > 0.0 else 0.0
    u = np.empty(n, dtype=np.float64)
    umax = -1.0e18
    for i in range(n):
        if (used_mask >> i) & 1:
            u[i] = -1.0e18
            continue
        base = rel_clf[i, 0]; q = rel_clf[i, 1]; ae = rel_clf[i, 2]
        le = rel_clf[i, 3]; st = rel_clf[i, 4]; isc = rel_clf[i, 5]; kr = rel_clf[i, 6]
        # per-club random effects (build_manager_effects.py): deviations ADDED to the pooled
        # weight on the three philosophy axes. Zero ⇒ byte-identical to the pooled model.
        dv_lip = rel_clf[i, 7]; dv_def = rel_clf[i, 8]; dv_sav = rel_clf[i, 9]
        hm = 1.0 if (want >= 0 and rel_throws[i] == want) else 0.0
        d = cur_inn - ae
        ui = (base
              + clf_w[0] * (-(d * d))
              + (clf_w[1] + dv_sav) * isc * save_sit
              + clf_w[2] * isc * late9
              + (clf_w[3] + dv_lip) * q * li_p
              + clf_w[4] * q * li_t
              + clf_w[5] * q * li_d
              + clf_w[6] * q * protect
              + (clf_w[7] + dv_def) * q * deficit
              + clf_w[8] * q * blowout
              + clf_w[9] * st * blowout
              + clf_w[10] * hm
              + clf_w[11] * hm * close_late
              + clf_w[12] * le * late9
              + clf_w[13] * kr * jam
              # MLB 5.10(g) legalises a position player in EXTRA innings, but managers essentially
              # never do it (extras are tied, not decided). rel_clf col 10 = is_position_player.
              + clf_w[14] * rel_clf[i, 10] * (1.0 if cur_inn > 9 else 0.0)
              # ASC gradient: mopping up a WON game vs conceding a LOST one are different calls.
              + clf_w[15] * rel_clf[i, 10] * protect
              + clf_w[16] * rel_clf[i, 10] * deficit)
        u[i] = ui
        if ui > umax:
            umax = ui
    if umax <= -1.0e17:
        return -1
    s = 0.0
    for i in range(n):
        if u[i] > -1.0e17:
            e = np.exp(u[i] - umax)
            u[i] = e
            s += e
        else:
            u[i] = 0.0
    target = udraw * s
    acc = 0.0
    for i in range(n):
        acc += u[i]
        if target < acc:
            return i
    for i in range(n - 1, -1, -1):     # fp guard: return the last eligible arm
        if u[i] > 0.0:
            return i
    return -1


# DEPRECATED (removed from the engine 2026-05): the per-PA baserunner-advancement
# fudge (wild pitches / passed balls / stolen bases / reached-on-error) is no
# longer used. All inter-PA advancement is now captured directly by the EMPIRICAL
# base-out transition tables (build_empirical_transitions.py), whose post_state is
# the NEXT PA's pre_state and therefore already reflects SB/WP/PB/error movement.
# Reached-on-error is folded into the empirical GO transition. Nothing is hand-tuned.

# ── Reliever removal-hazard table dims (train_reliever_removal_model.py) ─────
# Indexed [batters_faced_this_outing, runs_allowed_this_outing, inning-1, sd+OFF].
# 3-batter-minimum floor is baked into the table (hazard≈0 for bf<2). One league
# table shared by both teams (reliever pull dynamics aren't team-specific).
REL_BF_MAX  = 15
REL_ER_MAX  = 5
REL_INN_MAX = 11
REL_SD_OFF  = 8
REL_SD_MAX  = 16
REL_HAZARD_SHAPE = (REL_BF_MAX + 1, REL_ER_MAX + 1, REL_INN_MAX, REL_SD_MAX + 1)


# ═══════════════════════════════════════════════════════════════
# BASE RUNNING TRANSITION TABLES
# ═══════════════════════════════════════════════════════════════

def build_baserunning_tables(
    statcast_path=None,
):
    """
    Build league-average base running transition tables with stochastic advancement.

    Returns FIVE arrays. For each (state, outcome) pair the simulation draws:
      1. Main outcome (deterministic part): new_state_table, runs_table
      2. Optional stochastic branch (one additional RNG draw per PA):
           if u2 < stoc_prob -> use (stoc_new_state, stoc_runs)
           else              -> use (new_state_table, runs_table)

    EMPIRICAL PROBABILITIES (Retrosheet 2000-2024, Tango "The Book",
                              FanGraphs RE matrix 2021-2024):

    Single (1B):
      - Runner on 3B:     always scores (deterministic, 100%)
      - Runner on 2B:     scores with P=0.63 (STOCHASTIC)
                          Source: Tango/Retrosheet empirical
      - Runner on 1B:     advances to 2B (deterministic; 73% dominant outcome)

    Double (2B):
      - Runner on 3B:     always scores (100%)
      - Runner on 2B:     always scores (100%)
      - Runner on 1B:     scores with P=0.44 (STOCHASTIC)

    Ground out (GO):
      - Runner on 3B, 0 outs: scores with P=0.20 (STOCHASTIC)
      - Runner on 3B, 1 out:  scores with P=0.12 (STOCHASTIC)
      - 2 outs: inning ends, no scoring

    Fly out (FO):
      - Runner on 3B, <2 outs: scores with P=0.50 (STOCHASTIC sac fly)
      - 2 outs: inning ends, no scoring

    Source: Bukiet et al (1997), Tallavarjula (2026 SAGE), Tango "The Book",
            FanGraphs/Retrosheet play-by-play data 2019-2024.

    Returns:
        new_state_table: (24, 9) int8  - base (non-stochastic) new state
        runs_table:      (24, 9) int8  - base runs scored
        stoc_new_state:  (24, 9) int8  - state when stochastic fires
        stoc_runs:       (24, 9) int8  - runs when stochastic fires
        stoc_prob:       (24, 9) float32 - P(stochastic fires); 0.0=deterministic
    """
    def has_first(s):  return bool(s & 1)
    def has_second(s): return bool(s & 2)
    def has_third(s):  return bool(s & 4)
    def outs_of(s):    return s >> 3

    new_state      = np.zeros((N_STATES, N_OUTCOMES), dtype=np.int8)
    runs           = np.zeros((N_STATES, N_OUTCOMES), dtype=np.int8)
    stoc_new_state = np.zeros((N_STATES, N_OUTCOMES), dtype=np.int8)
    stoc_runs      = np.zeros((N_STATES, N_OUTCOMES), dtype=np.int8)
    stoc_prob      = np.zeros((N_STATES, N_OUTCOMES), dtype=np.float32)

    K_IDX   = OUTCOME_TO_IDX["K"]
    BB_IDX  = OUTCOME_TO_IDX["BB"]
    HBP_IDX = OUTCOME_TO_IDX["HBP"]
    GO_IDX  = OUTCOME_TO_IDX["GO"]
    FO_IDX  = OUTCOME_TO_IDX["FO"]
    S1_IDX  = OUTCOME_TO_IDX["1B"]
    S2_IDX  = OUTCOME_TO_IDX["2B"]
    S3_IDX  = OUTCOME_TO_IDX["3B"]
    HR_IDX  = OUTCOME_TO_IDX["HR"]

    # Stochastic probabilities from empirical literature
    P_S1_B2_SCORES = 0.63   # runner on 2B scores on single (Tango/Retrosheet)
    P_S1_B1_TO_3B  = 0.28   # runner on 1B advances 1st→3rd on single (~28%, Hardball Times/Retrosheet)
    P_S2_B1_SCORES = 0.44   # runner on 1B scores on double
    P_GO_B3_0OUT   = 0.20   # runner on 3B scores on GO, 0 outs
    P_GO_B3_1OUT   = 0.12   # runner on 3B scores on GO, 1 out
    P_FO_B3_SCORES = 0.50   # runner on 3B scores on FO (avg sac fly rate, <2 outs)

    def set_det(s, oi, ns, r):
        """Set deterministic transition (stoc_prob=0)."""
        new_state[s, oi]      = ns
        runs[s, oi]           = r
        stoc_new_state[s, oi] = ns
        stoc_runs[s, oi]      = r
        stoc_prob[s, oi]      = 0.0

    def set_stoc(s, oi, ns_base, r_base, ns_stoc, r_stoc, p):
        """Set stochastic transition."""
        new_state[s, oi]      = ns_base
        runs[s, oi]           = r_base
        stoc_new_state[s, oi] = ns_stoc
        stoc_runs[s, oi]      = r_stoc
        stoc_prob[s, oi]      = p

    for s in range(N_STATES):
        o  = outs_of(s)
        b1 = has_first(s)
        b2 = has_second(s)
        b3 = has_third(s)

        # ── Strikeout: purely deterministic ────────────────────
        new_o = o + 1
        ns = 24 if new_o >= 3 else (int(b1) + int(b2)*2 + int(b3)*4) + new_o*8
        set_det(s, K_IDX, ns, 0)

        # ── Walk / HBP: deterministic force advances ────────────
        for oi in [BB_IDX, HBP_IDX]:
            r = 0
            if b1 and b2 and b3:
                nb1, nb2, nb3 = 1, 1, 1;  r = 1
            elif b1 and b2:
                nb1, nb2, nb3 = 1, 1, 1
            elif b1:
                nb1, nb2, nb3 = 1, 1, int(b3)
            else:
                nb1, nb2, nb3 = 1, int(b2), int(b3)
            set_det(s, oi, nb1 + nb2*2 + nb3*4 + o*8, r)

        # ── Ground out ─────────────────────────────────────────
        # Priority:
        #   1. With 2 outs: inning ends (deterministic, no DP possible)
        #   2. b3=1, <2 outs: runner on 3B may score (stochastic, existing)
        #      DP not modeled when b3=1 — can't stack two stochastic events
        #   3. b1=1, b3=0, <2 outs: double play possible (NEW stochastic)
        #      P(DP) = 0.42 — Retrosheet empirical 2019-2024; Tallavarjula (2026)
        #      DP: +2 outs, b1 retired, b2 advances one base (if present)
        #      Base: +1 out, b1 stays (standard groundout)
        #   4. All other cases: deterministic +1 out

        # P(double play | model-GO, b1=1, b3=0, <2 outs), MEASURED from 2023+ Statcast
        # (n=26,636 such groundouts, using the exact L3 GO definition incl. force_out/
        # fielders_choice): 0.353 at 0 outs, 0.390 at 1 out. (NOT a guess — earlier 0.22 was
        # wrong; the original 0.42 was close but slightly high.)
        P_GO_DP = 0.353 if o == 0 else 0.390

        new_o = o + 1
        if new_o >= 3:
            # Already 2 outs: inning ends, no DP possible
            set_det(s, GO_IDX, 24, 0)
        elif b3:
            # Runner on 3B (b3=1): stochastic scoring, no DP modeled
            p = P_GO_B3_0OUT if o == 0 else P_GO_B3_1OUT
            ns_base = int(b1) + int(b2)*2 + 1*4 + new_o*8  # b3 stays
            ns_stoc = int(b1) + int(b2)*2 + 0*4 + new_o*8  # b3 scores
            set_stoc(s, GO_IDX, ns_base, 0, ns_stoc, 1, p)
        elif b1:
            # Runner on 1B, no runner on 3B: double play stochastic
            # Base (p=0.58): standard GO, +1 out, b1 advances to 2B.
            # Fielder threw to 1B to retire the batter, so b1 had to vacate
            # 1B and advances freely to 2B; b2 (if present) advances to 3B.
            ns_base = 0 + 1*2 + int(b2)*4 + new_o*8         # b1 → 2B, b2 → 3B
            # Stochastic DP (p=0.42): +2 outs, b1 retired
            #   b2 advances to 3B (if present), otherwise bases empty
            dp_outs = o + 2
            if dp_outs >= 3:
                ns_dp = 24   # inning ends on DP
            else:
                # b2 advances to 3B on the DP (fielder retires b1 at 2B,
                # b2 has time to advance); bases otherwise clear
                ns_dp = 0 + int(b2)*4 + dp_outs*8   # b2 -> 3B (bit2), b1 cleared
            set_stoc(s, GO_IDX, ns_base, 0, ns_dp, 0, P_GO_DP)
        elif b2:
            # b2 only (no b1 to force, no b3): productive out — b2 advances 2B→3B
            # ~50% of the time (grounder to the right side); else holds at 2B.
            ns_hold = 0 + 1*2 + 0*4 + new_o*8   # b2 holds at 2B
            ns_adv  = 0 + 0*2 + 1*4 + new_o*8   # b2 -> 3B
            set_stoc(s, GO_IDX, ns_hold, 0, ns_adv, 0, 0.50)
        else:
            # bases empty: deterministic standard GO
            set_det(s, GO_IDX, new_o*8, 0)

        # ── Fly out ────────────────────────────────────────────
        new_o = o + 1
        if new_o >= 3:
            set_det(s, FO_IDX, 24, 0)
        elif b3:
            # Stochastic: sac fly with P=0.50
            ns_base = int(b1) + int(b2)*2 + 1*4 + new_o*8  # b3 stays
            ns_stoc = int(b1) + int(b2)*2 + 0*4 + new_o*8  # b3 scores
            set_stoc(s, FO_IDX, ns_base, 0, ns_stoc, 1, P_FO_B3_SCORES)
        else:
            set_det(s, FO_IDX, int(b1) + int(b2)*2 + new_o*8, 0)

        # ── Single ─────────────────────────────────────────────
        # b3 always scores. Runner advancement (empirical, ~28% go 1st→3rd):
        #   b2 present: stoc(0.63) b2 scores AND b1→3B (aggressive read); base b2→3B, b1→2B.
        #   b1 only:    stoc(0.28) b1→3B (first-to-third); base b1→2B.
        r_base = int(b3)   # b3 always scores
        if b2:
            ns_base = 1 + int(b1)*2 + 1*4 + o*8          # base: b2→3B, b1→2B, batter→1B
            ns_stoc = 1 + 0*2      + int(b1)*4 + o*8     # stoc: b2 scores, b1→3B, batter→1B
            set_stoc(s, S1_IDX, ns_base, r_base, ns_stoc, r_base + 1, P_S1_B2_SCORES)
        elif b1:
            ns_base = 1 + 1*2 + 0*4 + o*8                # base: b1→2B, batter→1B
            ns_3rd  = 1 + 0*2 + 1*4 + o*8                # stoc: b1→3B (first-to-third), batter→1B
            set_stoc(s, S1_IDX, ns_base, r_base, ns_3rd, r_base, P_S1_B1_TO_3B)
        else:
            # No b1/b2: batter→1B (b3 already scored via r_base).
            set_det(s, S1_IDX, 1 + o*8, r_base)

        # ── Double ─────────────────────────────────────────────
        # b3 and b2 always score. b1: stochastic (P=0.44 scores).
        r_base = int(b3) + int(b2)

        if b1:
            # Base (p=0.56): b1 -> 3B, batter -> 2B
            ns_base = 0 + 1*2 + 1*4 + o*8   # 2B+3B
            # Stoc (p=0.44): b1 scores, batter -> 2B only
            ns_stoc = 0 + 1*2 + 0*4 + o*8   # 2B only
            set_stoc(s, S2_IDX, ns_base, r_base, ns_stoc, r_base+1, P_S2_B1_SCORES)
        else:
            ns = 0 + 1*2 + 0*4 + o*8
            set_det(s, S2_IDX, ns, r_base)

        # ── Triple: all runners score, batter to 3B. Deterministic. ───
        r = int(b1) + int(b2) + int(b3)
        set_det(s, S3_IDX, 4 + o*8, r)

        # ── Home run: all runners + batter score. Deterministic. ───────
        r = int(b1) + int(b2) + int(b3) + 1
        set_det(s, HR_IDX, 0 + o*8, r)

    return new_state, runs, stoc_new_state, stoc_runs, stoc_prob



def save_baserunning_tables(new_state, runs, stoc_new_state, stoc_runs, stoc_prob,
                             out_dir: str) -> None:
    """Save all 5 transition tables to disk for fast loading."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "br_new_state.npy"),      new_state)
    np.save(os.path.join(out_dir, "br_runs.npy"),           runs)
    np.save(os.path.join(out_dir, "br_stoc_new_state.npy"), stoc_new_state)
    np.save(os.path.join(out_dir, "br_stoc_runs.npy"),      stoc_runs)
    np.save(os.path.join(out_dir, "br_stoc_prob.npy"),      stoc_prob)


def load_baserunning_tables(cache_dir=None, suffix=""):
    """Load the EMPIRICAL base-running transition tables (3-tuple).
    suffix="_nosteal" loads the steal-excluded tables for the explicit-steal running-game path.

    These are measured directly from Statcast play-by-play by
    build_empirical_transitions.py — one categorical per (pre_state, outcome)
    over the (post_state, runs) outcomes actually observed, kept as the top-K
    most common transitions. This replaces ALL hand-coded baserunning
    probabilities (DP rate, first-to-third, sac-fly advancement) AND the
    P_EXTRA_ADV WP/PB/SB/error fudge, since inter-PA advancement is already
    baked into the observed post_state.

    Returns (emp_post_state, emp_runs, emp_cumprob), each (24, 9, K):
        emp_post_state : int8   resulting base-out state (24 = inning over)
        emp_runs       : int8   runs scored on the transition
        emp_cumprob    : float32 cumulative probability over the K transitions
    """
    proc = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "processed")
    post = np.load(os.path.join(proc, f"emp_post_state{suffix}.npy"))
    runs = np.load(os.path.join(proc, f"emp_runs{suffix}.npy"))
    cum  = np.load(os.path.join(proc, f"emp_cumprob{suffix}.npy"))
    return (np.ascontiguousarray(post, dtype=np.int8),
            np.ascontiguousarray(runs, dtype=np.int8),
            np.ascontiguousarray(cum,  dtype=np.float32))


# ═══════════════════════════════════════════════════════════════
# OUTCOME PROBABILITY ARRAYS
# ═══════════════════════════════════════════════════════════════

# ── Secular BABIP correction (2026 hot ball). The de-luck xBA/xwOBAcon anchor strips the current-season
# BABIP the posterior can't see (model ~.281 vs realized ~.289). HR gets its secular re-addition via the
# season recal's de-weathered target; CONTACT never did (BIP stays w=1, "owned by the de-luck anchor").
# We add it HERE — on the FINAL per-batter probs, after contact-geo/GF — as a BIP-PRESERVING hit/out ODDS
# multiplier. Predictive-basis by construction, so it CANNOT re-amplify downstream the way the recal-basis
# k_bb_hr_bip did (which overshot +0.172). 1.0 = OFF (golden-master bit-identical).
# Durable: nightly recompute = recency-weighted realized BABIP / model de-luck BABIP (env is the interim).
def _load_babip_odds_mult() -> float:
    """SERVING reads the nightly artifact data/processed/babip_correction.json (build_babip_correction.py);
    the BABIP_ODDS_MULT env OVERRIDES it for research/backtest (=1.0 ⇒ golden-master OFF, snapshots set it
    explicitly). Any error ⇒ 1.0 (OFF, safe)."""
    _env = os.environ.get("BABIP_ODDS_MULT")
    if _env is not None:
        return float(_env)
    try:
        import json as _json
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "babip_correction.json")
        with open(_p) as _f:
            return float(_json.load(_f)["k"])
    except Exception:
        return 1.0


_BABIP_ODDS_MULT = _load_babip_odds_mult()


def build_pa_prob_arrays(
    lineup: list[dict],
    game_probs: dict,
) -> np.ndarray:
    """
    Convert precomputed_game() output to a contiguous float32 array
    for the numba inner loop.

    Args:
        lineup:     list of {"batter_id": int, "handedness": int, "batting_order": int}
        game_probs: dict from Level3Predictor.precompute_game()

    Returns:
        pa_probs: (n_batters, N_OUTCOMES) float32 array
                  row i = probability vector for batter at position i in lineup
    """
    n = len(lineup)
    pa_probs = np.zeros((n, N_OUTCOMES), dtype=np.float32)
    for i, spec in enumerate(lineup):
        bid = spec["batter_id"]
        h   = spec["handedness"]
        if bid in game_probs and h in game_probs[bid]:
            probs_dict = game_probs[bid][h]
        elif bid in game_probs:
            # Fallback: use whichever handedness is available
            probs_dict = next(iter(game_probs[bid].values()))
        else:
            # Should not happen if precompute_game() was called correctly
            raise KeyError(f"Batter {bid} not found in game_probs")
        for j, o in enumerate(OUTCOMES):
            pa_probs[i, j] = float(probs_dict[o])
    if _BABIP_ODDS_MULT != 1.0:
        # Multiply the hit-vs-out ODDS within balls-in-play by _BABIP_ODDS_MULT, PRESERVING the BIP
        # total — so K/BB/HBP/HR and the BIP share of PAs are unchanged, and each batter's relative
        # hit-type mix and BABIP skill are preserved (uniform shift in odds space). GO=3,FO=4 (outs);
        # 1B=5,2B=6,3B=7 (hits on BIP).
        k   = _BABIP_ODDS_MULT
        hit = pa_probs[:, 5] + pa_probs[:, 6] + pa_probs[:, 7]
        out = pa_probs[:, 3] + pa_probs[:, 4]
        bip = hit + out
        ok  = (out > 1e-9) & (hit > 1e-9)
        new_odds = k * (hit / np.where(ok, out, 1.0))
        new_hit  = bip * new_odds / (1.0 + new_odds)
        hf = np.where(ok, new_hit / np.where(ok, hit, 1.0), 1.0).astype(np.float32)
        of = np.where(ok, (bip - new_hit) / np.where(ok, out, 1.0), 1.0).astype(np.float32)
        pa_probs[:, 5] *= hf; pa_probs[:, 6] *= hf; pa_probs[:, 7] *= hf
        pa_probs[:, 3] *= of; pa_probs[:, 4] *= of
    return pa_probs


def build_cumulative_probs(pa_probs: np.ndarray) -> np.ndarray:
    """
    Convert probability arrays to cumulative form for fast multinomial sampling.

    Random sampling trick: draw u ~ Uniform(0,1), find first index where
    cumsum > u. This is faster than np.random.choice inside numba.

    Returns:
        cum_probs: (n_batters, N_OUTCOMES) float32 — cumulative sums per row
    """
    return np.cumsum(pa_probs, axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# TIMES-THROUGH-THE-ORDER (TTOP) — starter per-PA decay
# ═══════════════════════════════════════════════════════════════
#
# The starter is tougher his 1st time through the order and worse his 3rd
# (measured 2024-26: +0.010 / +0.022 wOBA). We apply mean-preserving per-TTO
# multipliers to the starter's matchup vector so the redistribution leaves each
# pitcher's calibrated overall rate (a_hat_p) unchanged — it only changes WHEN
# the damage happens (critical for the live in-game model).

_TTOP_MULT = None  # (3, N_OUTCOMES) cached multipliers; identity if no params file


def _load_ttop_mult() -> np.ndarray:
    global _TTOP_MULT
    if _TTOP_MULT is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "processed", "ttop_params.json")
        try:
            with open(path) as f:
                m = np.array(json.load(f)["mult"], dtype=np.float32)
            if m.shape != (3, N_OUTCOMES):
                raise ValueError("bad ttop_params shape")
            _TTOP_MULT = m
        except Exception:
            _TTOP_MULT = np.ones((3, N_OUTCOMES), dtype=np.float32)  # TTOP disabled
    return _TTOP_MULT


# ── Tier-1 PA-dependence: continuous TTOP curve + base-state multipliers ─────
# Both are mean-preserving log-odds tilts on the matchup vector, ESTIMATED by
# estimate_pa_dependence.py and consumed here. Env kill-switches:
#   TTOP_CONTINUOUS=0 → fall back to the discrete 3-level step (legacy)
#   BASE_STATE=0       → identity base-state (off)
N_BF_AXIS = 31                                          # starter batters-faced axis
N_BUCKET  = 3                                           # base-state buckets
# pre-PA base occupancy (state & 7, bits = 1B/2B/3B) → strategic bucket:
#   0 empty | 1 1B-occupied (no open-base IBB incentive) | 2 RISP w/ 1B OPEN
BASE_BUCKET_LUT = np.array([0, 1, 2, 1, 2, 1, 2, 1], dtype=np.int8)

_TTOP_CURVE = None   # (N_BF_AXIS, N_OUTCOMES)
_BASE_MULT  = None   # (N_BUCKET, N_OUTCOMES)


def _load_ttop_curve() -> np.ndarray:
    """Continuous per-bf TTOP multiplier (N_BF_AXIS, 9), from estimate_pa_dependence.py.
    DEFAULT ON + artifact REQUIRED: when TTOP_CONTINUOUS != 0 the ttop_continuous.json artifact
    is HARD-REQUIRED (a missing/malformed file RAISES — no silent identity, so a broken serving
    tree is caught loudly). Explicit TTOP_CONTINUOUS=0 = the golden-master escape hatch: fall back
    to the discrete 3-level step (identity when that too is absent)."""
    global _TTOP_CURVE
    if _TTOP_CURVE is None:
        if os.environ.get("TTOP_CONTINUOUS", "1") != "0":
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "processed", "ttop_continuous.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TTOP_CONTINUOUS is ON (default) but {path} is missing. Run "
                    "estimate_pa_dependence.py (wired in finalize_fit) or set TTOP_CONTINUOUS=0 to disable.")
            m = np.array(json.load(open(path))["mult"], dtype=np.float64)
            if m.shape != (N_BF_AXIS, N_OUTCOMES):
                raise ValueError(f"ttop_continuous.json mult shape {m.shape} != {(N_BF_AXIS, N_OUTCOMES)}")
            _TTOP_CURVE = m
        else:                                            # TTOP_CONTINUOUS=0 → discrete 3-level → step to N_BF
            disc = _load_ttop_mult()                     # (3,9), identity if absent
            curve = np.empty((N_BF_AXIS, N_OUTCOMES), dtype=np.float64)
            for b in range(N_BF_AXIS):
                curve[b] = disc[min(b // 9, 2)]
            _TTOP_CURVE = curve
    return _TTOP_CURVE


_TTOP_CURVE_FRINGE = False   # (N_BF_AXIS, N_OUTCOMES) steeper curve for fringe/unproven starters; None→use global


def _load_ttop_curve_fringe() -> np.ndarray | None:
    """Steeper TTOP curve for FRINGE/unproven starters (build_fringe_ttop.py). diag_ttop_leash: fringe K
    declines ~−26% 1st→3rd vs ~−18% established. DEFAULT ON + artifact REQUIRED: when FRINGE_TTOP != 0 the
    ttop_continuous_fringe.json artifact is HARD-REQUIRED (missing/malformed RAISES). Explicit FRINGE_TTOP=0
    = golden-master escape hatch → None (caller uses the global curve for everyone)."""
    global _TTOP_CURVE_FRINGE
    if _TTOP_CURVE_FRINGE is False:
        if os.environ.get("FRINGE_TTOP", "1") != "0":
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "processed", "ttop_continuous_fringe.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FRINGE_TTOP is ON (default) but {path} is missing. Run build_fringe_ttop.py "
                    "(wired in finalize_fit) or set FRINGE_TTOP=0 to disable.")
            m = np.array(json.load(open(path))["mult"], dtype=np.float64)
            if m.shape != (N_BF_AXIS, N_OUTCOMES):
                raise ValueError(f"ttop_continuous_fringe.json mult shape {m.shape} != {(N_BF_AXIS, N_OUTCOMES)}")
            _TTOP_CURVE_FRINGE = m
        else:
            _TTOP_CURVE_FRINGE = None
    return _TTOP_CURVE_FRINGE


def _load_base_mult() -> np.ndarray:
    """Base-state multiplier (N_BUCKET, 9), from estimate_pa_dependence.py. DEFAULT ON + artifact
    REQUIRED: when BASE_STATE != 0 the base_state_params.json artifact is HARD-REQUIRED (missing/malformed
    RAISES — no silent identity). Explicit BASE_STATE=0 = golden-master escape hatch → identity ones."""
    global _BASE_MULT
    if _BASE_MULT is None:
        if os.environ.get("BASE_STATE", "1") != "0":
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "processed", "base_state_params.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"BASE_STATE is ON (default) but {path} is missing. Run estimate_pa_dependence.py "
                    "(wired in finalize_fit) or set BASE_STATE=0 to disable.")
            mm = np.array(json.load(open(path))["mult"], dtype=np.float64)
            if mm.shape != (N_BUCKET, N_OUTCOMES):
                raise ValueError(f"base_state_params.json mult shape {mm.shape} != {(N_BUCKET, N_OUTCOMES)}")
            _BASE_MULT = mm
        else:
            _BASE_MULT = np.ones((N_BUCKET, N_OUTCOMES), dtype=np.float64)
    return _BASE_MULT


_GAME_FORM = None


def _load_game_form():
    """Latent game-form factor: loadings L (9×n_factors) + sigma_diag (9) from
    build_game_form_factor.py. Absent → (None, None) → factor disabled."""
    global _GAME_FORM
    if _GAME_FORM is None:
        L = sd = None
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "processed", "game_form_factor.json")
            d = json.load(open(path))
            L  = np.asarray(d["L"], dtype=np.float64)            # (9, n_factors)
            sd = np.asarray(d["sigma_diag"], dtype=np.float64)   # (9,)
        except Exception:
            pass
        _GAME_FORM = (L, sd)
    return _GAME_FORM


_GAME_FORM_SPLIT = None


def _load_game_form_split():
    """De-confounded TWO-SHOCK loadings (build_gf_split.py): L_offense (from Σ_O, the offense
    day-form) + L_starter (from Σ_P, the pitcher/starter day-form), each with its sigma_diag.
    Returns (L_O, sd_O, L_P, sd_P) or None if the split file is absent (⇒ caller falls back to the
    single confounded L for both → golden-master safe)."""
    global _GAME_FORM_SPLIT
    if _GAME_FORM_SPLIT is None:
        res = False
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "processed", "game_form_factor_split.json")
            d = json.load(open(path))
            res = (np.asarray(d["L_offense"], np.float64), np.asarray(d["sdiag_offense"], np.float64),
                   np.asarray(d["L_starter"], np.float64), np.asarray(d["sdiag_starter"], np.float64))
        except Exception:
            res = False
        _GAME_FORM_SPLIT = res
    return _GAME_FORM_SPLIT or None


_GF_EXTRA = None


def _load_gf_extra():
    """(hr_logsd, k_logsd, bb_logsd) — per-GAME log-mult SDs for the two extra day-form channels:
      • CARRY → HR (Σ_carry, build_sigma_carry; beyond park+weather so it's NOT double-counting the
        pregame weather tilt), and
      • UMPIRE → K/BB (Σ_umpire, ABS-2026-adjusted; converted via the count leverage T_K/T_BB).
    Both are GAME-level (shared by both teams / both pitchers). The two channels load INDEPENDENTLY, each
    on its own gate:
      • GF_CARRY  default ON → sigma_carry.json is HARD-REQUIRED (missing RAISES; explicit GF_CARRY=0 → hr=0).
      • GF_UMPIRE default ON → sigma_umpire.json is loaded when present; it depends on the statsapi
        game_umpire fetch (build_sigma_umpire, a known blocker when that map is absent), so an ABSENT file is
        tolerated (→ k/bb=0, channel identity). A PRESENT-but-malformed file RAISES. Explicit GF_UMPIRE=0 → k/bb=0.
    Returns None only when BOTH channels are explicitly disabled AND nothing to load."""
    global _GF_EXTRA
    if _GF_EXTRA is None:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
        hr_logsd = k_logsd = bb_logsd = 0.0
        # ── CARRY channel (GF_CARRY): default ON, artifact hard-required ──
        if float(os.environ.get("GF_CARRY", "1.0")) > 0.0:
            cpath = os.path.join(base, "sigma_carry.json")
            if not os.path.exists(cpath):
                raise FileNotFoundError(
                    f"GF_CARRY is ON (default) but {cpath} is missing. Run build_sigma_carry.py "
                    "(wired in finalize_fit) or set GF_CARRY=0 to disable the carry→HR day-form channel.")
            hr_logsd = float(json.load(open(cpath)).get("hr_mult_logsd", 0.0))
        # ── UMPIRE channel (GF_UMPIRE): default ON; artifact required only if present (blocker-tolerant) ──
        if float(os.environ.get("GF_UMPIRE", "1.0")) > 0.0:
            upath = os.path.join(base, "sigma_umpire.json")
            if os.path.exists(upath):
                ump = json.load(open(upath))
                # k_mult_logsd / bb_mult_logsd are the FINAL per-PA K/BB log-mult SDs of the umpire day-form:
                # build_sigma_umpire ALREADY bakes in BOTH the count leverage AND the ABS-2026 retention
                # (sig_abs = sig·abs_retention → dK = sig_abs·spp·T_K → k_mult_logsd = |dK|/K_RATE). Read DIRECTLY;
                # do NOT re-multiply by abs_retention — that double-applied the 0.94 (train/serve mismatch, fixed
                # 2026-06-15 audit). (Earlier bug: prior code read nonexistent keys → channel silently 0; the fix
                # over-corrected by re-multiplying ret. Now the served SD equals the calibrated value.)
                k_logsd = float(ump.get("k_mult_logsd", 0.0))
                bb_logsd = float(ump.get("bb_mult_logsd", 0.0))
        _GF_EXTRA = (hr_logsd, k_logsd, bb_logsd) if (hr_logsd or k_logsd or bb_logsd) else False
    return _GF_EXTRA or None


def _build_gf_mult(n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """(n_sims, 2, 2, 9) f32 TWO-PART latent game-form multiplier, indexed
    [sim, offense_side, phase, outcome] with phase 0 = facing the STARTER, 1 = facing a reliever.
    Two independent, mean-neutral components, both using the estimated L / sigma_diag shape:
      • OFFENSE shock (GF_OFFENSE) — per BATTING side, applied to ALL its PAs (offense/
        environment day-form: ump zone, weather, the hitters' collective day; relievers see it too).
      • STARTER shock (GF_STARTER) — per PITCHING side, applied ONLY to PAs vs that side's
        starter (one arm × ~25 PAs = the dominant within-game pitching correlation that the single
        uniform shock under-fed → starter_K under-dispersed while totals over-dispersed). For
        batting side S the pitcher is side 1−S, so the starter draw used is st[1−S].
    Combined: phase0 = off[S]·st[1−S];  phase1 = off[S].  The −½·scale²·diag keeps E[mult]=1
    (marginals + means preserved; only variance added). GF_STARTER=0 ⇒ phases equal ⇒ kernel
    bit-identical to the prior single-shock (golden-master). GF_OFFENSE defaults to the legacy
    GF_SCALE. Env-tunable; absent factor file → ones."""
    # ALL four channels run at scale 1.0 on the data-measured day-form magnitudes — NO tuned constants.
    # The two former fudge factors (offense 0.4 / starter 1.5) are GONE; both root causes were fixed:
    #   • exact conversion: build_gf_split.realization_match row-scales the loadings so the kernel's
    #     renormalized multiplicative shock REALIZES the measured Σ rate-variance per outcome (replaces
    #     the lossy delta-method floor + top-2 truncation that under-fed the starter);
    #   • robust offense Σ: build_gf_robust re-estimates Σ_P/Σ_O from the two MARGINAL line-residual
    #     variances minus a directly-measured epistemic term (no fragile cross-cov; removes the offense
    #     epistemic double-count). Rebalanced K day-form Σ_P 2.89→1.83pp, Σ_O 2.22→2.59pp.
    # CLEAN-OOS validation (sim_vs_reality, 185 games 5/21-6/3 — strictly AFTER the 5/20 statcast/NUTS
    # training cutoff; draws=16, GF_*=1.0): total_runs 1.05, total_hits 1.01, total_HR 1.08, team_HR 1.02,
    # team_PA 1.02, all player props 0.98-1.04 — every channel in-band, no constants. K runs slightly under
    # (total_K 0.94, team_K 0.96, starter_K 0.83 ~2 SE): partly the modest robust Σ_P, partly an OOS K
    # mean under-bias (feed-side recal, NOT the kernel). GF_* stay env-overridable; default 1.0.
    s_off = float(os.environ.get("GF_OFFENSE", os.environ.get("GF_SCALE", "1.0")))
    s_st  = float(os.environ.get("GF_STARTER", "1.0"))
    # De-confounded two-shock loadings if available (L_O = offense day-form Σ_O, L_P = pitcher/starter
    # day-form Σ_P — orthogonal, no double-count); else fall back to the single confounded L for both.
    split = _load_game_form_split()
    if split is not None:
        L_O, sd_O, L_P, sd_P = split
    else:
        L1, sd1 = _load_game_form()
        L_O, sd_O, L_P, sd_P = L1, sd1, L1, sd1
    if L_O is None or (s_off <= 0.0 and s_st <= 0.0):
        return np.ones((n_sims, 2, 2, 9), dtype=np.float32)

    def _draw(scale, L, sdiag):
        if scale <= 0.0 or L is None:
            return np.ones((n_sims, 2, 9), dtype=np.float64)
        z = rng.standard_normal((n_sims, 2, L.shape[1]))  # per-side independent factor scores
        eps = (z @ L.T) * scale - 0.5 * (scale ** 2) * sdiag[None, None, :]
        return np.exp(eps)

    off = _draw(s_off, L_O, sd_O)                          # (n_sims, 2, 9) per batting side (offense day)
    st  = _draw(s_st,  L_P, sd_P)                          # (n_sims, 2, 9) per pitching side (starter day)
    out = np.empty((n_sims, 2, 2, 9), dtype=np.float32)
    for S in (0, 1):
        out[:, S, 0, :] = (off[:, S, :] * st[:, 1 - S, :]).astype(np.float32)   # vs starter
        out[:, S, 1, :] = off[:, S, :].astype(np.float32)                       # vs reliever

    # ── GAME-LEVEL day-form shocks shared by BOTH sides: carry→HR (Σ_carry), umpire→K/BB (Σ_umpire).
    # Carry = the park's carry that day (both teams hit there); the umpire calls the whole game (both
    # pitchers). Mean-neutral (−½σ²). Drawn from `rng` AFTER off/st (and after rng_seeds) ⇒ the kernel
    # xorshift outcome stream is untouched. GF_CARRY=0 & GF_UMPIRE=0 ⇒ no draw, no scaling ⇒ bit-identical.
    s_carry = float(os.environ.get("GF_CARRY", "1.0"))
    s_ump   = float(os.environ.get("GF_UMPIRE", "1.0"))
    # GAME_SHOCK_SHARE f∈[0,1]: fraction of each game-level carry/umpire shock that is COMMON to both teams.
    # f=1.0 ⇒ fully shared (legacy / golden-master). f<1.0 splits the shock into a shared part (σ√f) + a
    # per-team-INDEPENDENT part (σ√(1−f)) with σ_shared²+σ_indep²=σ² ⇒ each team's MARGINAL HR/K/BB variance
    # is UNCHANGED (no regression on total_HR/total_K calibration) but the cross-team run covariance scales
    # with f. The robust 428-game empirical cross-team Cov(home,away runs)≈0 (corr −0.07), while a fully-shared
    # carry+umpire produces ≈+0.47 → it OVER-couples the two teams (game-total slightly over-dispersed 1.08,
    # winning-margin under-dispersed 0.83). f matches the empirical cross-team cov, correcting BOTH at once.
    # Park-ψ epistemic stays fully shared (genuine same-park parameter uncertainty, only +0.09 of the cov).
    f_share = min(1.0, max(0.0, float(os.environ.get("GAME_SHOCK_SHARE", "1.0"))))
    sc_f, si_f = np.sqrt(f_share), np.sqrt(1.0 - f_share)
    ext = _load_gf_extra()
    if ext is not None:
        hr_sd, k_sd, bb_sd = ext
        # LIVE carry override: when LIVE_CARRY_LOGMULT is set (the in-game posterior-mean HR log-multiplier
        # from observed fly-ball carry — build_live_carry / live_carry_params.json), CENTER the carry shock on
        # it with the residual posterior spread (LIVE_CARRY_RESID_LOGSD) instead of the mean-neutral prior
        # draw. Used on resume from a live mid-game state. Unset ⇒ identical to the prior draw (golden-master).
        if s_carry > 0.0 and hr_sd > 0.0:
            zc = rng.standard_normal(n_sims)                                # shared component (GM consumption)
            lc_mu = os.environ.get("LIVE_CARRY_LOGMULT")
            if lc_mu is not None:                                           # live override → fully shared (one game)
                mu = float(lc_mu); rsd = float(os.environ.get("LIVE_CARRY_RESID_LOGSD", "0") or 0.0)
                shock = np.exp(mu + rsd * zc - 0.5 * rsd * rsd)            # E[shock]=exp(μ): rest-of-game HR tilt
                out[:, :, :, 8] *= shock.astype(np.float32)[:, None, None]   # HR
            else:
                sd = s_carry * hr_sd
                if f_share < 1.0:                                          # split shared + per-team-independent
                    zt = rng.standard_normal((n_sims, 2))                  # per BATTING side idiosyncratic carry
                    for S in (0, 1):
                        out[:, S, :, 8] *= np.exp(sd * (sc_f * zc + si_f * zt[:, S]) - 0.5 * sd * sd).astype(np.float32)[:, None]
                else:                                                      # fully shared (golden-master)
                    out[:, :, :, 8] *= np.exp(sd * zc - 0.5 * sd * sd).astype(np.float32)[:, None, None]
        if s_ump > 0.0 and (k_sd > 0.0 or bb_sd > 0.0):
            zu = rng.standard_normal(n_sims)                              # shared component (GM consumption)
            luk = os.environ.get("LIVE_UMP_K_LOGMULT")
            if luk is not None:
                # LIVE umpire override: CENTER the zone shock on the in-game day-form estimate (K log-mult muk
                # from observed called-strike tilt — build_live_umpire). BB moves opposite by the calibrated
                # bb:k leverage ratio (wide zone → K↑ BB↓). Residual spread = post-observation reduced σ.
                muk = float(luk); rk = float(os.environ.get("LIVE_UMP_RESID_K_LOGSD", "0") or 0.0)
                ratio = bb_sd / max(k_sd, 1e-9)
                out[:, :, :, 0] *= np.exp(muk + rk * zu - 0.5 * rk * rk).astype(np.float32)[:, None, None]
                out[:, :, :, 1] *= np.exp(-muk * ratio - rk * ratio * zu - 0.5 * (rk * ratio) ** 2).astype(np.float32)[:, None, None]
            else:
                ks, bs = s_ump * k_sd, s_ump * bb_sd
                if f_share < 1.0:                                          # K & BB share ONE zone shock per side
                    zt = rng.standard_normal((n_sims, 2))                  # per PITCHING/zone side idiosyncratic
                    for S in (0, 1):
                        zone = sc_f * zu + si_f * zt[:, S]
                        out[:, S, :, 0] *= np.exp(ks * zone - 0.5 * ks * ks).astype(np.float32)[:, None]   # K
                        out[:, S, :, 1] *= np.exp(-bs * zone - 0.5 * bs * bs).astype(np.float32)[:, None]  # BB
                else:                                                      # fully shared (golden-master)
                    out[:, :, :, 0] *= np.exp(ks * zu - 0.5 * ks * ks).astype(np.float32)[:, None, None]   # K
                    out[:, :, :, 1] *= np.exp(-bs * zu - 0.5 * bs * bs).astype(np.float32)[:, None, None]  # BB (wide zone → K↑ BB↓)
    return out


_IBB_PROP = None


def _load_ibb_prop() -> dict:
    """Per-batter IBB propensity multiplier {batter_id: corr} on the 1B-open BB lift
    (build_ibb_propensity.py). DEFAULT ON + artifact REQUIRED: when IBB_PROP != 0 the ibb_propensity.json
    artifact is HARD-REQUIRED (missing/malformed RAISES — no silent {}). Explicit IBB_PROP=0 = golden-master
    escape hatch → {} (every batter 1.0, the flat base_mult lift only). EB-shrunk so only magnets deviate."""
    global _IBB_PROP
    if _IBB_PROP is None:
        if os.environ.get("IBB_PROP", "1") != "0":
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "processed", "ibb_propensity.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"IBB_PROP is ON (default) but {path} is missing. Run build_ibb_propensity.py "
                    "(wired in finalize_fit) or set IBB_PROP=0 to disable.")
            _IBB_PROP = {int(k): float(v) for k, v in json.load(open(path))["corr"].items()}
        else:
            _IBB_PROP = {}
    return _IBB_PROP


def _perpitcher_ttop_renorm(curve: np.ndarray, avg_bf: float, scale: float = 2.5) -> np.ndarray:
    """DEPTH-AWARE (per-pitcher) mean-preserve. The league curve is mean-preserved over the LEAGUE bf
    distribution (~19 BF); applied to a DEEP starter (~24 BF) it over-weights the low-K late innings, so the
    outing-average falls BELOW the calibrated marginal a_hat (the small ace game-level K compression) — and
    the reverse over-boosts short arms. Re-normalize so the curve's outing-average over THIS pitcher's
    bf-SURVIVAL weighting w_p[bf]=P(still pitching at bf) (≈ logistic centered at his avg_bf) = 1 per outcome.
    Talent (a_hat) is untouched — only the WHEN of the decline is re-centered to the pitcher's own depth."""
    bf = np.arange(curve.shape[0], dtype=np.float64)
    w = 1.0 / (1.0 + np.exp((bf - float(avg_bf)) / scale))      # PA-weighting = bf survival
    s = w.sum()
    if s <= 0:
        return curve
    factor = (w[:, None] / s * curve).sum(0)                     # (9,) per-outcome outing-avg of the league curve
    factor = np.where(factor > 1e-9, factor, 1.0)
    return curve / factor[None, :]


def build_starter_tto_raw(pa_probs: np.ndarray,
                          curve: np.ndarray | None = None,
                          avg_bf: float | None = None) -> np.ndarray:
    """
    Build the starter's per-bf RAW matchup array (continuous TTOP).

    pa_probs: (n_batters, N_OUTCOMES) base matchup probs (log5 starter×batter).
    Returns:  (n_batters, N_BF_AXIS, N_OUTCOMES) RAW probs — index [slot, min(bf, 30)].

    The TTOP curve is mean-preserving over the starter bf distribution, so each
    pitcher's calibrated overall rate is unchanged; only the WHEN of the decline
    shifts (continuous in batters faced, no bf=9/18 discontinuity). Base-state is
    applied separately in the kernel, so these arrays are RAW (not cumulative).
    When avg_bf is given and TTOP_PERPITCHER is on, the curve is re-mean-preserved over THIS pitcher's
    bf-survival (depth-aware) so deep aces aren't over-suppressed / short arms over-boosted. Gate off ⇒ identity.
    """
    if curve is None:
        curve = _load_ttop_curve()
    if avg_bf is not None and os.environ.get("TTOP_PERPITCHER", "0") != "0":
        curve = _perpitcher_ttop_renorm(curve, avg_bf)
    n = pa_probs.shape[0]
    out = np.empty((n, N_BF_AXIS, N_OUTCOMES), dtype=np.float32)
    for b in range(N_BF_AXIS):
        adj = pa_probs * curve[b][None, :]              # (n, N_OUTCOMES)
        adj = adj / adj.sum(axis=1, keepdims=True)      # renormalize
        out[:, b, :] = adj
    return out


# ═══════════════════════════════════════════════════════════════
# NUMBA INNER LOOP
# ═══════════════════════════════════════════════════════════════

def _make_simulate_games_numba():
    """
    Factory that returns the numba-compiled simulate_games function.
    Called once at module load time after imports are confirmed.

    Returns None if numba is unavailable.

    The inner loop design:
      - Each simulation is fully independent → prange is correct
      - RNG: each sim gets its own seed (base_seed + sim_idx) for reproducibility
      - All state is local to the sim: no shared mutable state between sims
      - Lookup tables (new_state_table, runs_table) are read-only → safe with prange

    Performance notes:
      - int8 tables fit in L1 cache (24×9 = 216 bytes each)
      - float32 cumprobs: typical lineup is 9×9 = 324 bytes — also L1 cache
      - Inner loop is ~27 PA/inning × 18 half-innings ≈ 486 iterations per sim
      - At 50k sims: 24.3M iterations total, each ~10 ops → ~250M ops
      - Numba + LLVM vectorises this to ~50-100ms on M5 Pro
    """
    if not _NUMBA_AVAILABLE:
        return None

    # njit-compile the bullpen-manager picker so the kernel can call it.
    _pick_rel_nb = njit(cache=True)(_pick_reliever)
    _platoon_pick_nb = njit(cache=True)(_platoon_pick)
    _clf_choose_nb = njit(cache=True)(_clf_choose)

    @njit(parallel=True, cache=True)
    def simulate_games(
        home_cum_probs:    np.ndarray,   # (9, N_BF_AXIS, N_OUTCOMES) f32 RAW — [slot, min(bf,30)]
        away_cum_probs:    np.ndarray,   # (9, N_BF_AXIS, N_OUTCOMES) f32 RAW — [slot, min(bf,30)]
        home_bull_cum:     np.ndarray,   # (N_OUTCOMES,) float32 RAW — general bullpen blend
        away_bull_cum:     np.ndarray,   # (N_OUTCOMES,) float32 RAW
        emp_post_state:    np.ndarray,   # (24, 9, K) int8 — empirical resulting state (24=inning over)
        emp_runs_table:    np.ndarray,   # (24, 9, K) int8 — empirical runs scored on transition
        emp_cumprob:       np.ndarray,   # (24, 9, K) float32 — cumulative prob over K transitions
        home_exit_lut:     np.ndarray,   # (LUT_SIZE,) f64 — home starter removal-hazard table over live axes (exit_model.py)
        away_exit_lut:     np.ndarray,   # (LUT_SIZE,) f64 — away starter removal-hazard table over live axes
        home_ppo:          np.ndarray,   # (N_OUTCOMES,) float64 — home starter pitches/outcome
        away_ppo:          np.ndarray,   # (N_OUTCOMES,) float64 — away starter pitches/outcome
        pitch_resid_vals:  np.ndarray,   # (N_OUTCOMES, MAXP) float64 — zero-mean residual values
        pitch_resid_cum:   np.ndarray,   # (N_OUTCOMES, MAXP) float64 — cumulative prob
        stochastic_pitches: int,         # 1 = draw per-PA pitch count; 0 = deterministic mean
        home_rel_cum:      np.ndarray,   # (n_home_rel, 9, 9) f32 RAW — per-reliever per-slot vectors
        home_rel_role:     np.ndarray,   # (n_home_rel,) int8 — closer/setup/middle/long
        home_rel_stamina:  np.ndarray,   # (n_home_rel,) int8 — batters covered before a change
        home_closer_idx:   int,          # index of the closer in the home pen (−1 if none)
        away_rel_cum:      np.ndarray,   # (n_away_rel, 9, 9) f32 RAW
        away_rel_role:     np.ndarray,   # (n_away_rel,) int8
        away_rel_stamina:  np.ndarray,   # (n_away_rel,) int8
        away_closer_idx:   int,
        base_mult:         np.ndarray,   # (N_BUCKET, N_OUTCOMES) f32 — base-state log-odds tilt
        base_bucket:       np.ndarray,   # (8,) int8 — (state & 7) → base bucket
        rel_hazard:        np.ndarray,   # (RBF+1, RER+1, RINN, RSD+1) f32 — reliever pull hazard
        home_rel_throws:   np.ndarray,   # (n_home_rel,) int8 — L=0/R=1 (platoon-aware pick)
        away_rel_throws:   np.ndarray,   # (n_away_rel,) int8
        home_bat_stand:    np.ndarray,   # (9,) int8 — home batter stand per slot: L=0/R=1/S or unknown=-1
        away_bat_stand:    np.ndarray,   # (9,) int8 — away batter stand per slot
        home_bat_ibb:      np.ndarray,   # (9,) f32 — home batter IBB-propensity (× BB in 1B-open); 1.0=neutral
        away_bat_ibb:      np.ndarray,   # (9,) f32 — away batter IBB-propensity
        rng_seeds:         np.ndarray,   # (n_sims,) uint64 — per-sim xorshift seeds
        gf_mult:           np.ndarray,   # (n_sims, 2, 2, 9) f32 — [sim, off_side(0=home,1=away), phase(0=vs SP,1=vs pen), outcome]; ones=off
        home_rel_clf:      np.ndarray,   # (n_home_rel, 6) f64 — conditional-logit feature bundle
        away_rel_clf:      np.ndarray,   # (n_away_rel, 6) f64
        clf_w:             np.ndarray,   # (10,) f64 — conditional-logit interaction weights
        clf_on:            int,          # 1 = learned selection; 0 = deterministic (golden-master)
        track:             int,          # 1 = record per-batter [PA,H,HR,TB,RBI,R,K] + starter K
        home_bstats:       np.ndarray,   # (n_sims,9,7) int32 — written in-place when track==1 (else dummy)
        away_bstats:       np.ndarray,   # (n_sims,9,7) int32
        home_sk:           np.ndarray,   # (n_sims,) int32 — home starter strikeouts (while SP in)
        away_sk:           np.ndarray,   # (n_sims,) int32
        home_sbf:          np.ndarray,   # (n_sims,) int32 — home starter batters-faced (PA while SP in)
        away_sbf:          np.ndarray,   # (n_sims,) int32 — away starter batters-faced
        home_oc:           np.ndarray,   # (n_sims,9) int32 — home-batter realized outcome counts (diag)
        away_oc:           np.ndarray,   # (n_sims,9) int32 — away-batter realized outcome counts (diag)
        home_fed:          np.ndarray,   # (n_sims,9) f64 — Σ pre-tilt matchup prob cum[k] over home PAs (diag)
        away_fed:          np.ndarray,   # (n_sims,9) f64 — Σ pre-tilt matchup prob cum[k] over away PAs (diag)
        home_eff:          np.ndarray,   # (n_sims,9) f64 — Σ post-tilt normalized prob (cum*bm/tot) over home PAs
        away_eff:          np.ndarray,   # (n_sims,9) f64 — Σ post-tilt normalized prob (cum*bm/tot) over away PAs
        resume:            int,          # 0 = simulate from game start (golden-master path); 1 = resume from si64/sf64 state
        si64:              np.ndarray,   # (24,) int64 live-state scalars (see _build_start_arrays); ignored when resume==0
        sf64:              np.ndarray,   # (2,)  float64 live pitch counts [home_pitches, away_pitches]; ignored when resume==0
        steal_cfg:         np.ndarray,   # (11,) f32 steal cfg [on,succ,h_off,a_off,h_ds,a_ds,h_da,a_da,att_o0,o1,o2]; on=0 ⇒ no steal (golden master)
        emp_runner_adv:    np.ndarray,   # (24,9,K) f32 — runner-advancement level per transition (min-centered base_sum+4·runs)
        bat_advz:          np.ndarray,   # (2,9) f64 — per-slot runner speed-z [row0=home lineup, row1=away lineup]
        bat_steal:         np.ndarray,   # (2,9) f64 — per-slot steal-ATTEMPT mult (runner's own rate/league); replaces team st_*off
        runner_beta:       float,        # RUNNER-QUALITY XBT speed-tilt (log-odds/SD); 0.0 ⇒ no tilt ⇒ golden-master
        home_f5:           np.ndarray,   # (n_sims,) int32 — home runs at END OF INNING 5 (F5 snapshot), written in-place when track==1
        away_f5:           np.ndarray,   # (n_sims,) int32 — away runs at END OF INNING 5 (F5 snapshot), written in-place when track==1
        home_f8:           np.ndarray,   # (n_sims,) int32 — home runs at END OF INNING 8 (F8 snapshot), written in-place when track==1
        away_f8:           np.ndarray,   # (n_sims,) int32 — away runs at END OF INNING 8 (F8 snapshot), written in-place when track==1
        home_so_a:         np.ndarray,   # (n_sims,) int32 — home starter OUTS recorded (while SP in); in-place, track==1
        away_so_a:         np.ndarray,   # (n_sims,) int32 — away starter OUTS recorded (while SP in); in-place, track==1
        home_f1:           np.ndarray,   # (n_sims,) int32 — home runs at END OF INNING 1 (RFI); in-place, track==1
        away_f1:           np.ndarray,   # (n_sims,) int32 — away runs at END OF INNING 1 (RFI); in-place, track==1
        went_extra:        np.ndarray,   # (n_sims,) int32 — 1 if game reached extra innings (regulation tied), else 0; in-place, track==1
        n_innings:         int = 9,
        stop_on_tie:       int = 0,      # 1 = stop after n_innings even if tied (F5 semantics)
    ) -> np.ndarray:
        """
        Simulate n_sims complete games with per-PA dynamic starter removal.

        At each plate appearance while the starter is pitching, a Bernoulli draw
        is made against the pre-computed hazard table (P(removal | bf, er, inning,
        score_diff, is_start_of_half_inning)).  This is a truly dynamic model:
        struggling pitchers (high ER, poor score_diff) face escalating hazard
        probabilities, just as they would in a real game.

        Returns:
            scores: (n_sims, 2) int32 — [home_runs, away_runs]
        """
        n_sims = len(rng_seeds)
        K = emp_post_state.shape[2]
        # ── steal config (running-game layer); st_on=0 ⇒ block skipped (no RNG) ⇒ golden master ──
        st_on = steal_cfg[0]; st_succ = steal_cfg[1]
        st_hoff = steal_cfg[2]; st_aoff = steal_cfg[3]
        st_hds = steal_cfg[4]; st_ads = steal_cfg[5]
        st_hda = steal_cfg[6]; st_ada = steal_cfg[7]
        st_ao0 = steal_cfg[8]; st_ao1 = steal_cfg[9]; st_ao2 = steal_cfg[10]
        scores = np.zeros((n_sims, 2), dtype=np.int32)

        for sim in prange(n_sims):
            # Per-sim RNG: xorshift64 — state must be non-zero.
            rng_state = np.uint64(rng_seeds[sim]) | np.uint64(1)

            home_runs = 0
            away_runs = 0
            home_bf   = 0   # BFs completed by home starter (0 = none yet)
            away_bf   = 0
            home_pitches = 0.0  # cumulative pitches thrown by home starter (drives the hook)
            away_pitches = 0.0
            home_er   = 0   # earned runs charged to home starter
            away_er   = 0
            home_sp_br = 0  # baserunners (H+BB+HBP) allowed by home starter (GAM jam signal)
            away_sp_br = 0
            home_using_starter = True
            away_using_starter = True

            # ── Bullpen-manager state (individual sequenced relievers) ──
            n_home_rel = home_rel_cum.shape[0]
            n_away_rel = away_rel_cum.shape[0]
            # MLB 5.10(g) support: which arm (if any) is the synthetic POSITION PLAYER.
            # rel_clf col 10 is an identity flag set unconditionally by build_bullpen_profiles, so
            # this works on the deterministic path too. -1 when the feature is off (the default).
            home_pp_idx = -1
            for _i in range(n_home_rel):
                if home_rel_clf[_i, 10] > 0.5:
                    home_pp_idx = _i
            away_pp_idx = -1
            for _i in range(n_away_rel):
                if away_rel_clf[_i, 10] > 0.5:
                    away_pp_idx = _i
            home_cur_rel = -1      # index of the reliever currently pitching (−1 = none)
            away_cur_rel = -1
            home_cur_rel_bf = 0    # batters faced by the current reliever
            away_cur_rel_bf = 0
            home_cur_rel_er = 0    # runs allowed by the current reliever this outing
            away_cur_rel_er = 0
            home_rel_used = 0      # bitmask of relievers already used
            away_rel_used = 0

            # Batting-order position persists across innings for each team.
            away_batter_pos = 0
            home_batter_pos = 0

            # ── Per-batter tracking base identities (track==1 only) ──
            bp_away0 = -1; bp_away1 = -1; bp_away2 = -1   # base identities (lineup slot on 1B/2B/3B; -1 empty)
            bp_home0 = -1; bp_home1 = -1; bp_home2 = -1
            tmp = np.empty(4, np.int64)                    # scratch for lead-first runner order

            # ── RESUME: override the game-start init with the LIVE game state ──
            # resume==0 leaves every var at its game-start value and first_iter=0, so all
            # resume branches below are dead and the RNG stream / control flow are IDENTICAL
            # to the pre-resume kernel (golden-master). resume==1 seeds the live state; the
            # first inning iteration then honors start_half (skip the already-played top half)
            # and the partial base-out state r_state.
            first_iter = 0
            start_half = 0          # 0 = resume into TOP (away batting); 1 = into BOTTOM (home batting)
            r_state    = 0          # base-out state to seed the resumed half-inning
            r_is_eoi   = 1          # is the resumed PA the first of its half-inning?
            if resume == 1:
                first_iter = 1
                start_half = int(si64[1]); r_state = int(si64[2]); r_is_eoi = int(si64[23])
                home_runs  = int(si64[3]);  away_runs = int(si64[4])
                home_bf    = int(si64[5]);  away_bf   = int(si64[6])
                home_er    = int(si64[7]);  away_er   = int(si64[8])
                home_sp_br = int(si64[9]);  away_sp_br = int(si64[10])
                home_using_starter = (si64[11] == 1); away_using_starter = (si64[12] == 1)
                home_batter_pos = int(si64[13]); away_batter_pos = int(si64[14])
                home_cur_rel = int(si64[15]); away_cur_rel = int(si64[16])
                home_cur_rel_bf = int(si64[17]); away_cur_rel_bf = int(si64[18])
                home_cur_rel_er = int(si64[19]); away_cur_rel_er = int(si64[20])
                home_rel_used = int(si64[21]); away_rel_used = int(si64[22])
                home_pitches = sf64[0]; away_pitches = sf64[1]

            MAX_INNINGS = 20
            inning = int(si64[0]) if resume == 1 else 0
            while inning < MAX_INNINGS:

                # ── Away half-inning (home pitcher pitching) ──────────────
                # On a BOTTOM-half resume the top half is already complete → skip it.
                do_away = 0 if (first_iter == 1 and start_half == 1) else 1
                if first_iter == 1 and start_half == 0:
                    state = r_state          # seed the partial top half from the live state
                    if track == 1:
                        bp_away0 = -1; bp_away1 = -1; bp_away2 = -1
                    is_eoi = r_is_eoi
                else:
                    # Ghost runner on 2B in extras (Manfred rule, regular season 2020+).
                    state       = 2 if (stop_on_tie == 0 and inning >= n_innings) else 0
                    if track == 1:
                        bp_away0 = -1; bp_away2 = -1
                        bp_away1 = ((away_batter_pos - 1) % 9) if (stop_on_tie == 0 and inning >= n_innings) else -1
                    is_eoi      = 1   # first PA of half-inning → between-inning decision
                inning_runs = 0

                while do_away == 1:
                    # ── Dynamic hazard removal check ───────────────────────
                    if home_using_starter:
                        # Starter removal hazard: O(1) lookup in the per-start LUT baked from the
                        # LightGBM discrete-time hazard (exit_model.py), over the live state axes
                        # pitch_count × earned_runs × accrued_baserunners × base-out × end-of-inning.
                        _pc = int(home_pitches)
                        if _pc > 130:
                            _pc = 130
                        elif _pc < 0:
                            _pc = 0
                        _er = home_er if home_er < 8 else 8
                        _br = home_sp_br if home_sp_br < 6 else 6
                        if home_exit_lut.shape[0] == 4138291:  # v2 EXACT layout (exit_model, 2026-07-21):
                            # br lives IN the dense grid (additivity leaked −11..21% hazard in the
                            # decision zone → the fat BF right tail), bo_adj is pc-conditioned, and
                            # the trailing cell is h_shock — the competing-risk injury hazard mixed
                            # in PROBABILITY space (an injury can end any PA regardless of state).
                            _sd = home_runs - away_runs      # pitcher's team lead (positive = winning)
                            if _sd < -4:
                                _sd = -4
                            elif _sd > 4:
                                _sd = 4
                            _sd += 4
                            _brv = home_sp_br if home_sp_br < 14 else 14   # v2 br cap = training p99.9, not the legacy 6
                            if is_eoi == 1:   # inning start: bases empty or the extras ghost (state==2)
                                _L = home_exit_lut[3819960 + (((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 2 + (1 if state == 2 else 0)]
                            else:             # mid-inning: full (pc,er,sd,br,base_out) dense cell — zero approximation
                                _L = home_exit_lut[(((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 24 + state]
                            h = 1.0 / (1.0 + np.exp(-_L))
                            _hs = home_exit_lut[4138290]
                            h = _hs + (1.0 - _hs) * h
                        elif home_exit_lut.shape[0] > 2389:  # live score_diff axis (exit_model SD layout: pc×er×sd×eoi)
                            _sd = home_runs - away_runs      # pitcher's team lead (positive = winning)
                            if _sd < -4:
                                _sd = -4
                            elif _sd > 4:
                                _sd = 4
                            _sd += 4
                            h = 1.0 / (1.0 + np.exp(-(home_exit_lut[((_pc * 9 + _er) * 9 + _sd) * 2 + is_eoi]
                                                      + home_exit_lut[21222 + state] + home_exit_lut[21246 + _br])))
                        else:
                            h = 1.0 / (1.0 + np.exp(-(home_exit_lut[(_pc * 9 + _er) * 2 + is_eoi]
                                                      + home_exit_lut[2358 + state] + home_exit_lut[2382 + _br])))
                        # Bernoulli draw via xorshift64
                        rng_state ^= rng_state << np.uint64(13)
                        rng_state ^= rng_state >> np.uint64(7)
                        rng_state ^= rng_state << np.uint64(17)
                        u_rem = (np.float64(rng_state >> np.uint64(11))
                                 * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                        if u_rem < h:
                            home_using_starter = False
                    is_eoi = 0   # only True for the very first PA of each half-inning

                    if home_using_starter:
                        cum = away_cum_probs[away_batter_pos % 9,
                                             min(away_batter_pos, away_cum_probs.shape[1] - 1)]  # continuous bf
                    elif n_home_rel > 0:
                        # ── Individual-reliever manager (HOME pen defending) ──
                        home_lead = home_runs - away_runs
                        save_sit = 1 if (inning >= SAVE_INNING_IDX
                                         and 1 <= home_lead <= SAVE_LEAD_MAX) else 0
                        blowout  = 1 if (home_lead > BLOWOUT_MARGIN
                                         or home_lead < -BLOWOUT_MARGIN) else 0
                        # Leverage Index proxy: a high-LI jam = late, close, runners
                        # on (≥2) with ≤1 out → summon the best "fireman" arm.
                        nrun = (state & 1) + ((state >> 1) & 1) + ((state >> 2) & 1)
                        hi_lev = 1 if (inning >= HIGH_LEV_INNING_IDX
                                       and -HIGH_LEV_MARGIN <= home_lead <= HIGH_LEV_MARGIN
                                       and nrun >= 2 and (state // 8) <= 1) else 0
                        need = 0
                        if home_cur_rel < 0:
                            need = 1
                        else:
                            # Stochastic removal hazard (replaces fixed stamina): the
                            # 3-batter-min floor is baked into the table (haz≈0 for bf<3).
                            rbf = home_cur_rel_bf if home_cur_rel_bf < REL_BF_MAX else REL_BF_MAX
                            rer = home_cur_rel_er if home_cur_rel_er < REL_ER_MAX else REL_ER_MAX
                            rinn = inning if inning < REL_INN_MAX else REL_INN_MAX - 1
                            rsd = home_lead + REL_SD_OFF
                            if rsd < 0:
                                rsd = 0
                            elif rsd > REL_SD_MAX:
                                rsd = REL_SD_MAX
                            haz = rel_hazard[rbf, rer, rinn, rsd]
                            rng_state ^= rng_state << np.uint64(13)
                            rng_state ^= rng_state >> np.uint64(7)
                            rng_state ^= rng_state << np.uint64(17)
                            u_h = (np.float64(rng_state >> np.uint64(11))
                                   * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                            if u_h < haz:
                                need = 1
                        # Closer enters fresh for a 9th-inning save (between innings).
                        if (is_eoi == 1 and save_sit == 1 and home_closer_idx >= 0
                                and ((home_rel_used >> home_closer_idx) & 1) == 0):
                            need = 1
                        # Fireman: a developing high-LI jam (3-batter min met, current
                        # arm not already a top-2 quality arm) → make a change.
                        if hi_lev == 1 and home_cur_rel_bf >= 3 and home_cur_rel > 1:
                            need = 1
                        if need == 1:
                            # majority hand of the next ≤3 AWAY batters (home pen faces away lineup)
                            nL = 0; nR = 0
                            for jj in range(3):
                                bs = away_bat_stand[(away_batter_pos + jj) % 9]
                                if bs == 0:
                                    nL += 1
                                elif bs == 1:
                                    nR += 1
                            want = -1
                            if nL > nR:
                                want = 0
                            elif nR > nL:
                                want = 1
                            # MLB 5.10(g): block the position-player arm unless the game is decided
                            # (trailing 8+, leading 10+, or extra innings). inning is 0-based.
                            _h_ppb = (home_pp_idx if not (inning >= 9 or home_lead <= -PP_LOSING_BY or home_lead >= PP_WINNING_BY) else -1)
                            if clf_on == 1:
                                # LEARNED conditional-logit pick (mirrors real managers).
                                late9 = 1 if inning >= SAVE_INNING_IDX else 0
                                alead = home_lead if home_lead >= 0 else -home_lead
                                close_late = 1 if (inning >= HIGH_LEV_INNING_IDX and alead <= 3) else 0
                                # SIGNED lead: protect vs conserve (replaces symmetric abslead)
                                protect_h = float(home_lead) if home_lead > 0 else 0.0
                                deficit_h = float(-home_lead) if home_lead < 0 else 0.0
                                # Leverage Index — HOME pen pitches the TOP half ⇒ is_bot=0
                                li_i = inning if inning < LI_N_INN else LI_N_INN - 1
                                li_d = home_lead
                                if li_d < -LI_SD_MAX:
                                    li_d = -LI_SD_MAX
                                elif li_d > LI_SD_MAX:
                                    li_d = LI_SD_MAX
                                li_s = state if state < 24 else 23
                                li_h = LI_TABLE[li_i, 0, li_s, li_d + LI_SD_OFF]
                                jam_h = 1.0 if (nrun >= 2 and (state // 8) <= 1) else 0.0
                                rng_state ^= rng_state << np.uint64(13)
                                rng_state ^= rng_state >> np.uint64(7)
                                rng_state ^= rng_state << np.uint64(17)
                                udraw = (np.float64(rng_state >> np.uint64(11))
                                         * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                                nxt = _clf_choose_nb(home_rel_used, n_home_rel, home_rel_clf,
                                                     home_rel_throws, clf_w, inning + 1, save_sit,
                                                     blowout, late9, close_late,
                                                     li_h, protect_h, deficit_h, jam_h, want, udraw, _h_ppb)
                            else:
                                # deterministic role/leverage pick + platoon override (golden-master path)
                                nxt = _pick_rel_nb(home_rel_used, n_home_rel, home_rel_role,
                                                   home_closer_idx, inning, save_sit, blowout, hi_lev,
                                                   home_lead, _h_ppb)
                                nxt = _platoon_pick_nb(nxt, home_rel_used, n_home_rel, home_rel_role,
                                                    home_closer_idx, inning, save_sit, blowout,
                                                    hi_lev, home_rel_throws, want, _h_ppb)
                            if nxt < 0:
                                # pen exhausted → recycle the WORST already-used arm
                                # (arms sorted best→worst; highest used index = worst).
                                for ri in range(n_home_rel - 1, -1, -1):
                                    if (home_rel_used >> ri) & 1 == 1:
                                        nxt = ri
                                        break
                            if nxt >= 0:
                                home_cur_rel = nxt
                                home_cur_rel_bf = 0
                                home_cur_rel_er = 0
                                home_rel_used |= (1 << nxt)
                        cum = home_rel_cum[home_cur_rel, away_batter_pos % 9]
                    else:
                        cum = home_bull_cum     # empty pen (no relievers) → league blend

                    # ── STEAL of 2B (running-game layer): away runner on 1B, 2B open, home battery.
                    # st_on=0 ⇒ skipped entirely (no RNG drawn) ⇒ byte-identical golden master.
                    # state+1 = 1B→2B (safe: 2B empty, no carry); state+7 = remove 1B + add 1 out.
                    if st_on > 0.5 and (state & 1) == 1 and (state & 2) == 0:
                        co = state >> 3
                        st_ao = bat_steal[1, bp_away0] if (runner_beta != 0.0 and bp_away0 >= 0) else st_aoff
                        st_att = (st_ao0 if co == 0 else (st_ao1 if co == 1 else st_ao2)) * st_ao * st_hda
                        rng_state ^= rng_state << np.uint64(13)
                        rng_state ^= rng_state >> np.uint64(7)
                        rng_state ^= rng_state << np.uint64(17)
                        u_sb = (np.float64(rng_state >> np.uint64(11)) * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                        if u_sb < st_att:
                            rng_state ^= rng_state << np.uint64(13)
                            rng_state ^= rng_state >> np.uint64(7)
                            rng_state ^= rng_state << np.uint64(17)
                            u_sc = (np.float64(rng_state >> np.uint64(11)) * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                            if u_sc < st_succ * st_hds:
                                state += 1                                    # SB: 1B→2B
                                if track == 1:
                                    bp_away1 = bp_away0; bp_away0 = -1
                            elif co + 1 >= 3:                                 # CS = 3rd out → inning over
                                away_runs += inning_runs
                                away_batter_pos += 1
                                break
                            else:
                                state += 7                                    # CS: remove 1B + 1 out
                                if track == 1:
                                    bp_away0 = -1

                    # xorshift64 → PA outcome
                    rng_state ^= rng_state << np.uint64(13)
                    rng_state ^= rng_state >> np.uint64(7)
                    rng_state ^= rng_state << np.uint64(17)
                    u = (np.float64(rng_state >> np.uint64(11))
                         * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))

                    # base-state tilt: p_i = raw_i * base_mult[bucket,i], renormalize,
                    # then draw with the SAME u (single RNG draw → stream preserved).
                    # 1B-open bucket (2): BB further scaled by the AWAY batter's IBB propensity.
                    bk = base_bucket[state & 7]
                    ibb = away_bat_ibb[away_batter_pos % 9] if bk == 2 else np.float32(1.0)
                    gfm = gf_mult[sim, 1, 0] if home_using_starter else gf_mult[sim, 1, 1]   # away off vs home SP(0)/pen(1)
                    tot = 0.0
                    for k in range(9):
                        bm = base_mult[bk, k] * (ibb if k == 1 else np.float32(1.0)) * gfm[k]
                        tot += cum[k] * bm
                    target = u * tot
                    acc = 0.0
                    outcome = 8
                    for k in range(9):
                        bm = base_mult[bk, k] * (ibb if k == 1 else np.float32(1.0)) * gfm[k]
                        acc += cum[k] * bm
                        if target < acc:
                            outcome = k
                            break

                    # Empirical base-running transition: sample (post_state, runs)
                    # from the observed categorical for this (state, outcome).
                    rng_state ^= rng_state << np.uint64(13)
                    rng_state ^= rng_state >> np.uint64(7)
                    rng_state ^= rng_state << np.uint64(17)
                    u2 = (np.float64(rng_state >> np.uint64(11))
                          * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                    if runner_beta != 0.0 and track == 1 and (state & 7) != 0:
                        # RUNNER-QUALITY XBT tilt: boost extra-advancement transitions for fast on-base
                        # runners (avg speed-z of the runners actually on base). exp() tilt is mean-preserving
                        # over the categorical (re-normalized). beta=0 ⇒ else-branch ⇒ byte-identical.
                        zr = 0.0; nr = 0
                        if bp_away0 >= 0:
                            zr += bat_advz[1, bp_away0]; nr += 1
                        if bp_away1 >= 0:
                            zr += bat_advz[1, bp_away1]; nr += 1
                        if bp_away2 >= 0:
                            zr += bat_advz[1, bp_away2]; nr += 1
                        bz = runner_beta * (zr / nr) if nr > 0 else 0.0
                        tot = 0.0; prev = 0.0
                        for kk in range(K):
                            pk = emp_cumprob[state, outcome, kk] - prev
                            prev = emp_cumprob[state, outcome, kk]
                            w = 1.0 + bz * emp_runner_adv[state, outcome, kk]
                            if w < 0.0:
                                w = 0.0
                            tot += pk * w
                        targ = u2 * tot; acc = 0.0; prev = 0.0; sel = 0
                        for kk in range(K):
                            sel = kk
                            pk = emp_cumprob[state, outcome, kk] - prev
                            prev = emp_cumprob[state, outcome, kk]
                            w = 1.0 + bz * emp_runner_adv[state, outcome, kk]
                            if w < 0.0:
                                w = 0.0
                            acc += pk * w
                            if targ < acc:
                                break
                    else:
                        sel = 0
                        for kk in range(K):
                            sel = kk
                            if u2 < emp_cumprob[state, outcome, kk]:
                                break
                    new_s = emp_post_state[state, outcome, sel]
                    r     = emp_runs_table[state, outcome, sel]

                    inning_runs += r
                    if track == 1:
                        _s = away_batter_pos % 9
                        away_oc[sim, outcome] += 1                   # realized outcome dist (diag)
                        # fed-vs-effective (window/sampling-free tilt distortion): accumulate the
                        # pre-tilt matchup prob cum[k] (fed) and the post-tilt normalized prob the
                        # draw samples from, cum[k]*bm/tot (effective). Both over the SAME PA stream.
                        for _k in range(9):
                            _bm = base_mult[bk, _k] * (ibb if _k == 1 else np.float32(1.0)) * gfm[_k]
                            away_fed[sim, _k] += cum[_k]
                            away_eff[sim, _k] += cum[_k] * _bm / tot
                        away_bstats[sim, _s, 0] += 1                  # PA
                        if outcome >= 5:                             # 5=1B 6=2B 7=3B 8=HR
                            away_bstats[sim, _s, 1] += 1             # H
                            away_bstats[sim, _s, 3] += outcome - 4   # TB (1B=1..HR=4)
                            if outcome == 8:
                                away_bstats[sim, _s, 2] += 1         # HR
                        away_bstats[sim, _s, 4] += r                 # RBI = runs on this PA
                        if outcome == 0:
                            away_bstats[sim, _s, 6] += 1             # K
                        if home_using_starter and outcome == 0:
                            home_sk[sim] += 1                        # home starter K
                        if home_using_starter:
                            # starter OUTS recorded this PA: outs = state>>3; on an
                            # inning-ending out (new_s>=24) it's 3-old_outs (correctly
                            # 2 for a GIDP), else new_outs-old_outs. Reads state only — no RNG.
                            _oo0 = state >> 3
                            home_so_a[sim] += (3 - _oo0) if new_s >= 24 else ((new_s >> 3) - _oo0)
                        # --- inline _credit_runs_and_advance (R attribution + base identity) ---
                        nn = 0
                        if bp_away2 >= 0: tmp[nn] = bp_away2; nn += 1   # 3rd (lead)
                        if bp_away1 >= 0: tmp[nn] = bp_away1; nn += 1   # 2nd
                        if bp_away0 >= 0: tmp[nn] = bp_away0; nn += 1   # 1st
                        tmp[nn] = _s; nn += 1                          # batter scores last
                        for _i in range(r):
                            if _i < nn:
                                away_bstats[sim, tmp[_i], 5] += 1      # R (run scored)
                        bp_away0 = -1; bp_away1 = -1; bp_away2 = -1
                        if new_s < 24:
                            _j = r
                            if ((new_s >> 2) & 1) == 1 and _j < nn: bp_away2 = tmp[_j]; _j += 1
                            if ((new_s >> 1) & 1) == 1 and _j < nn: bp_away1 = tmp[_j]; _j += 1
                            if (new_s & 1) == 1 and _j < nn:        bp_away0 = tmp[_j]; _j += 1
                    if home_using_starter:
                        home_er += r
                        home_bf += 1
                        if outcome == 1 or outcome == 2 or outcome >= 5:   # BB/HBP/1B/2B/3B/HR → reached base
                            home_sp_br += 1
                        if stochastic_pitches == 1:
                            rng_state ^= rng_state << np.uint64(13)
                            rng_state ^= rng_state >> np.uint64(7)
                            rng_state ^= rng_state << np.uint64(17)
                            u_pc = (np.float64(rng_state >> np.uint64(11))
                                    * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                            psel = 0
                            for pk in range(pitch_resid_cum.shape[1]):
                                psel = pk
                                if u_pc < pitch_resid_cum[outcome, pk]:
                                    break
                            p_add = home_ppo[outcome] + pitch_resid_vals[outcome, psel]
                            if p_add < 1.0:
                                p_add = 1.0
                            home_pitches += p_add
                        else:
                            home_pitches += home_ppo[outcome]
                    else:
                        if home_cur_rel >= 0:
                            home_cur_rel_bf += 1
                            home_cur_rel_er += r

                    if new_s >= 24:   # 3 outs — half-inning over
                        away_runs += inning_runs
                        away_batter_pos += 1   # out-maker's slot is done; next inning starts with the next batter
                        break
                    else:
                        state = new_s
                        away_batter_pos += 1

                # ── Home half-inning (away pitcher pitching) ──────────────
                # Skip the bottom of the final inning only in FULL-GAME mode. In F5
                # mode (stop_on_tie) the 5th is not the game's end — the home team
                # always bats the full bottom of the 5th, so never skip it.
                if do_away == 1 and stop_on_tie == 0 and inning == n_innings - 1 and home_runs > away_runs:
                    inning += 1
                    break  # walk-off — home wins without batting

                if first_iter == 1 and start_half == 1:
                    state = r_state          # seed the partial bottom half from the live state
                    if track == 1:
                        bp_home0 = -1; bp_home1 = -1; bp_home2 = -1
                    is_eoi = r_is_eoi
                else:
                    # Ghost runner on 2B in extras (Manfred rule, regular season 2020+).
                    state       = 2 if (stop_on_tie == 0 and inning >= n_innings) else 0
                    if track == 1:
                        bp_home0 = -1; bp_home2 = -1
                        bp_home1 = ((home_batter_pos - 1) % 9) if (stop_on_tie == 0 and inning >= n_innings) else -1
                    is_eoi      = 1   # reset for new half-inning
                inning_runs = 0

                while True:
                    # ── Dynamic hazard removal check ───────────────────────
                    if away_using_starter:
                        # Starter removal hazard: O(1) lookup in the per-start LUT (exit_model.py).
                        _pc = int(away_pitches)
                        if _pc > 130:
                            _pc = 130
                        elif _pc < 0:
                            _pc = 0
                        _er = away_er if away_er < 8 else 8
                        _br = away_sp_br if away_sp_br < 6 else 6
                        if away_exit_lut.shape[0] == 4138291:  # v2 EXACT layout (see home-side comment)
                            _sd = away_runs - home_runs      # pitcher's team lead (positive = winning)
                            if _sd < -4:
                                _sd = -4
                            elif _sd > 4:
                                _sd = 4
                            _sd += 4
                            _brv = away_sp_br if away_sp_br < 14 else 14   # v2 br cap = training p99.9, not the legacy 6
                            if is_eoi == 1:   # inning start: bases empty or the extras ghost (state==2)
                                _L = away_exit_lut[3819960 + (((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 2 + (1 if state == 2 else 0)]
                            else:             # mid-inning: full (pc,er,sd,br,base_out) dense cell — zero approximation
                                _L = away_exit_lut[(((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 24 + state]
                            h = 1.0 / (1.0 + np.exp(-_L))
                            _hs = away_exit_lut[4138290]
                            h = _hs + (1.0 - _hs) * h
                        elif away_exit_lut.shape[0] > 2389:  # live score_diff axis (exit_model SD layout: pc×er×sd×eoi)
                            _sd = away_runs - home_runs      # pitcher's team lead (positive = winning)
                            if _sd < -4:
                                _sd = -4
                            elif _sd > 4:
                                _sd = 4
                            _sd += 4
                            h = 1.0 / (1.0 + np.exp(-(away_exit_lut[((_pc * 9 + _er) * 9 + _sd) * 2 + is_eoi]
                                                      + away_exit_lut[21222 + state] + away_exit_lut[21246 + _br])))
                        else:
                            h = 1.0 / (1.0 + np.exp(-(away_exit_lut[(_pc * 9 + _er) * 2 + is_eoi]
                                                      + away_exit_lut[2358 + state] + away_exit_lut[2382 + _br])))
                        rng_state ^= rng_state << np.uint64(13)
                        rng_state ^= rng_state >> np.uint64(7)
                        rng_state ^= rng_state << np.uint64(17)
                        u_rem = (np.float64(rng_state >> np.uint64(11))
                                 * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                        if u_rem < h:
                            away_using_starter = False
                    is_eoi = 0

                    if away_using_starter:
                        cum = home_cum_probs[home_batter_pos % 9,
                                             min(home_batter_pos, home_cum_probs.shape[1] - 1)]  # continuous bf
                    elif n_away_rel > 0:
                        # ── Individual-reliever manager (AWAY pen defending) ──
                        away_lead = away_runs - home_runs
                        save_sit = 1 if (inning >= SAVE_INNING_IDX
                                         and 1 <= away_lead <= SAVE_LEAD_MAX) else 0
                        blowout  = 1 if (away_lead > BLOWOUT_MARGIN
                                         or away_lead < -BLOWOUT_MARGIN) else 0
                        nrun = (state & 1) + ((state >> 1) & 1) + ((state >> 2) & 1)
                        hi_lev = 1 if (inning >= HIGH_LEV_INNING_IDX
                                       and -HIGH_LEV_MARGIN <= away_lead <= HIGH_LEV_MARGIN
                                       and nrun >= 2 and (state // 8) <= 1) else 0
                        need = 0
                        if away_cur_rel < 0:
                            need = 1
                        else:
                            rbf = away_cur_rel_bf if away_cur_rel_bf < REL_BF_MAX else REL_BF_MAX
                            rer = away_cur_rel_er if away_cur_rel_er < REL_ER_MAX else REL_ER_MAX
                            rinn = inning if inning < REL_INN_MAX else REL_INN_MAX - 1
                            rsd = away_lead + REL_SD_OFF
                            if rsd < 0:
                                rsd = 0
                            elif rsd > REL_SD_MAX:
                                rsd = REL_SD_MAX
                            haz = rel_hazard[rbf, rer, rinn, rsd]
                            rng_state ^= rng_state << np.uint64(13)
                            rng_state ^= rng_state >> np.uint64(7)
                            rng_state ^= rng_state << np.uint64(17)
                            u_h = (np.float64(rng_state >> np.uint64(11))
                                   * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                            if u_h < haz:
                                need = 1
                        if (is_eoi == 1 and save_sit == 1 and away_closer_idx >= 0
                                and ((away_rel_used >> away_closer_idx) & 1) == 0):
                            need = 1
                        if hi_lev == 1 and away_cur_rel_bf >= 3 and away_cur_rel > 1:
                            need = 1
                        if need == 1:
                            # majority hand of the next ≤3 HOME batters (away pen faces home lineup)
                            nL = 0; nR = 0
                            for jj in range(3):
                                bs = home_bat_stand[(home_batter_pos + jj) % 9]
                                if bs == 0:
                                    nL += 1
                                elif bs == 1:
                                    nR += 1
                            want = -1
                            if nL > nR:
                                want = 0
                            elif nR > nL:
                                want = 1
                            # MLB 5.10(g): block the position-player arm unless the game is decided
                            # (trailing 8+, leading 10+, or extra innings). inning is 0-based.
                            _a_ppb = (away_pp_idx if not (inning >= 9 or away_lead <= -PP_LOSING_BY or away_lead >= PP_WINNING_BY) else -1)
                            if clf_on == 1:
                                # LEARNED conditional-logit pick (mirrors real managers).
                                late9 = 1 if inning >= SAVE_INNING_IDX else 0
                                alead = away_lead if away_lead >= 0 else -away_lead
                                close_late = 1 if (inning >= HIGH_LEV_INNING_IDX and alead <= 3) else 0
                                # SIGNED lead: protect vs conserve (replaces symmetric abslead)
                                protect_a = float(away_lead) if away_lead > 0 else 0.0
                                deficit_a = float(-away_lead) if away_lead < 0 else 0.0
                                # Leverage Index — AWAY pen pitches the BOTTOM half ⇒ is_bot=1
                                li_i = inning if inning < LI_N_INN else LI_N_INN - 1
                                li_d = away_lead
                                if li_d < -LI_SD_MAX:
                                    li_d = -LI_SD_MAX
                                elif li_d > LI_SD_MAX:
                                    li_d = LI_SD_MAX
                                li_s = state if state < 24 else 23
                                li_a = LI_TABLE[li_i, 1, li_s, li_d + LI_SD_OFF]
                                jam_a = 1.0 if (nrun >= 2 and (state // 8) <= 1) else 0.0
                                rng_state ^= rng_state << np.uint64(13)
                                rng_state ^= rng_state >> np.uint64(7)
                                rng_state ^= rng_state << np.uint64(17)
                                udraw = (np.float64(rng_state >> np.uint64(11))
                                         * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                                nxt = _clf_choose_nb(away_rel_used, n_away_rel, away_rel_clf,
                                                     away_rel_throws, clf_w, inning + 1, save_sit,
                                                     blowout, late9, close_late,
                                                     li_a, protect_a, deficit_a, jam_a, want, udraw, _a_ppb)
                            else:
                                nxt = _pick_rel_nb(away_rel_used, n_away_rel, away_rel_role,
                                                   away_closer_idx, inning, save_sit, blowout, hi_lev,
                                                   away_lead, _a_ppb)
                                nxt = _platoon_pick_nb(nxt, away_rel_used, n_away_rel, away_rel_role,
                                                    away_closer_idx, inning, save_sit, blowout,
                                                    hi_lev, away_rel_throws, want, _a_ppb)
                            if nxt < 0:
                                for ri in range(n_away_rel - 1, -1, -1):
                                    if (away_rel_used >> ri) & 1 == 1:
                                        nxt = ri
                                        break
                            if nxt >= 0:
                                away_cur_rel = nxt
                                away_cur_rel_bf = 0
                                away_cur_rel_er = 0
                                away_rel_used |= (1 << nxt)
                        cum = away_rel_cum[away_cur_rel, home_batter_pos % 9]
                    else:
                        cum = away_bull_cum     # empty pen (no relievers) → league blend

                    # ── STEAL of 2B: home runner on 1B, 2B open, away battery (mirror of away block) ──
                    if st_on > 0.5 and (state & 1) == 1 and (state & 2) == 0:
                        co = state >> 3
                        st_ho = bat_steal[0, bp_home0] if (runner_beta != 0.0 and bp_home0 >= 0) else st_hoff
                        st_att = (st_ao0 if co == 0 else (st_ao1 if co == 1 else st_ao2)) * st_ho * st_ada
                        rng_state ^= rng_state << np.uint64(13)
                        rng_state ^= rng_state >> np.uint64(7)
                        rng_state ^= rng_state << np.uint64(17)
                        u_sb = (np.float64(rng_state >> np.uint64(11)) * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                        if u_sb < st_att:
                            rng_state ^= rng_state << np.uint64(13)
                            rng_state ^= rng_state >> np.uint64(7)
                            rng_state ^= rng_state << np.uint64(17)
                            u_sc = (np.float64(rng_state >> np.uint64(11)) * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                            if u_sc < st_succ * st_ads:
                                state += 1                                    # SB: 1B→2B
                                if track == 1:
                                    bp_home1 = bp_home0; bp_home0 = -1
                            elif co + 1 >= 3:                                 # CS = 3rd out → inning over
                                home_runs += inning_runs
                                home_batter_pos += 1
                                break
                            else:
                                state += 7                                    # CS: remove 1B + 1 out
                                if track == 1:
                                    bp_home0 = -1

                    rng_state ^= rng_state << np.uint64(13)
                    rng_state ^= rng_state >> np.uint64(7)
                    rng_state ^= rng_state << np.uint64(17)
                    u = (np.float64(rng_state >> np.uint64(11))
                         * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))

                    # base-state tilt: p_i = raw_i * base_mult[bucket,i], renormalize,
                    # then draw with the SAME u (single RNG draw → stream preserved).
                    # 1B-open bucket (2): BB further scaled by the HOME batter's IBB propensity.
                    bk = base_bucket[state & 7]
                    ibb = home_bat_ibb[home_batter_pos % 9] if bk == 2 else np.float32(1.0)
                    gfm = gf_mult[sim, 0, 0] if away_using_starter else gf_mult[sim, 0, 1]   # home off vs away SP(0)/pen(1)
                    tot = 0.0
                    for k in range(9):
                        bm = base_mult[bk, k] * (ibb if k == 1 else np.float32(1.0)) * gfm[k]
                        tot += cum[k] * bm
                    target = u * tot
                    acc = 0.0
                    outcome = 8
                    for k in range(9):
                        bm = base_mult[bk, k] * (ibb if k == 1 else np.float32(1.0)) * gfm[k]
                        acc += cum[k] * bm
                        if target < acc:
                            outcome = k
                            break

                    # Empirical base-running transition: sample (post_state, runs)
                    rng_state ^= rng_state << np.uint64(13)
                    rng_state ^= rng_state >> np.uint64(7)
                    rng_state ^= rng_state << np.uint64(17)
                    u2 = (np.float64(rng_state >> np.uint64(11))
                          * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                    if runner_beta != 0.0 and track == 1 and (state & 7) != 0:
                        zr = 0.0; nr = 0
                        if bp_home0 >= 0:
                            zr += bat_advz[0, bp_home0]; nr += 1
                        if bp_home1 >= 0:
                            zr += bat_advz[0, bp_home1]; nr += 1
                        if bp_home2 >= 0:
                            zr += bat_advz[0, bp_home2]; nr += 1
                        bz = runner_beta * (zr / nr) if nr > 0 else 0.0
                        tot = 0.0; prev = 0.0
                        for kk in range(K):
                            pk = emp_cumprob[state, outcome, kk] - prev
                            prev = emp_cumprob[state, outcome, kk]
                            w = 1.0 + bz * emp_runner_adv[state, outcome, kk]
                            if w < 0.0:
                                w = 0.0
                            tot += pk * w
                        targ = u2 * tot; acc = 0.0; prev = 0.0; sel = 0
                        for kk in range(K):
                            sel = kk
                            pk = emp_cumprob[state, outcome, kk] - prev
                            prev = emp_cumprob[state, outcome, kk]
                            w = 1.0 + bz * emp_runner_adv[state, outcome, kk]
                            if w < 0.0:
                                w = 0.0
                            acc += pk * w
                            if targ < acc:
                                break
                    else:
                        sel = 0
                        for kk in range(K):
                            sel = kk
                            if u2 < emp_cumprob[state, outcome, kk]:
                                break
                    new_s = emp_post_state[state, outcome, sel]
                    r     = emp_runs_table[state, outcome, sel]

                    inning_runs += r
                    if track == 1:
                        _s = home_batter_pos % 9
                        home_oc[sim, outcome] += 1                   # realized outcome dist (diag)
                        for _k in range(9):                          # fed-vs-effective tilt distortion (diag)
                            _bm = base_mult[bk, _k] * (ibb if _k == 1 else np.float32(1.0)) * gfm[_k]
                            home_fed[sim, _k] += cum[_k]
                            home_eff[sim, _k] += cum[_k] * _bm / tot
                        home_bstats[sim, _s, 0] += 1                  # PA
                        if outcome >= 5:                             # 5=1B 6=2B 7=3B 8=HR
                            home_bstats[sim, _s, 1] += 1             # H
                            home_bstats[sim, _s, 3] += outcome - 4   # TB (1B=1..HR=4)
                            if outcome == 8:
                                home_bstats[sim, _s, 2] += 1         # HR
                        home_bstats[sim, _s, 4] += r                 # RBI = runs on this PA
                        if outcome == 0:
                            home_bstats[sim, _s, 6] += 1             # K
                        if away_using_starter and outcome == 0:
                            away_sk[sim] += 1                        # away starter K
                        if away_using_starter:
                            # starter OUTS recorded this PA (see home_so_a note). Reads state only — no RNG.
                            _oo1 = state >> 3
                            away_so_a[sim] += (3 - _oo1) if new_s >= 24 else ((new_s >> 3) - _oo1)
                        # --- inline _credit_runs_and_advance (R attribution + base identity) ---
                        nn = 0
                        if bp_home2 >= 0: tmp[nn] = bp_home2; nn += 1   # 3rd (lead)
                        if bp_home1 >= 0: tmp[nn] = bp_home1; nn += 1   # 2nd
                        if bp_home0 >= 0: tmp[nn] = bp_home0; nn += 1   # 1st
                        tmp[nn] = _s; nn += 1                          # batter scores last
                        for _i in range(r):
                            if _i < nn:
                                home_bstats[sim, tmp[_i], 5] += 1      # R (run scored)
                        bp_home0 = -1; bp_home1 = -1; bp_home2 = -1
                        if new_s < 24:
                            _j = r
                            if ((new_s >> 2) & 1) == 1 and _j < nn: bp_home2 = tmp[_j]; _j += 1
                            if ((new_s >> 1) & 1) == 1 and _j < nn: bp_home1 = tmp[_j]; _j += 1
                            if (new_s & 1) == 1 and _j < nn:        bp_home0 = tmp[_j]; _j += 1
                    if away_using_starter:
                        away_er += r
                        away_bf += 1
                        if outcome == 1 or outcome == 2 or outcome >= 5:   # reached base
                            away_sp_br += 1
                        if stochastic_pitches == 1:
                            rng_state ^= rng_state << np.uint64(13)
                            rng_state ^= rng_state >> np.uint64(7)
                            rng_state ^= rng_state << np.uint64(17)
                            u_pc = (np.float64(rng_state >> np.uint64(11))
                                    * (1.0 / np.float64(np.uint64(0x1FFFFFFFFFFFFF))))
                            psel = 0
                            for pk in range(pitch_resid_cum.shape[1]):
                                psel = pk
                                if u_pc < pitch_resid_cum[outcome, pk]:
                                    break
                            p_add = away_ppo[outcome] + pitch_resid_vals[outcome, psel]
                            if p_add < 1.0:
                                p_add = 1.0
                            away_pitches += p_add
                        else:
                            away_pitches += away_ppo[outcome]
                    else:
                        if away_cur_rel >= 0:
                            away_cur_rel_bf += 1
                            away_cur_rel_er += r

                    # Walk-off — full-game mode only. In F5 mode (stop_on_tie) the
                    # bottom of the 5th is played to completion (no walk-off).
                    if stop_on_tie == 0 and inning >= n_innings - 1:
                        if home_runs + inning_runs > away_runs:
                            home_runs += inning_runs
                            home_batter_pos += 1
                            break

                    if new_s >= 24:
                        home_runs += inning_runs
                        home_batter_pos += 1   # out-maker's slot is done; next inning starts with the next batter
                        break
                    else:
                        state = new_s
                        home_batter_pos += 1

                first_iter = 0   # resume seeding applies to the FIRST iteration only
                # ── F5 snapshot: score at the END OF INNING 5 (both halves complete).
                # Reached here only after the bottom half of `inning` finished; the
                # bottom-5 is ALWAYS played (walk-off/skip logic fires only at
                # inning >= n_innings-1), so at inning==4 this is exactly the F5 score.
                # Gated on track==1: the portfolio full-game sim runs track=1 and passes
                # real (n_sims,) arrays; other paths pass size-1 dummies (never indexed).
                # RFI snapshot: runs at END OF INNING 1 (both halves complete; bottom-1
                # is always played). inning is 0-based ⇒ inning==0 is the 1st inning.
                if track == 1 and inning == 0:
                    home_f1[sim] = home_runs
                    away_f1[sim] = away_runs
                if track == 1 and inning == 4:
                    home_f5[sim] = home_runs
                    away_f5[sim] = away_runs
                # F8 snapshot: score at END OF INNING 8 (both halves complete; bottom-8 is always
                # played — skip logic fires only at inning >= n_innings-1). Same track==1 gating as F5.
                if track == 1 and inning == 7:
                    home_f8[sim] = home_runs
                    away_f8[sim] = away_runs
                # After each complete inning: increment counter.
                # stop_on_tie=1 (F5): always stop after n_innings, preserving ties.
                # stop_on_tie=0 (full game): continue in extras until tie is broken.
                inning += 1
                if inning >= n_innings:
                    if stop_on_tie or home_runs != away_runs:
                        break
                    elif track == 1:
                        went_extra[sim] = 1   # regulation ended tied ⇒ extra innings

            scores[sim, 0] = home_runs
            scores[sim, 1] = away_runs
            if track == 1:                                # (else home_sbf/away_sbf are shape-1 dummies)
                home_sbf[sim] = home_bf                    # starter batters-faced (PA) this sim
                away_sbf[sim] = away_bf

        return scores

    return simulate_games


# Build numba function at module load time
_simulate_games_numba = _make_simulate_games_numba()


def _credit_runs_and_advance(bp, batter_slot, new_s, r):
    """Lead-runner-first run attribution + base-identity reconstruction.

    bp = [slot_on_1st, slot_on_2nd, slot_on_3rd]  (-1 = empty)  BEFORE the PA.
    Returns (scorers, new_bp): scorers = lineup slots that scored on this PA
    (len == r exactly), new_bp = base identities AFTER the PA, consistent with the
    sampled post-state `new_s`.

    The empirical base-out table is ANONYMOUS (counts, not identities), so identity
    is reconstructed by the no-passing rule: lead runners (3rd→2nd→1st→batter) score
    first; on outs the trailing runners/batter are retired first; surviving runners
    fill the new occupied bases most-advanced-first. Run COUNT is exact (== r); only
    the which-runner assignment is heuristic (correct in the overwhelming majority).
    """
    order = [b for b in (bp[2], bp[1], bp[0]) if b >= 0]   # 3rd, 2nd, 1st (lead-first)
    order.append(batter_slot)                               # batter scores last (HR-type only)
    scorers = order[:r]
    if new_s >= 24:                                         # inning over → bases cleared
        return scorers, [-1, -1, -1]
    survivors = order[r:]                                   # didn't score, advanced-order
    occ = (new_s & 1, (new_s >> 1) & 1, (new_s >> 2) & 1)   # (1st, 2nd, 3rd) occupancy
    new_bp = [-1, -1, -1]
    j = 0
    for base_idx in (2, 1, 0):                              # fill 3rd, 2nd, 1st most-advanced first
        if occ[base_idx] and j < len(survivors):
            new_bp[base_idx] = survivors[j]; j += 1
    return scorers, new_bp


def _simulate_games_numpy(
    home_cum_probs:    np.ndarray,
    away_cum_probs:    np.ndarray,
    home_bull_cum:     np.ndarray,   # empty-pen (no relievers) league-blend fallback
    away_bull_cum:     np.ndarray,
    emp_post_state:    np.ndarray,   # (24, 9, K) int8
    emp_runs_table:    np.ndarray,   # (24, 9, K) int8
    emp_cumprob:       np.ndarray,   # (24, 9, K) float32
    home_exit_lut:     np.ndarray,   # (LUT_SIZE,) f64 — home starter removal-hazard table over live axes (exit_model.py)
    away_exit_lut:     np.ndarray,   # (LUT_SIZE,) f64 — away starter removal-hazard table over live axes
    home_ppo:          np.ndarray,   # (N_OUTCOMES,) float64 — home starter pitches/outcome
    away_ppo:          np.ndarray,   # (N_OUTCOMES,) float64 — away starter pitches/outcome
    pitch_resid_vals:  np.ndarray,   # (N_OUTCOMES, MAXP) float64 — zero-mean residual values
    pitch_resid_cum:   np.ndarray,   # (N_OUTCOMES, MAXP) float64 — cumulative prob
    stochastic_pitches: int,         # 1 = draw per-PA pitch count; 0 = deterministic mean
    home_rel_cum:      np.ndarray,   # (n_home_rel, 9, 9) RAW per-slot vectors
    home_rel_role:     np.ndarray,   # (n_home_rel,) int8
    home_rel_stamina:  np.ndarray,   # (n_home_rel,) int8
    home_closer_idx:   int,
    away_rel_cum:      np.ndarray,
    away_rel_role:     np.ndarray,
    away_rel_stamina:  np.ndarray,
    away_closer_idx:   int,
    base_mult:         np.ndarray,   # (N_BUCKET, N_OUTCOMES) f32 — base-state log-odds tilt
    base_bucket:       np.ndarray,   # (8,) int8 — (state & 7) → base bucket
    rel_hazard:        np.ndarray,   # reliever pull hazard table (REL_HAZARD_SHAPE)
    home_rel_throws:   np.ndarray,   # (n_home_rel,) int8 — L=0/R=1
    away_rel_throws:   np.ndarray,   # (n_away_rel,) int8
    home_bat_stand:    np.ndarray,   # (9,) int8 — home batter stand per slot (L=0/R=1/unknown=-1)
    away_bat_stand:    np.ndarray,   # (9,) int8 — away batter stand per slot
    home_bat_ibb:      np.ndarray,   # (9,) f32 — home batter IBB propensity (× BB in 1B-open)
    away_bat_ibb:      np.ndarray,   # (9,) f32 — away batter IBB propensity
    rng_seeds:         np.ndarray,
    gf_mult:           np.ndarray,   # (n_sims, 2, 2, 9) f32 — [sim, off_side(0=home,1=away), phase(0=vs SP,1=vs pen), outcome]; ones=off
    home_rel_clf:      np.ndarray = None,   # (n_home_rel, 6) f64 — conditional-logit feature bundle
    away_rel_clf:      np.ndarray = None,   # (n_away_rel, 6) f64
    clf_w:             np.ndarray = None,    # (10,) f64 — conditional-logit interaction weights
    clf_on:            int = 0,              # 1 = learned selection; 0 = deterministic
    n_innings:         int = 9,
    stop_on_tie:       int = 0,      # 1 = stop after n_innings even if tied (F5 semantics)
    track:             int = 0,      # 1 = accumulate per-batter [PA,H,HR,TB,RBI] by lineup slot
    steal_cfg:         np.ndarray = None,   # accepted for API parity; IGNORED in numpy fallback (production = numba steal)
) -> np.ndarray:
    """
    NumPy fallback — same dynamic-hazard logic as the Numba kernel, but slower.
    Used when numba is not available.

    NOTE: as of the per-batter tracking work, the track=1 path is handled by the
    FAST numba kernel (`_simulate_games_numba`). This pure-Python kernel is retained
    as the VALIDATION-ONLY reference for that tracking (and as the no-numba fallback).
    Reference/validation only — reach it via simulate_game(force_numpy_track=True).
    """
    n_sims = len(rng_seeds)
    K = emp_post_state.shape[2]
    scores = np.zeros((n_sims, 2), dtype=np.int32)
    # Per-batter counting stats by lineup slot: [PA, H, HR, TB, RBI, R]. RBI = runs on
    # the batter's own PA; R (runs scored) comes from the lead-runner-first identity
    # overlay (_credit_runs_and_advance) since the base-out table is anonymous. Both
    # RBI and R sum EXACTLY to team runs. HRR = H + R + RBI.
    home_bstats = np.zeros((n_sims, 9, 7), dtype=np.int32) if track else None  # [PA,H,HR,TB,RBI,R,K]
    away_bstats = np.zeros((n_sims, 9, 7), dtype=np.int32) if track else None
    home_sk = np.zeros(n_sims, dtype=np.int32) if track else None   # home starter strikeouts (while SP in)
    away_sk = np.zeros(n_sims, dtype=np.int32) if track else None   # away starter strikeouts

    for sim in range(n_sims):
        rng = np.random.default_rng(int(rng_seeds[sim]))

        home_runs = away_runs = 0
        home_bf = away_bf = home_er = away_er = 0
        home_sp_br = away_sp_br = 0   # baserunners allowed by each starter (GAM jam signal)
        home_pitches = away_pitches = 0.0   # true pitch count → effective-BF hook
        home_using_starter = away_using_starter = True
        away_batter_pos = home_batter_pos = 0

        # Bullpen-manager state (individual sequenced relievers)
        n_home_rel = home_rel_cum.shape[0]; n_away_rel = away_rel_cum.shape[0]
        # MLB 5.10(g) support: which arm (if any) is the synthetic POSITION PLAYER.
        # rel_clf col 10 is an identity flag set unconditionally by build_bullpen_profiles, so
        # this works on the deterministic path too. -1 when the feature is off (the default).
        home_pp_idx = -1
        for _i in range(n_home_rel):
            if home_rel_clf[_i, 10] > 0.5:
                home_pp_idx = _i
        away_pp_idx = -1
        for _i in range(n_away_rel):
            if away_rel_clf[_i, 10] > 0.5:
                away_pp_idx = _i
        home_cur_rel = away_cur_rel = -1
        home_cur_rel_bf = away_cur_rel_bf = 0
        home_cur_rel_er = away_cur_rel_er = 0
        home_rel_used = away_rel_used = 0

        MAX_INNINGS = 20
        inning = 0
        while inning < MAX_INNINGS:

            # Away half-inning (home pitcher)
            state = (2 if (stop_on_tie == 0 and inning >= n_innings) else 0)  # ghost runner on 2B in extras
            if track:
                bp = ([-1, (away_batter_pos - 1) % 9, -1]
                      if (stop_on_tie == 0 and inning >= n_innings) else [-1, -1, -1])
            inning_runs = 0; is_eoi = 1
            while True:
                if home_using_starter:
                    _pc = int(home_pitches)
                    _pc = 130 if _pc > 130 else (0 if _pc < 0 else _pc)
                    _er = int(home_er); _er = _er if _er < 8 else 8
                    _br = int(home_sp_br); _br = _br if _br < 6 else 6
                    if home_exit_lut.shape[0] == 4138291:  # v2 EXACT layout (see main-kernel comment)
                        _sd = int(home_runs - away_runs)   # pitcher's team lead (positive = winning)
                        _sd = -4 if _sd < -4 else (4 if _sd > 4 else _sd)
                        _sd += 4
                        _brv = int(home_sp_br); _brv = _brv if _brv < 14 else 14   # v2 br cap = training p99.9
                        if int(is_eoi) == 1:  # inning start: bases empty or the extras ghost (state==2)
                            _L = home_exit_lut[3819960 + (((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 2 + (1 if int(state) == 2 else 0)]
                        else:                 # mid-inning: full (pc,er,sd,br,base_out) dense cell — zero approximation
                            _L = home_exit_lut[(((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 24 + int(state)]
                        h = 1.0 / (1.0 + np.exp(-_L))
                        _hs = home_exit_lut[4138290]
                        h = _hs + (1.0 - _hs) * h
                    elif home_exit_lut.shape[0] > 2389:  # live score_diff axis (exit_model SD layout: pc×er×sd×eoi)
                        _sd = int(home_runs - away_runs)   # pitcher's team lead (positive = winning)
                        _sd = -4 if _sd < -4 else (4 if _sd > 4 else _sd)
                        _sd += 4
                        h = 1.0 / (1.0 + np.exp(-(home_exit_lut[((_pc * 9 + _er) * 9 + _sd) * 2 + int(is_eoi)]
                                                  + home_exit_lut[21222 + int(state)] + home_exit_lut[21246 + _br])))
                    else:
                        h = 1.0 / (1.0 + np.exp(-(home_exit_lut[(_pc * 9 + _er) * 2 + int(is_eoi)]
                                                  + home_exit_lut[2358 + int(state)] + home_exit_lut[2382 + _br])))
                    if rng.random() < h:
                        home_using_starter = False
                is_eoi = 0

                if home_using_starter:
                    cum = away_cum_probs[away_batter_pos % 9, min(away_batter_pos, away_cum_probs.shape[1] - 1)]
                elif n_home_rel > 0:
                    home_lead = home_runs - away_runs
                    save_sit = 1 if (inning >= SAVE_INNING_IDX and 1 <= home_lead <= SAVE_LEAD_MAX) else 0
                    blowout  = 1 if abs(home_lead) > BLOWOUT_MARGIN else 0
                    nrun = (state & 1) + ((state >> 1) & 1) + ((state >> 2) & 1)
                    hi_lev = 1 if (inning >= HIGH_LEV_INNING_IDX and abs(home_lead) <= HIGH_LEV_MARGIN
                                   and nrun >= 2 and (state // 8) <= 1) else 0
                    if home_cur_rel < 0:
                        need = True
                    else:
                        rbf = min(home_cur_rel_bf, REL_BF_MAX); rer = min(home_cur_rel_er, REL_ER_MAX)
                        rinn = min(inning, REL_INN_MAX - 1)
                        rsd = int(np.clip(home_lead + REL_SD_OFF, 0, REL_SD_MAX))
                        need = bool(rng.random() < rel_hazard[rbf, rer, rinn, rsd])
                    if (is_eoi == 1 and save_sit == 1 and home_closer_idx >= 0
                            and ((home_rel_used >> home_closer_idx) & 1) == 0):
                        need = True
                    if hi_lev == 1 and home_cur_rel_bf >= 3 and home_cur_rel > 1:
                        need = True
                    if need:
                        nL = sum(1 for jj in range(3) if away_bat_stand[(away_batter_pos+jj) % 9] == 0)
                        nR = sum(1 for jj in range(3) if away_bat_stand[(away_batter_pos+jj) % 9] == 1)
                        want = 0 if nL > nR else (1 if nR > nL else -1)
                        # MLB 5.10(g): block the position-player arm unless the game is decided
                        # (trailing 8+, leading 10+, or extra innings). inning is 0-based.
                        _h_ppb = (home_pp_idx if not (inning >= 9 or home_lead <= -PP_LOSING_BY or home_lead >= PP_WINNING_BY) else -1)
                        if clf_on == 1:
                            alead = abs(home_lead)
                            late9 = 1 if inning >= SAVE_INNING_IDX else 0
                            close_late = 1 if (inning >= HIGH_LEV_INNING_IDX and alead <= 3) else 0
                            _li = float(LI_TABLE[min(inning, LI_N_INN - 1), 0,
                                                 min(state, 23),
                                                 max(-LI_SD_MAX, min(LI_SD_MAX, home_lead)) + LI_SD_OFF])
                            _jam = 1.0 if (nrun >= 2 and (state // 8) <= 1) else 0.0
                            nxt = _clf_choose(home_rel_used, n_home_rel, home_rel_clf, home_rel_throws,
                                              clf_w, inning + 1, save_sit, blowout, late9, close_late,
                                              _li, float(max(0, home_lead)), float(max(0, -home_lead)),
                                              _jam, want, rng.random(), _h_ppb)
                        else:
                            nxt = _pick_reliever(home_rel_used, n_home_rel, home_rel_role,
                                                 home_closer_idx, inning, save_sit, blowout, hi_lev,
                                                 home_lead, _h_ppb)
                            nxt = _platoon_pick(nxt, home_rel_used, n_home_rel, home_rel_role,
                                                home_closer_idx, inning, save_sit, blowout,
                                                hi_lev, home_rel_throws, want, _h_ppb)
                        if nxt < 0:                          # pen exhausted → recycle worst used arm
                            for ri in range(n_home_rel - 1, -1, -1):
                                if (home_rel_used >> ri) & 1 == 1:
                                    nxt = ri; break
                        if nxt >= 0:
                            home_cur_rel = nxt; home_cur_rel_bf = 0; home_cur_rel_er = 0
                            home_rel_used |= (1 << nxt)
                    cum = home_rel_cum[home_cur_rel, away_batter_pos % 9]
                else:
                    cum = home_bull_cum     # empty pen (no relievers) → league blend
                u       = rng.random()
                # base-state tilt (raw vector → renormalize) with the same single draw u;
                # 1B-open bucket (2): BB scaled by the away batter's IBB propensity.
                bk      = base_bucket[state & 7]
                ibb = away_bat_ibb[away_batter_pos % 9] if bk == 2 else 1.0
                gfm = gf_mult[sim, 1, 0] if home_using_starter else gf_mult[sim, 1, 1]   # away off vs home SP(0)/pen(1)
                tot = 0.0
                for _k in range(N_OUTCOMES):
                    tot += cum[_k] * base_mult[bk, _k] * (ibb if _k == 1 else 1.0) * gfm[_k]
                target = u * tot
                acc = 0.0; outcome = N_OUTCOMES - 1
                for _k in range(N_OUTCOMES):
                    acc += cum[_k] * base_mult[bk, _k] * (ibb if _k == 1 else 1.0) * gfm[_k]
                    if target < acc:
                        outcome = _k; break
                u2      = rng.random()
                sel     = min(int(np.searchsorted(emp_cumprob[state, outcome], u2, side="right")), K - 1)
                new_s   = int(emp_post_state[state, outcome, sel])
                r       = int(emp_runs_table[state, outcome, sel])
                inning_runs += r
                if track:
                    _s = away_batter_pos % 9
                    away_bstats[sim, _s, 0] += 1                 # PA
                    if outcome >= 5:                             # 5=1B 6=2B 7=3B 8=HR
                        away_bstats[sim, _s, 1] += 1             # H
                        away_bstats[sim, _s, 3] += outcome - 4   # TB
                        if outcome == 8:
                            away_bstats[sim, _s, 2] += 1         # HR
                    away_bstats[sim, _s, 4] += r                 # RBI (runs on this PA)
                    if outcome == 0:
                        away_bstats[sim, _s, 6] += 1             # K (batter strikeout)
                    _sc, bp = _credit_runs_and_advance(bp, _s, new_s, r)
                    for _rs in _sc:
                        away_bstats[sim, _rs, 5] += 1            # R (run scored)
                    if home_using_starter and outcome == 0:
                        home_sk[sim] += 1                        # home starter strikeout
                if home_using_starter:
                    home_er += r; home_bf += 1
                    if outcome == 1 or outcome == 2 or outcome >= 5:
                        home_sp_br += 1
                    if stochastic_pitches == 1:
                        psel = min(int(np.searchsorted(pitch_resid_cum[outcome], rng.random(),
                                                       side="right")), pitch_resid_cum.shape[1] - 1)
                        home_pitches += max(home_ppo[outcome] + pitch_resid_vals[outcome, psel], 1.0)
                    else:
                        home_pitches += home_ppo[outcome]
                else:
                    if home_cur_rel >= 0:
                        home_cur_rel_bf += 1
                        home_cur_rel_er += r
                away_batter_pos += 1   # lineup advances on every PA, incl. the 3rd out
                if new_s >= 24:
                    away_runs += inning_runs; break
                state = new_s

            # Skip bottom of final inning only in full-game mode (not F5).
            if stop_on_tie == 0 and inning == n_innings - 1 and home_runs > away_runs:
                inning += 1; break

            # Home half-inning (away pitcher)
            state = (2 if (stop_on_tie == 0 and inning >= n_innings) else 0)  # ghost runner on 2B in extras
            if track:
                bp = ([-1, (home_batter_pos - 1) % 9, -1]
                      if (stop_on_tie == 0 and inning >= n_innings) else [-1, -1, -1])
            inning_runs = 0; is_eoi = 1
            while True:
                if away_using_starter:
                    _pc = int(away_pitches)
                    _pc = 130 if _pc > 130 else (0 if _pc < 0 else _pc)
                    _er = int(away_er); _er = _er if _er < 8 else 8
                    _br = int(away_sp_br); _br = _br if _br < 6 else 6
                    if away_exit_lut.shape[0] == 4138291:  # v2 EXACT layout (see main-kernel comment)
                        _sd = int(away_runs - home_runs)   # pitcher's team lead (positive = winning)
                        _sd = -4 if _sd < -4 else (4 if _sd > 4 else _sd)
                        _sd += 4
                        _brv = int(away_sp_br); _brv = _brv if _brv < 14 else 14   # v2 br cap = training p99.9
                        if int(is_eoi) == 1:  # inning start: bases empty or the extras ghost (state==2)
                            _L = away_exit_lut[3819960 + (((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 2 + (1 if int(state) == 2 else 0)]
                        else:                 # mid-inning: full (pc,er,sd,br,base_out) dense cell — zero approximation
                            _L = away_exit_lut[(((_pc * 9 + _er) * 9 + _sd) * 15 + _brv) * 24 + int(state)]
                        h = 1.0 / (1.0 + np.exp(-_L))
                        _hs = away_exit_lut[4138290]
                        h = _hs + (1.0 - _hs) * h
                    elif away_exit_lut.shape[0] > 2389:  # live score_diff axis (exit_model SD layout: pc×er×sd×eoi)
                        _sd = int(away_runs - home_runs)   # pitcher's team lead (positive = winning)
                        _sd = -4 if _sd < -4 else (4 if _sd > 4 else _sd)
                        _sd += 4
                        h = 1.0 / (1.0 + np.exp(-(away_exit_lut[((_pc * 9 + _er) * 9 + _sd) * 2 + int(is_eoi)]
                                                  + away_exit_lut[21222 + int(state)] + away_exit_lut[21246 + _br])))
                    else:
                        h = 1.0 / (1.0 + np.exp(-(away_exit_lut[(_pc * 9 + _er) * 2 + int(is_eoi)]
                                                  + away_exit_lut[2358 + int(state)] + away_exit_lut[2382 + _br])))
                    if rng.random() < h:
                        away_using_starter = False
                is_eoi = 0

                if away_using_starter:
                    cum = home_cum_probs[home_batter_pos % 9, min(home_batter_pos, home_cum_probs.shape[1] - 1)]
                elif n_away_rel > 0:
                    away_lead = away_runs - home_runs
                    save_sit = 1 if (inning >= SAVE_INNING_IDX and 1 <= away_lead <= SAVE_LEAD_MAX) else 0
                    blowout  = 1 if abs(away_lead) > BLOWOUT_MARGIN else 0
                    nrun = (state & 1) + ((state >> 1) & 1) + ((state >> 2) & 1)
                    hi_lev = 1 if (inning >= HIGH_LEV_INNING_IDX and abs(away_lead) <= HIGH_LEV_MARGIN
                                   and nrun >= 2 and (state // 8) <= 1) else 0
                    if away_cur_rel < 0:
                        need = True
                    else:
                        rbf = min(away_cur_rel_bf, REL_BF_MAX); rer = min(away_cur_rel_er, REL_ER_MAX)
                        rinn = min(inning, REL_INN_MAX - 1)
                        rsd = int(np.clip(away_lead + REL_SD_OFF, 0, REL_SD_MAX))
                        need = bool(rng.random() < rel_hazard[rbf, rer, rinn, rsd])
                    if (is_eoi == 1 and save_sit == 1 and away_closer_idx >= 0
                            and ((away_rel_used >> away_closer_idx) & 1) == 0):
                        need = True
                    if hi_lev == 1 and away_cur_rel_bf >= 3 and away_cur_rel > 1:
                        need = True
                    if need:
                        nL = sum(1 for jj in range(3) if home_bat_stand[(home_batter_pos+jj) % 9] == 0)
                        nR = sum(1 for jj in range(3) if home_bat_stand[(home_batter_pos+jj) % 9] == 1)
                        want = 0 if nL > nR else (1 if nR > nL else -1)
                        # MLB 5.10(g): block the position-player arm unless the game is decided
                        # (trailing 8+, leading 10+, or extra innings). inning is 0-based.
                        _a_ppb = (away_pp_idx if not (inning >= 9 or away_lead <= -PP_LOSING_BY or away_lead >= PP_WINNING_BY) else -1)
                        if clf_on == 1:
                            alead = abs(away_lead)
                            late9 = 1 if inning >= SAVE_INNING_IDX else 0
                            close_late = 1 if (inning >= HIGH_LEV_INNING_IDX and alead <= 3) else 0
                            _li = float(LI_TABLE[min(inning, LI_N_INN - 1), 1,
                                                 min(state, 23),
                                                 max(-LI_SD_MAX, min(LI_SD_MAX, away_lead)) + LI_SD_OFF])
                            _jam = 1.0 if (nrun >= 2 and (state // 8) <= 1) else 0.0
                            nxt = _clf_choose(away_rel_used, n_away_rel, away_rel_clf, away_rel_throws,
                                              clf_w, inning + 1, save_sit, blowout, late9, close_late,
                                              _li, float(max(0, away_lead)), float(max(0, -away_lead)),
                                              _jam, want, rng.random(), _a_ppb)
                        else:
                            nxt = _pick_reliever(away_rel_used, n_away_rel, away_rel_role,
                                                 away_closer_idx, inning, save_sit, blowout, hi_lev,
                                                 away_lead, _a_ppb)
                            nxt = _platoon_pick(nxt, away_rel_used, n_away_rel, away_rel_role,
                                                away_closer_idx, inning, save_sit, blowout,
                                                hi_lev, away_rel_throws, want, _a_ppb)
                        if nxt < 0:                          # pen exhausted → recycle worst used arm
                            for ri in range(n_away_rel - 1, -1, -1):
                                if (away_rel_used >> ri) & 1 == 1:
                                    nxt = ri; break
                        if nxt >= 0:
                            away_cur_rel = nxt; away_cur_rel_bf = 0; away_cur_rel_er = 0
                            away_rel_used |= (1 << nxt)
                    cum = away_rel_cum[away_cur_rel, home_batter_pos % 9]
                else:
                    cum = away_bull_cum     # empty pen (no relievers) → league blend
                u       = rng.random()
                # base-state tilt (raw vector → renormalize) with the same single draw u;
                # 1B-open bucket (2): BB scaled by the home batter's IBB propensity.
                bk      = base_bucket[state & 7]
                ibb = home_bat_ibb[home_batter_pos % 9] if bk == 2 else 1.0
                gfm = gf_mult[sim, 0, 0] if away_using_starter else gf_mult[sim, 0, 1]   # home off vs away SP(0)/pen(1)
                tot = 0.0
                for _k in range(N_OUTCOMES):
                    tot += cum[_k] * base_mult[bk, _k] * (ibb if _k == 1 else 1.0) * gfm[_k]
                target = u * tot
                acc = 0.0; outcome = N_OUTCOMES - 1
                for _k in range(N_OUTCOMES):
                    acc += cum[_k] * base_mult[bk, _k] * (ibb if _k == 1 else 1.0) * gfm[_k]
                    if target < acc:
                        outcome = _k; break
                u2      = rng.random()
                sel     = min(int(np.searchsorted(emp_cumprob[state, outcome], u2, side="right")), K - 1)
                new_s   = int(emp_post_state[state, outcome, sel])
                r       = int(emp_runs_table[state, outcome, sel])
                inning_runs += r
                if track:
                    _s = home_batter_pos % 9
                    home_bstats[sim, _s, 0] += 1                 # PA
                    if outcome >= 5:                             # 5=1B 6=2B 7=3B 8=HR
                        home_bstats[sim, _s, 1] += 1             # H
                        home_bstats[sim, _s, 3] += outcome - 4   # TB
                        if outcome == 8:
                            home_bstats[sim, _s, 2] += 1         # HR
                    home_bstats[sim, _s, 4] += r                 # RBI (runs on this PA)
                    if outcome == 0:
                        home_bstats[sim, _s, 6] += 1             # K (batter strikeout)
                    _sc, bp = _credit_runs_and_advance(bp, _s, new_s, r)
                    for _rs in _sc:
                        home_bstats[sim, _rs, 5] += 1            # R (run scored)
                    if away_using_starter and outcome == 0:
                        away_sk[sim] += 1                        # away starter strikeout
                if away_using_starter:
                    away_er += r; away_bf += 1
                    if outcome == 1 or outcome == 2 or outcome >= 5:
                        away_sp_br += 1
                    if stochastic_pitches == 1:
                        psel = min(int(np.searchsorted(pitch_resid_cum[outcome], rng.random(),
                                                       side="right")), pitch_resid_cum.shape[1] - 1)
                        away_pitches += max(away_ppo[outcome] + pitch_resid_vals[outcome, psel], 1.0)
                    else:
                        away_pitches += away_ppo[outcome]
                else:
                    if away_cur_rel >= 0:
                        away_cur_rel_bf += 1
                        away_cur_rel_er += r
                home_batter_pos += 1   # lineup advances on every PA, incl. the 3rd out
                if stop_on_tie == 0 and inning >= n_innings - 1:   # walk-off: full-game only
                    if home_runs + inning_runs > away_runs:
                        home_runs += inning_runs; break
                if new_s >= 24:
                    home_runs += inning_runs; break
                state = new_s

            inning += 1
            if inning >= n_innings:
                if stop_on_tie or home_runs != away_runs:
                    break

        scores[sim, 0] = home_runs
        scores[sim, 1] = away_runs

    if track:
        return scores, home_bstats, away_bstats, home_sk, away_sk
    return scores


# ═══════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """
    Full simulation output for a single game.

    Markets derived from the joint score distribution:
      moneyline:   P(home wins)
      run_total:   over/under on total runs (at standard half-point lines)
      spread:      P(home covers) at each spread
      first_five:  same markets for first 5 innings only
    """
    p_home_win:    float
    score_dist:    np.ndarray     # (n_sims, 2) int32

    # Market probabilities (at standard lines)
    markets:       dict = field(default_factory=dict)
    # ci90 = Monte-Carlo STANDARD ERROR of each market prob (half-width at n_sims).
    # It shrinks toward 0 as n_sims grows — it is NOT a predictive interval and must
    # NOT be read as confidence in an edge. The real predictive uncertainty comes
    # from the posterior-predictive draws (param_draws) + aleatoric spread, not this.
    ci90:          dict = field(default_factory=dict)   # MC std-error half-widths (see note)

    # Metadata
    n_sims:        int = 0
    elapsed_sec:   float = 0.0
    used_numba:    bool = False
    # Per-batter counting stats by lineup slot, shape (n_sims, 9, 6) = [PA,H,HR,TB,RBI,R].
    # Populated only when simulate_game(track_batter_stats=True); None otherwise.
    # HRR (hits+runs+RBIs) = H + R + RBI per batter.
    home_batter_stats: np.ndarray = None
    away_batter_stats: np.ndarray = None
    # Per-sim starter strikeout counts (n_sims,) — Ks recorded while the starter is in.
    home_starter_k: np.ndarray = None
    away_starter_k: np.ndarray = None
    # Per-sim starter batters-faced (n_sims,) — PAs while the starter is in. With the
    # starter K above this gives starter K/PA; (team total − starter) gives the reliever
    # split. numba-track path only (None on the numpy/no-track paths).
    home_starter_bf: np.ndarray = None
    away_starter_bf: np.ndarray = None
    # Per-sim realized outcome counts (n_sims, 9) over the 9 L3 outcomes — diagnostic only
    # (sources baserunner/run over-production by giving the sim's full realized per-PA dist,
    # incl BB/GO/FO which batter_stats omits). numba-track path only.
    home_outcomes: np.ndarray = None
    away_outcomes: np.ndarray = None
    # Per-sim probability sums (n_sims, 9): home_fed/away_fed = Σ pre-tilt matchup prob cum[k]
    # over PAs; home_eff/away_eff = Σ post-tilt normalized prob (cum*base_mult*ibb*gfm / tot) the
    # draw samples from. fed-vs-eff marginal = the kernel's deterministic, sampling-free tilt
    # distortion per outcome (the base-state/IBB/GF effect on K, BB, …). numba-track path only.
    home_fed: np.ndarray = None
    away_fed: np.ndarray = None
    home_eff: np.ndarray = None
    away_eff: np.ndarray = None
    # F5 snapshot: (n_sims, 2) int32 [home_runs, away_runs] at the END OF INNING 5, drawn
    # jointly with score_dist (same kernel replicate → row-aligned with score_dist and the
    # per-batter arrays). This is the root-fix for the F5↔full-game correlation gap: F5
    # markets read off this instead of the separate run_first_five sim. numba-track path only.
    f5_score_dist: np.ndarray = None
    # F8 snapshot: (n_sims, 2) int32 [home_runs, away_runs] at the END OF INNING 8 — same joint
    # replicate as score_dist/f5_score_dist. Enables sim through-8 cross-team covariance (the CLEAN
    # cross-team cov target, reality 2026 ≈ −0.57; full-game cov is extras-contaminated). Numba track path only.
    f8_score_dist: np.ndarray = None
    # Per-sim starter OUTS recorded (n_sims,) int32 — outs credited while the starter is in
    # (K + GO + FO + the extra out on a GIDP; inning-ending outs counted exactly). Grades the
    # Starter-outs market. numba-track path only (None otherwise).
    home_starter_outs: np.ndarray = None
    away_starter_outs: np.ndarray = None
    # RFI snapshot: (n_sims, 2) int32 [home_runs, away_runs] at the END OF INNING 1 — joint
    # replicate (row-aligned with score_dist). Grades the "run in the 1st inning" market.
    # numba-track path only.
    f1_score_dist: np.ndarray = None
    # Per-sim extra-innings flag (n_sims,) int32 — 1 if regulation ended tied ⇒ the game went to
    # extras, else 0. Grades the extra-innings market. numba-track path only.
    went_extra: np.ndarray = None

    def summary(self) -> str:
        home_mean = self.score_dist[:, 0].mean()
        away_mean = self.score_dist[:, 1].mean()
        total_mean = (self.score_dist[:, 0] + self.score_dist[:, 1]).mean()
        return (
            f"P(home win)={self.p_home_win:.3f}  "
            f"Home {home_mean:.2f} — Away {away_mean:.2f}  "
            f"Total={total_mean:.2f}  "
            f"[{self.n_sims:,} sims in {self.elapsed_sec:.2f}s, "
            f"numba={'✓' if self.used_numba else '✗'}]"
        )


def compute_markets(
    score_dist: np.ndarray,
    spreads:    list[float] = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
    totals:     list[float] = [x + 0.5 for x in range(0, 21)],
) -> tuple[dict, dict]:
    """
    Compute market probabilities and 90% confidence intervals from score distribution.

    CIs use Wilson score interval on the Bernoulli proportion.
    At n=50,000: CI half-width ≈ 0.4% for p≈0.5 — sufficient for edge detection.

    Args:
        score_dist: (n_sims, 2) int32 array of [home_runs, away_runs]
        spreads:    Run line spreads to evaluate P(home covers)
        totals:     Over/under totals to evaluate P(over). Default: 1.5–10.5.
                    Broad default (0.5-20.5) covers the exchange's alternate total ladders
                    for both full-game and F5 markets; extra keys are harmless.

    Returns:
        markets: dict of market → probability
        ci90:    dict of market → 90% CI half-width
    """
    n = len(score_dist)
    home = score_dist[:, 0]
    away = score_dist[:, 1]
    total = home + away

    markets = {}
    ci90    = {}

    def add(key: str, p: float):
        markets[key] = float(p)
        # Wilson CI (90% = z=1.645)
        z = 1.645
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1-p) / n + z**2 / (4 * n**2)) / denom
        ci90[key] = float(half)

    # Moneyline — compute each side directly so tied games aren't allocated to either
    home_wins = (home > away).mean()
    away_wins = (away > home).mean()
    add("moneyline_home", home_wins)
    add("moneyline_away", away_wins)

    # Tie — near-zero for full-game sims (extras resolve ties);
    # meaningful (~10-20%) for F5 sims when stop_on_tie=True
    add("tie", (home == away).mean())

    # Run totals
    for t in totals:
        add(f"over_{t}", (total > t).mean())
        add(f"under_{t}", (total < t).mean())

    # Run line (spread)
    for s in spreads:
        # "home -1.5" means home must win by 2+
        if s < 0:
            add(f"home_{abs(s):.1f}_rl", (home - away > abs(s)).mean())
        else:
            add(f"away_{s:.1f}_rl", (away - home > s).mean())

    # First five: use first 5 innings score distribution
    # (score_dist already handles first_five if n_innings=5 was passed)

    return markets, ci90


# ═══════════════════════════════════════════════════════════════
# MAIN SIMULATION FUNCTION
# ═══════════════════════════════════════════════════════════════

# ── LIVE resume: scalar game-state arrays consumed by the numba kernel ──────────
# si64 (int64, len 24) index layout — see comments in _make_simulate_games_numba's resume block:
#   0 inning(0-based)  1 half(0=top/away batting, 1=bottom/home batting)  2 base-out state
#   3 home_runs  4 away_runs  5 home_bf  6 away_bf  7 home_er  8 away_er  9 home_sp_br 10 away_sp_br
#   11 home_using_starter(0/1)  12 away_using_starter(0/1)  13 home_batter_pos(cumulative PA count)
#   14 away_batter_pos(cumulative)  15 home_cur_rel(-1=none)  16 away_cur_rel  17 home_cur_rel_bf
#   18 away_cur_rel_bf  19 home_cur_rel_er  20 away_cur_rel_er  21 home_rel_used(bitmask)
#   22 away_rel_used(bitmask)  23 is_eoi(1 if the resumed PA is the first of its half-inning)
# sf64 (float64, len 2): 0 home_pitches  1 away_pitches  (cumulative pitch counts driving the hook)
_SI64_KEYS = ["inning", "half", "state", "home_runs", "away_runs", "home_bf", "away_bf",
              "home_er", "away_er", "home_sp_br", "away_sp_br", "home_using_starter",
              "away_using_starter", "home_batter_pos", "away_batter_pos", "home_cur_rel",
              "away_cur_rel", "home_cur_rel_bf", "away_cur_rel_bf", "home_cur_rel_er",
              "away_cur_rel_er", "home_rel_used", "away_rel_used", "is_eoi"]


def _build_start_arrays(start_state: dict | None):
    """(resume_int, si64, sf64) for the kernel. None ⇒ resume=0 + dummies (golden-master path).
    Sensible non-zero defaults: using_starter=1, cur_rel=-1, is_eoi=1."""
    if start_state is None:
        return 0, np.zeros(24, np.int64), np.zeros(2, np.float64)
    s = start_state
    si = np.zeros(24, np.int64)
    for i, k in enumerate(_SI64_KEYS):
        si[i] = int(s.get(k, 0))
    if "home_using_starter" not in s: si[11] = 1
    if "away_using_starter" not in s: si[12] = 1
    if "home_cur_rel" not in s:       si[15] = -1
    if "away_cur_rel" not in s:       si[16] = -1
    if "is_eoi" not in s:             si[23] = 1
    sf = np.array([float(s.get("home_pitches", 0.0)), float(s.get("away_pitches", 0.0))], np.float64)
    return 1, si, sf


def simulate_game(
    home_pa_probs:    np.ndarray,
    away_pa_probs:    np.ndarray,
    home_bull_probs:  np.ndarray,
    away_bull_probs:  np.ndarray,
    baserunning_tables: tuple,           # 5-tuple from load_baserunning_tables()
    n_sims:           int = N_SIMS_DEFAULT,
    n_innings:        int = 9,
    seed=None,
    spreads=None,
    totals=None,
    stop_on_tie:      bool = False,      # True = stop at n_innings even if tied (F5 semantics)
    home_pitcher_params: dict | None = None,  # {pitcher_avg_bf, pitcher_avg_pitches, pitcher_k_pct}
    away_pitcher_params: dict | None = None,
    home_relievers: tuple | None = None,  # (rel_cum, rel_role, rel_stamina[, rel_throws]) → individual manager
    away_relievers: tuple | None = None,
    home_batter_stand: np.ndarray | None = None,  # (9,) int8 L=0/R=1/unknown=-1 → platoon-aware pen pick
    away_batter_stand: np.ndarray | None = None,
    home_batter_ibb: np.ndarray | None = None,  # (9,) f32 per-slot IBB propensity (× BB in 1B-open); 1.0=neutral
    away_batter_ibb: np.ndarray | None = None,
    home_batter_advz: np.ndarray | None = None, # (9,) f64 per-slot runner speed-z (XBT advancement); 0=neutral
    away_batter_advz: np.ndarray | None = None,
    home_batter_steal: np.ndarray | None = None, # (9,) f64 per-slot steal-attempt mult (runner rate/league); 1=neutral
    away_batter_steal: np.ndarray | None = None,
    model_path: str | None = None,
    track_batter_stats: bool = False,   # True → FAST numba track kernel; per-batter [PA,H,HR,TB,RBI,R,K]
    force_numpy_track: bool | None = None,  # validation-only: True → slow numpy reference kernel for the
                                            # track path (cross-check). None/False (default) ⇒ FAST numba
                                            # kernel. No env-var bypass — numba is the unconditional default.
    start_state: dict | None = None,        # LIVE resume: dict of game-state scalars (see _build_start_arrays).
                                            # None (default) ⇒ simulate from game start (resume=0, golden-master).
    steal_cfg: np.ndarray | None = None,    # (11,) f32 running-game steal config (steal_model.build_cfg).
                                            # None / cfg[0]=0 ⇒ no steal (kernel skips block) ⇒ golden-master.
) -> "SimulationResult":
    """
    Run n_sims Monte Carlo game simulations and return markets + score distribution.

    Args:
        home_pa_probs:        PA outcome probs for home lineup vs away starter (9 × 9)
        away_pa_probs:        PA outcome probs for away lineup vs home starter (9 × 9)
        home_bull_probs:      Home bullpen aggregate (9,)
        away_bull_probs:      Away bullpen aggregate (9,)
        baserunning_tables:   5-tuple from load_baserunning_tables()
        n_sims:               Number of Monte Carlo simulations
        n_innings:            Innings to simulate (9 or 5)
        seed:                 None = random; int = reproducible
        spreads:              Run line spreads (default [-1.5, -0.5, 0.5, 1.5])
        totals:               Over/under totals (default 0.5–20.5)
        stop_on_tie:          If True, game ends after exactly n_innings even if tied.
                              Use for F5 markets — preserves the tie outcome.
                              If False (default), tied games continue in extras.
        home_pitcher_params:  Pitcher workload features for the home starter.
                              Keys: pitcher_avg_bf, pitcher_avg_pitches, pitcher_k_pct.
                              Defaults to league average if None.
        away_pitcher_params:  Same for the away starter.
    """
    if spreads is None:
        spreads = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    if totals is None:
        totals = [x + 0.5 for x in range(0, 21)]

    emp_post_c, emp_runs_c, emp_cum_c = baserunning_tables
    if steal_cfg is None:
        steal_cfg = np.zeros(11, dtype=np.float32)   # neutral ⇒ kernel skips steal block (golden master)
    steal_cfg = np.asarray(steal_cfg, dtype=np.float32)
    t0 = time.time()
    actual_seed = seed if seed is not None else int(time.time_ns() & 0xFFFFFFFF)
    rng = np.random.default_rng(actual_seed)

    # Starter matchup arrays: per-bf RAW vectors (continuous TTOP), index [slot, min(bf,30)].
    # ALL pitcher vectors are now RAW (base-state is applied in the kernel per PA);
    # the bullpen blends already arrive raw, so no cumsum. FRINGE starters (NOT in the established-starter
    # set = unproven/unseen arms, < 12 starts) get the STEEPER fringe TTOP curve (build_fringe_ttop.py):
    # they collapse the 2nd time through, which the global curve under-states. (set via exit_model.estab_ttop)
    home_bull_cum = np.ascontiguousarray(home_bull_probs, dtype=np.float32)
    away_bull_cum = np.ascontiguousarray(away_bull_probs, dtype=np.float32)
    # Tier-1 PA-dependence tables (passed to both kernels).
    base_mult_arr   = np.ascontiguousarray(_load_base_mult(), dtype=np.float32)
    base_bucket_arr = np.ascontiguousarray(BASE_BUCKET_LUT, dtype=np.int8)
    rel_hazard_arr  = load_reliever_hazard()   # reliever stochastic pull hazard

    emp_post_c = np.ascontiguousarray(emp_post_c, dtype=np.int8)
    emp_runs_c = np.ascontiguousarray(emp_runs_c, dtype=np.int8)
    emp_cum_c  = np.ascontiguousarray(emp_cum_c,  dtype=np.float32)

    # ── RUNNER-QUALITY base-advancement (#92): per-transition advancement level + per-slot speed-z + β ──
    # adv = base_sum(post_state) + 4·runs, PROB-WEIGHTED-MEAN-centered per (state,outcome). The kernel uses a
    # LINEAR mean-preserving tilt w_k = 1 + β·z·adv_c (NOT exp — exp is convex ⇒ Jensen ⇒ league-wide run
    # inflation). Mean-centering ⇒ Σ p_k·w_k = 1 and E over zero-mean speed-z is unchanged ⇒ LEAGUE-NEUTRAL
    # (fast teams advance more, slow less, aggregate runs preserved). runner_beta=0 ⇒ no tilt ⇒ golden-master.
    _bsum = np.array([((s & 1) + 2 * ((s >> 1) & 1) + 3 * ((s >> 2) & 1)) if s < 24 else 0
                      for s in range(25)], dtype=np.float32)   # base-occupancy sum; 24 (inning over)=0
    _padv = _bsum[np.clip(emp_post_c.astype(np.int64), 0, 24)] + 4.0 * emp_runs_c.astype(np.float32)
    _pk = np.diff(emp_cum_c.astype(np.float64), axis=2, prepend=0.0)   # per-transition prob (padding ⇒ 0)
    _wmean = (_pk * _padv).sum(axis=2, keepdims=True)         # prob-weighted mean adv per (state,outcome)
    emp_runner_adv = np.ascontiguousarray(_padv - _wmean, dtype=np.float32)   # mean-centered ⇒ tilt is neutral
    bat_advz = np.zeros((2, 9), dtype=np.float64)
    if home_batter_advz is not None:
        bat_advz[0, :len(home_batter_advz)] = np.asarray(home_batter_advz, np.float64)[:9]
    if away_batter_advz is not None:
        bat_advz[1, :len(away_batter_advz)] = np.asarray(away_batter_advz, np.float64)[:9]
    bat_steal = np.ones((2, 9), dtype=np.float64)             # per-slot steal-attempt mult (runner's rate/league); 1.0 neutral
    if home_batter_steal is not None:
        bat_steal[0, :len(home_batter_steal)] = np.asarray(home_batter_steal, np.float64)[:9]
    if away_batter_steal is not None:
        bat_steal[1, :len(away_batter_steal)] = np.asarray(away_batter_steal, np.float64)[:9]
    runner_beta = float(os.environ.get("RUNNER_BETA", "0.30"))  # XBT log-odds/SD (data β≈0.30); 0 ⇒ off (golden-master)
    # run-neutral centering: subtract a small offset from all speed-z so the LEAGUE aggregate runs are unchanged
    # (the advancement→runs nonlinearity + OBP-weighted on-base pool skew leave a residual inflation that the
    # raw on-base-mean centering misses). Calibrated so β=0.30 reproduces the β=0 run total (≈STEAL_CAL idea).
    bat_advz = bat_advz - float(os.environ.get("RUNNER_ADV_CENTER", "0.085"))

    # ── Build per-pitcher hazard lookup tables (once, ~10 ms each) ──────────
    def _pitcher_params(params: dict | None) -> tuple[float, float, float]:
        if params is None:
            return 21.9, 83.0, 0.22
        return (
            float(params.get("pitcher_avg_bf",      21.9)),
            float(params.get("pitcher_avg_pitches",  83.0)),
            float(params.get("pitcher_k_pct",         0.22)),
        )

    home_avg_bf, home_avg_pit, home_k_pct = _pitcher_params(home_pitcher_params)
    away_avg_bf, away_avg_pit, away_k_pct = _pitcher_params(away_pitcher_params)

    # ── Starter outing-length removal hazard (exit_model.py) ─────────────────
    # The LightGBM discrete-time hazard is baked, per start, to a dense lookup table over the live
    # state axes (pitch_count × earned_runs × accrued_baserunners × base-out × end-of-inning); the
    # kernel does an O(1) lookup per plate appearance. Pregame features (avg_bf / avg_pitches / k_pct)
    # are fixed within a start, so the whole start is one table. Outing-length variance is produced by
    # the sim generating diverse state trajectories the hazard reacts to — not by injected randomness.
    import exit_model as _exit
    # EXIT_LOGIT_ADJ: optional global removal-hazard logit shift (bake→sim transfer). DEFAULT 0 — the
    # recency-shrunk leash fixed the age bias and most of the level; the true all-starts BF bias is only
    # ~−0.16 (the old −0.30 was over-corrected to a val_kp ≥9-FILTER artifact). Left at 0 (no band-aid);
    # the residual is being closed at the root (over-dispersion / pitch residual). NEGATIVE = longer outings.
    _eadj = float(os.environ.get("EXIT_LOGIT_ADJ", "0.0"))
    _hpid_e = int(home_pitcher_params.get("pitcher_id", -1)) if home_pitcher_params else -1
    _apid_e = int(away_pitcher_params.get("pitcher_id", -1)) if away_pitcher_params else -1
    # Bullpen-fatigue tilt: a gassed pen (heavy late-inning use the prior 2 days) extends ITS OWN starter.
    # Empirical within-pitcher +0.288 BF/SD (t=10.1, R² 0.7%) → β=−0.131 logit/SD on the new hazard scale
    # (BF/logit slope ≈ −2.2). Mean-neutral (z-score) so it leaves the BF mean / tier / age calibration
    # intact. pen_fatigue_z rides in the pitcher_params hook; absent ⇒ 0 ⇒ no tilt (safe).
    _pfb = float(os.environ.get("STARTER_PEN_FATIGUE_BETA", "-0.131"))
    _hpf = _pfb * float((home_pitcher_params or {}).get("pen_fatigue_z", 0.0))
    _apf = _pfb * float((away_pitcher_params or {}).get("pen_fatigue_z", 0.0))
    # Ramp/injury exit-hazard features (pregame-constant; set by pregame_pipeline via _ramp_features → train==serve).
    # Used by the bake iff the loaded model was trained with them; absent ⇒ 0 ⇒ neutral (back-compat).
    def _rf_of(pp):
        pp = pp or {}
        return dict(ramp_budget_delta=float(pp.get("ramp_budget_delta", 0.0)),
                    il_first=float(pp.get("il_first", 0.0)), il_arm_first=float(pp.get("il_arm_first", 0.0)))
    _hls = float((home_pitcher_params or {}).get("mkt_leash_scale", 1.0))   # STARTER_MKT_LEASH override (1.0 = no-op)
    _als = float((away_pitcher_params or {}).get("mkt_leash_scale", 1.0))
    home_exit_lut = _exit.bake_start_lut(home_avg_bf, home_avg_pit, home_k_pct, _eadj + _hpf, _hpid_e, leash_scale=_hls, **_rf_of(home_pitcher_params))
    away_exit_lut = _exit.bake_start_lut(away_avg_bf, away_avg_pit, away_k_pct, _eadj + _apf, _apid_e, leash_scale=_als, **_rf_of(away_pitcher_params))
    # ── Starter TTOP matchup arrays (the FRINGE curve is picked via the established-starter set) ──
    # home_cum = home lineup vs the AWAY starter; away_cum = away lineup vs the HOME starter — so each
    # array's TTOP follows the OPPOSING starter being faced. A starter NOT in the established set
    # (unproven/unseen, < 12 starts) uses the steeper fringe curve. The membership set is re-homed onto
    # the exit-hazard pkl (exit_model.estab_ttop) — a TTOP feature distinct from the baked removal hazard.
    _fr_curve = _load_ttop_curve_fringe()
    _estab = _exit.estab_ttop()
    _use_fr = (_fr_curve is not None) and bool(_estab)   # only discriminate when the membership set is present
    _hp_id = int(home_pitcher_params.get("pitcher_id", -1)) if home_pitcher_params else -1
    _ap_id = int(away_pitcher_params.get("pitcher_id", -1)) if away_pitcher_params else -1
    _home_uses_fr = _use_fr and (_ap_id not in _estab)   # home batters face AWAY starter
    _away_uses_fr = _use_fr and (_hp_id not in _estab)   # away batters face HOME starter
    # depth-aware (per-pitcher) mean-preserve: each array's TTOP is re-centered to the OPPOSING starter's own
    # avg_bf (home batters face the AWAY starter, and vice-versa) so deep aces aren't over-suppressed. Gated.
    home_cum = build_starter_tto_raw(home_pa_probs, curve=(_fr_curve if _home_uses_fr else None), avg_bf=home_avg_bf)
    away_cum = build_starter_tto_raw(away_pa_probs, curve=(_fr_curve if _away_uses_fr else None), avg_bf=away_avg_bf)

    # Per-pitcher pitch-economy vectors (drive the effective-BF hook). Default to
    # the league PITCHES_PER_OUTCOME when no pitcher economy is supplied.
    home_ppo = resolve_pitch_economy(home_pitcher_params)
    away_ppo = resolve_pitch_economy(away_pitcher_params)

    # Per-outcome pitch-count residual distribution for stochastic per-PA draws.
    pitch_resid_vals, pitch_resid_cum = load_pitch_count_dist()
    stochastic_pitches_int = int(STOCHASTIC_PITCHES)

    # ── Individual-reliever bullpen manager arrays (empty → blended-tier path) ──
    def _rel_arrays(relievers):
        if relievers is None or len(relievers[0]) == 0:
            return (np.zeros((0, 9, N_OUTCOMES), np.float32),   # (n_rel, 9 slots, 9 outcomes)
                    np.zeros(0, np.int8), np.zeros(0, np.int8), -1, np.zeros(0, np.int8),
                    np.zeros((0, CLF_N_RELCLF_COLS), np.float64))
        rel_cum, rel_role, rel_stamina = relievers[0], relievers[1], relievers[2]
        rel_cum = np.ascontiguousarray(rel_cum, dtype=np.float32)
        rel_role = np.ascontiguousarray(rel_role, dtype=np.int8)
        rel_stamina = np.ascontiguousarray(rel_stamina, dtype=np.int8)
        cl = np.where(rel_role == 0)[0]
        closer_idx = int(cl[0]) if len(cl) else -1
        # throws-hand (element 4 of the tuple if present; default all-R → no platoon pref)
        if len(relievers) > 4 and relievers[4] is not None and len(relievers[4]):
            rel_throws = np.ascontiguousarray(relievers[4], dtype=np.int8)
        else:
            rel_throws = np.ones(rel_cum.shape[0], dtype=np.int8)
        # conditional-logit feature bundle (element 5; zeros ⇒ clf no-op, golden-master)
        if len(relievers) > 5 and relievers[5] is not None and len(relievers[5]):
            rel_clf = np.ascontiguousarray(relievers[5], dtype=np.float64)
        else:
            # width = the kernel's declared contract, NOT 6: all-zero is a documented clf no-op,
            # but a 6-wide array would still be read at cols 7..10 (silent OOB). Conforming zeros
            # keep the no-op semantics AND satisfy the contract gate below.
            rel_clf = np.zeros((rel_cum.shape[0], CLF_N_RELCLF_COLS), np.float64)
        return rel_cum, rel_role, rel_stamina, closer_idx, rel_throws, rel_clf

    h_rcum, h_rrole, h_rstam, h_cidx, h_rthr, h_rclf = _rel_arrays(home_relievers)
    a_rcum, a_rrole, a_rstam, a_cidx, a_rthr, a_rclf = _rel_arrays(away_relievers)
    # Learned reliever-selection (conditional logit). clf_on=0 ⇒ deterministic path,
    # bit-identical (golden-master). Default ON when a model file exists; env override.
    from build_bullpen_profiles import clf_int_weights as _clf_int_weights
    _clf_w = _clf_int_weights()
    clf_w  = (np.ascontiguousarray(_clf_w, np.float64) if _clf_w is not None
              else np.zeros(CLF_N_INT, np.float64))
    clf_on = (int(os.environ.get("BULLPEN_CLF", "1")) if _clf_w is not None else 0)
    # ── CONTRACT GATE (2026-08-07, see CLF_N_INT) ────────────────────────────────────────────
    # Refuse to run the learned picker unless EVERY array meets the kernel's declared widths.
    # Without this a split deploy (new kernel + old build_bullpen_profiles) reads out of bounds
    # silently and corrupts every reliever choice while looking perfectly healthy — which is
    # exactly what happened on 2026-08-07. Raising is strictly better than serving garbage.
    if clf_on:
        _need = [("clf_w", int(clf_w.shape[0]), CLF_N_INT)]
        for _nm, _a in (("home rel_clf", h_rclf), ("away rel_clf", a_rclf)):
            if _a.shape[0]:                      # empty pen indexes nothing — width is moot
                _need.append((_nm, int(_a.shape[1]), CLF_N_RELCLF_COLS))
        _bad = [(n, got, req) for n, got, req in _need if got < req]
        if _bad:
            raise RuntimeError(
                "CLF CONTRACT VIOLATION — game_simulation requires "
                f"clf_w>={CLF_N_INT}, rel_clf cols>={CLF_N_RELCLF_COLS}; got "
                + ", ".join(f"{n}={got} (need {req})" for n, got, req in _bad)
                + ". build_bullpen_profiles.py is out of sync with this kernel — they are an "
                  "ATOMIC deploy pair. Deploy the matching builder, or unset BULLPEN_CLF.")
    # Per-slot batter stand (L=0/R=1/unknown=-1). Default = -1 everywhere ⇒ no platoon
    # preference ⇒ kernel is bit-identical to the pre-#16 path (golden-master safe).
    home_bat_stand = (np.ascontiguousarray(home_batter_stand, np.int8)
                      if home_batter_stand is not None else np.full(9, -1, np.int8))
    away_bat_stand = (np.ascontiguousarray(away_batter_stand, np.int8)
                      if away_batter_stand is not None else np.full(9, -1, np.int8))
    # Per-slot IBB propensity (× BB in the 1B-open bucket). Default = 1.0 everywhere
    # ⇒ no change ⇒ kernel is bit-identical to the pre-IBB path (golden-master safe).
    home_bat_ibb = (np.ascontiguousarray(home_batter_ibb, np.float32)
                    if home_batter_ibb is not None else np.ones(9, np.float32))
    away_bat_ibb = (np.ascontiguousarray(away_batter_ibb, np.float32)
                    if away_batter_ibb is not None else np.ones(9, np.float32))

    rng_seeds = rng.integers(1, 2**62, size=n_sims, dtype=np.uint64)
    # Latent game-form factor (Tier-2c). Drawn AFTER rng_seeds so the xorshift outcome
    # stream is untouched ⇒ OFF (ones) is bit-identical to the pre-factor sim.
    gf_mult = _build_gf_mult(n_sims, rng)
    stop_on_tie_int = int(stop_on_tie)
    # LIVE resume scalars (resume=0 + dummy arrays when start_state is None → golden-master path).
    resume_int, start_si64, start_sf64 = _build_start_arrays(start_state)
    if resume_int == 1 and not (_NUMBA_AVAILABLE and _simulate_games_numba is not None):
        raise RuntimeError("start_state resume requires the numba kernel (numpy reference path not wired)")

    # Tiny dummy tracking arrays for the non-tracking numba path (track=0). The kernel
    # never writes them when track==0; they exist only to satisfy the typed signature.
    _dummy_bstats = np.zeros((1, 9, 7), np.int32)
    _dummy_sk     = np.zeros(1, np.int32)
    _dummy_oc     = np.zeros((1, 9), np.int32)
    _dummy_fed    = np.zeros((1, 9), np.float64)
    _dummy_f5     = np.zeros(1, np.int32)   # size-1 dummy for the non-track numba path (never indexed; write gated on track==1)

    used_numba = False
    home_bstats = away_bstats = home_sk = away_sk = None
    home_sbf = away_sbf = None                            # starter batters-faced (numba-track path only)
    home_oc = away_oc = None                              # realized outcome counts (numba-track diag)
    home_fed = away_fed = home_eff = away_eff = None      # fed/effective prob sums (numba-track diag)
    home_f5 = away_f5 = None                              # F5 snapshot (numba/numpy track path only)
    home_f8 = away_f8 = None                              # F8 snapshot (end of inning 8; numba track path)
    home_so_arr = away_so_arr = None                     # starter OUTS (numba track path only)
    home_f1 = away_f1 = None                             # RFI snapshot (end of inning 1; numba track path)
    went_extra = None                                    # extra-innings flag (numba track path only)
    # The numba kernel is the UNCONDITIONAL default for BOTH paths (game/team markets and
    # per-batter tracking) — it is ~1000x faster than the numpy kernel and bit-identical
    # on the game/team output. The pure-Python numpy kernel is reachable ONLY by:
    #   • an explicit force_numpy_track=True  (cross-check / reference), or
    #   • numba being unavailable             (fallback, branch below).
    # There is deliberately NO environment-variable bypass: a stray env var must never be
    # able to silently route a call to the 1000x-slower kernel (this footgun cost real
    # debugging time). force_numpy_track defaults to None ⇒ numba.
    _force_numpy_track = bool(force_numpy_track) if force_numpy_track is not None else False
    if track_batter_stats and _force_numpy_track:
        # Validation-only path: the original pure-Python reference kernel.
        score_dist, home_bstats, away_bstats, home_sk, away_sk = _simulate_games_numpy(
            home_cum, away_cum,
            home_bull_cum, away_bull_cum,
            emp_post_c, emp_runs_c, emp_cum_c,
            home_exit_lut, away_exit_lut,
            home_ppo, away_ppo,
            pitch_resid_vals, pitch_resid_cum, stochastic_pitches_int,
            h_rcum, h_rrole, h_rstam, h_cidx,
            a_rcum, a_rrole, a_rstam, a_cidx,
            base_mult_arr, base_bucket_arr, rel_hazard_arr,
            h_rthr, a_rthr, home_bat_stand, away_bat_stand,
            home_bat_ibb, away_bat_ibb,
            rng_seeds, gf_mult, h_rclf, a_rclf, clf_w, clf_on, n_innings, stop_on_tie_int, track=1,
        )
    elif track_batter_stats and _NUMBA_AVAILABLE and _simulate_games_numba is not None:
        # Per-batter tracking now runs in the FAST numba kernel (track=1, in-place arrays).
        home_bstats = np.zeros((n_sims, 9, 7), dtype=np.int32)
        away_bstats = np.zeros((n_sims, 9, 7), dtype=np.int32)
        home_sk     = np.zeros(n_sims, dtype=np.int32)
        away_sk     = np.zeros(n_sims, dtype=np.int32)
        home_sbf    = np.zeros(n_sims, dtype=np.int32)
        away_sbf    = np.zeros(n_sims, dtype=np.int32)
        home_oc     = np.zeros((n_sims, 9), dtype=np.int32)
        away_oc     = np.zeros((n_sims, 9), dtype=np.int32)
        home_fed    = np.zeros((n_sims, 9), dtype=np.float64)
        away_fed    = np.zeros((n_sims, 9), dtype=np.float64)
        home_eff    = np.zeros((n_sims, 9), dtype=np.float64)
        away_eff    = np.zeros((n_sims, 9), dtype=np.float64)
        home_f5     = np.full(n_sims, -1, dtype=np.int32)  # F5 snapshot (end of inning 5); -1 SENTINEL = not yet written
        away_f5     = np.full(n_sims, -1, dtype=np.int32)  # (a resume STARTING past inning 5 never fires it → stays -1)
        home_f8     = np.full(n_sims, -1, dtype=np.int32)  # F8 snapshot (end of inning 8); -1 SENTINEL = not yet written
        away_f8     = np.full(n_sims, -1, dtype=np.int32)
        home_so_arr = np.zeros(n_sims, dtype=np.int32)     # home starter OUTS recorded (while SP in)
        away_so_arr = np.zeros(n_sims, dtype=np.int32)     # away starter OUTS recorded (while SP in)
        home_f1     = np.full(n_sims, -1, dtype=np.int32)  # RFI snapshot (end of inning 1); -1 SENTINEL = not yet written
        away_f1     = np.full(n_sims, -1, dtype=np.int32)
        went_extra  = np.zeros(n_sims, dtype=np.int32)     # 1 = game reached extra innings
        score_dist = _simulate_games_numba(
            home_cum, away_cum,
            home_bull_cum, away_bull_cum,
            emp_post_c, emp_runs_c, emp_cum_c,
            home_exit_lut, away_exit_lut,
            home_ppo, away_ppo,
            pitch_resid_vals, pitch_resid_cum, stochastic_pitches_int,
            h_rcum, h_rrole, h_rstam, h_cidx,
            a_rcum, a_rrole, a_rstam, a_cidx,
            base_mult_arr, base_bucket_arr, rel_hazard_arr,
            h_rthr, a_rthr, home_bat_stand, away_bat_stand,
            home_bat_ibb, away_bat_ibb,
            rng_seeds, gf_mult, h_rclf, a_rclf, clf_w, clf_on,
            1, home_bstats, away_bstats, home_sk, away_sk, home_sbf, away_sbf, home_oc, away_oc,
            home_fed, away_fed, home_eff, away_eff,
            resume_int, start_si64, start_sf64, steal_cfg,
            emp_runner_adv, bat_advz, bat_steal, runner_beta,
            home_f5, away_f5,
            home_f8, away_f8,
            home_so_arr, away_so_arr,
            home_f1, away_f1,
            went_extra,
            n_innings, stop_on_tie_int,
        )
        used_numba = True
    elif track_batter_stats:
        # No numba available → fall back to the numpy reference kernel for tracking.
        score_dist, home_bstats, away_bstats, home_sk, away_sk = _simulate_games_numpy(
            home_cum, away_cum,
            home_bull_cum, away_bull_cum,
            emp_post_c, emp_runs_c, emp_cum_c,
            home_exit_lut, away_exit_lut,
            home_ppo, away_ppo,
            pitch_resid_vals, pitch_resid_cum, stochastic_pitches_int,
            h_rcum, h_rrole, h_rstam, h_cidx,
            a_rcum, a_rrole, a_rstam, a_cidx,
            base_mult_arr, base_bucket_arr, rel_hazard_arr,
            h_rthr, a_rthr, home_bat_stand, away_bat_stand,
            home_bat_ibb, away_bat_ibb,
            rng_seeds, gf_mult, h_rclf, a_rclf, clf_w, clf_on, n_innings, stop_on_tie_int, track=1,
        )
    elif _NUMBA_AVAILABLE and _simulate_games_numba is not None:
        score_dist = _simulate_games_numba(
            home_cum, away_cum,
            home_bull_cum, away_bull_cum,
            emp_post_c, emp_runs_c, emp_cum_c,
            home_exit_lut, away_exit_lut,
            home_ppo, away_ppo,
            pitch_resid_vals, pitch_resid_cum, stochastic_pitches_int,
            h_rcum, h_rrole, h_rstam, h_cidx,
            a_rcum, a_rrole, a_rstam, a_cidx,
            base_mult_arr, base_bucket_arr, rel_hazard_arr,
            h_rthr, a_rthr, home_bat_stand, away_bat_stand,
            home_bat_ibb, away_bat_ibb,
            rng_seeds, gf_mult, h_rclf, a_rclf, clf_w, clf_on,
            0, _dummy_bstats, _dummy_bstats, _dummy_sk, _dummy_sk, _dummy_sk, _dummy_sk, _dummy_oc, _dummy_oc,
            _dummy_fed, _dummy_fed, _dummy_fed, _dummy_fed,
            resume_int, start_si64, start_sf64, steal_cfg,
            emp_runner_adv, bat_advz, bat_steal, runner_beta,
            _dummy_f5, _dummy_f5,
            _dummy_f5, _dummy_f5,
            _dummy_sk, _dummy_sk,          # home_so_a, away_so_a (int32 dummies; track=0 never writes)
            _dummy_f5, _dummy_f5,          # home_f1, away_f1
            _dummy_f5,                     # went_extra
            n_innings, stop_on_tie_int,
        )
        used_numba = True
    else:
        score_dist = _simulate_games_numpy(
            home_cum, away_cum,
            home_bull_cum, away_bull_cum,
            emp_post_c, emp_runs_c, emp_cum_c,
            home_exit_lut, away_exit_lut,
            home_ppo, away_ppo,
            pitch_resid_vals, pitch_resid_cum, stochastic_pitches_int,
            h_rcum, h_rrole, h_rstam, h_cidx,
            a_rcum, a_rrole, a_rstam, a_cidx,
            base_mult_arr, base_bucket_arr, rel_hazard_arr,
            h_rthr, a_rthr, home_bat_stand, away_bat_stand,
            home_bat_ibb, away_bat_ibb,
            rng_seeds, gf_mult, h_rclf, a_rclf, clf_w, clf_on, n_innings, stop_on_tie_int,
        )

    elapsed = time.time() - t0
    p_home_win    = float((score_dist[:, 0] > score_dist[:, 1]).mean())
    markets, ci90 = compute_markets(score_dist, spreads=spreads, totals=totals)
    # F5 snapshot → (n_sims, 2), row-aligned with score_dist. Only the numba track path fills
    # home_f5/away_f5; every other path leaves them None ⇒ f5_score_dist stays None. F8: a resume
    # STARTING past inning 5 never fires the snapshot, so any -1 sentinel remaining ⇒ the F5 score is
    # UNKNOWN (not 0-0) ⇒ return None rather than misleading zeros. Pregame sims fire it for every sim
    # (the bottom of the 5th is always played) ⇒ no -1 remains ⇒ bit-identical to before.
    _f5_ok = (home_f5 is not None and away_f5 is not None
              and home_f5.min() >= 0 and away_f5.min() >= 0)
    f5_score_dist = np.stack([home_f5, away_f5], axis=1) if _f5_ok else None
    _f8_ok = (home_f8 is not None and away_f8 is not None
              and home_f8.min() >= 0 and away_f8.min() >= 0)
    f8_score_dist = np.stack([home_f8, away_f8], axis=1) if _f8_ok else None
    # RFI snapshot → (n_sims, 2) runs at END OF INNING 1. Same -1-sentinel guard as F5/F8:
    # a resume STARTING past inning 1 never fires it ⇒ None rather than misleading zeros.
    _f1_ok = (home_f1 is not None and away_f1 is not None
              and home_f1.min() >= 0 and away_f1.min() >= 0)
    f1_score_dist = np.stack([home_f1, away_f1], axis=1) if _f1_ok else None

    return SimulationResult(
        p_home_win=p_home_win,
        score_dist=score_dist,
        markets=markets,
        ci90=ci90,
        n_sims=n_sims,
        elapsed_sec=elapsed,
        used_numba=used_numba,
        home_batter_stats=home_bstats,
        away_batter_stats=away_bstats,
        home_starter_k=home_sk,
        away_starter_k=away_sk,
        home_starter_bf=home_sbf,
        away_starter_bf=away_sbf,
        home_outcomes=home_oc,
        away_outcomes=away_oc,
        home_fed=home_fed,
        away_fed=away_fed,
        home_eff=home_eff,
        away_eff=away_eff,
        f5_score_dist=f5_score_dist,
        f8_score_dist=f8_score_dist,
        home_starter_outs=home_so_arr,
        away_starter_outs=away_so_arr,
        f1_score_dist=f1_score_dist,
        went_extra=went_extra,
    )


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE LOADER
# ═══════════════════════════════════════════════════════════════

class SimulationEngine:
    """
    Stateful wrapper around simulate_game() that handles:
      - Loading and caching base running tables
      - Warming up numba JIT on first call
      - Building aggregate bullpen pool from roster data

    Usage:
        engine = SimulationEngine(cache_dir="/path/to/mlb_data/data/processed")
        result = engine.run_pregame(
            home_pa_probs, away_pa_probs,
            home_bull_probs, away_bull_probs,
        )
        print(result.summary())
    """

    def __init__(self, cache_dir=None, model_path=None):
        self._cache_dir  = cache_dir
        self._model_path = model_path   # retained for API compatibility (unused by the GAM hook)
        self._tables     = load_baserunning_tables(cache_dir)  # 3-tuple (emp_post, emp_runs, emp_cum)
        # running-game: steal_cfg ([0]=0 ⇒ off ⇒ golden master) + no-steal tables (for the explicit-steal
        # path, so league steals aren't double-counted). Falls back to the standard tables if not built.
        self.steal_cfg   = np.zeros(11, dtype=np.float32)
        try:
            self._tables_nosteal = load_baserunning_tables(cache_dir, suffix="_nosteal")
        except Exception:
            self._tables_nosteal = self._tables
        self._warmed_up  = False
        # Starter-exit removal hazard is the per-start LUT baked by exit_model.bake_start_lut
        # (data/processed/exit_hazard.pkl) inside run_pregame — no per-engine table pre-build.

        if _NUMBA_AVAILABLE:
            print("  SimulationEngine: numba available — will JIT compile on first run.")
        else:
            print("  SimulationEngine: numba not available — using numpy fallback.")

    def warmup(self, n_sims: int = 100) -> None:
        """
        Warm up numba JIT by running a small dummy simulation.
        Call this at server startup to avoid compilation delay on first real game.

        Typical compile time: 5-10 seconds on M5 Pro (longer first time due to
        signature change from threshold → hazard table).
        After warmup, 50k sims run in ~50-150ms.
        """
        if self._warmed_up or not _NUMBA_AVAILABLE:
            return
        print("  Warming up numba JIT...", end=" ", flush=True)
        t0 = time.time()
        dummy_probs = np.tile(LEAGUE_AVG_PA, (9, 1))
        dummy_bull  = LEAGUE_AVG_PA.copy()
        # Use the pre-built league-average hazard table for both pitchers
        self.run_pregame(dummy_probs, dummy_probs, dummy_bull, dummy_bull,
                         n_sims=n_sims)
        self._warmed_up = True
        print(f"done ({time.time() - t0:.1f}s)")

    def run_pregame(
        self,
        home_pa_probs:       np.ndarray,
        away_pa_probs:       np.ndarray,
        home_bull_probs:     np.ndarray,
        away_bull_probs:     np.ndarray,
        n_sims:              int = N_SIMS_DEFAULT,
        seed:                int | None = None,
        home_pitcher_params: dict | None = None,
        away_pitcher_params: dict | None = None,
        home_relievers: tuple | None = None,
        away_relievers: tuple | None = None,
        home_batter_stand: np.ndarray | None = None,
        away_batter_stand: np.ndarray | None = None,
        home_batter_ibb: np.ndarray | None = None,
        away_batter_ibb: np.ndarray | None = None,
        home_batter_advz: np.ndarray | None = None,
        away_batter_advz: np.ndarray | None = None,
        home_batter_steal: np.ndarray | None = None,
        away_batter_steal: np.ndarray | None = None,
        track_batter_stats: bool = False,
        force_numpy_track: bool | None = None,
        start_state: dict | None = None,
    ) -> SimulationResult:
        """
        Full 9-inning simulation for pre-game markets.

        Args:
            home_relievers / away_relievers: (rel_cum, rel_role, rel_stamina, ...)
              activating the individual-reliever bullpen manager (the only bullpen
              model; blended leverage tiers were removed).
            home_pitcher_params / away_pitcher_params: optional dicts with keys
              pitcher_avg_bf, pitcher_avg_pitches, pitcher_k_pct.
              If None, uses league-average hazard table.
            home_relievers / away_relievers: optional (rel_cum, rel_role,
              rel_stamina) → activates the individual-reliever bullpen manager.
            track_batter_stats: True → also return per-batter [PA,H,HR,TB,RBI,R,K]
              by lineup slot (home/away_batter_stats) + per-sim starter K. Runs the
              FAST numba kernel with in-place tracking (pass force_numpy_track=True to
              run the slow numpy reference kernel instead, for validation).
        """
        return simulate_game(
            home_pa_probs, away_pa_probs,
            home_bull_probs, away_bull_probs,
            self._tables_nosteal if self.steal_cfg[0] > 0 else self._tables,
            steal_cfg=self.steal_cfg,
            n_sims=n_sims, n_innings=9, seed=seed,
            home_pitcher_params=home_pitcher_params,
            away_pitcher_params=away_pitcher_params,
            home_relievers=home_relievers,
            away_relievers=away_relievers,
            home_batter_stand=home_batter_stand,
            away_batter_stand=away_batter_stand,
            home_batter_ibb=home_batter_ibb,
            away_batter_ibb=away_batter_ibb,
            home_batter_advz=home_batter_advz,
            away_batter_advz=away_batter_advz,
            home_batter_steal=home_batter_steal,
            away_batter_steal=away_batter_steal,
            model_path=self._model_path,
            track_batter_stats=track_batter_stats,
            force_numpy_track=force_numpy_track,
            start_state=start_state,
        )

    def run_first_five(
        self,
        home_pa_probs:       np.ndarray,
        away_pa_probs:       np.ndarray,
        home_bull_probs:     np.ndarray,
        away_bull_probs:     np.ndarray,
        n_sims:              int = N_SIMS_DEFAULT,
        seed:                int | None = None,
        home_pitcher_params: dict | None = None,
        away_pitcher_params: dict | None = None,
        home_relievers: tuple | None = None,
        away_relievers: tuple | None = None,
        home_batter_stand: np.ndarray | None = None,
        away_batter_stand: np.ndarray | None = None,
        home_batter_ibb: np.ndarray | None = None,
        away_batter_ibb: np.ndarray | None = None,
        home_batter_advz: np.ndarray | None = None,
        away_batter_advz: np.ndarray | None = None,
        home_batter_steal: np.ndarray | None = None,
        away_batter_steal: np.ndarray | None = None,
    ) -> SimulationResult:
        """First-5-innings simulation for F5 markets.

        Uses stop_on_tie=True so the simulation ends after exactly 5 innings
        regardless of score — tied games remain tied in the score distribution,
        giving a meaningful 'tie' market probability (~10-20% in practice).
        """
        return simulate_game(
            home_pa_probs, away_pa_probs,
            home_bull_probs, away_bull_probs,
            self._tables_nosteal if self.steal_cfg[0] > 0 else self._tables,
            steal_cfg=self.steal_cfg,
            n_sims=n_sims, n_innings=5, seed=seed, stop_on_tie=True,
            home_pitcher_params=home_pitcher_params,
            away_pitcher_params=away_pitcher_params,
            home_relievers=home_relievers,
            away_relievers=away_relievers,
            home_batter_stand=home_batter_stand,
            away_batter_stand=away_batter_stand,
            home_batter_ibb=home_batter_ibb,
            away_batter_ibb=away_batter_ibb,
            home_batter_advz=home_batter_advz,
            away_batter_advz=away_batter_advz,
            home_batter_steal=home_batter_steal,
            away_batter_steal=away_batter_steal,
            model_path=self._model_path,
        )

    def run_dynamic_baseline(
        self,
        home_pa_probs:   np.ndarray,
        away_pa_probs:   np.ndarray,
        home_bull_probs: np.ndarray,
        away_bull_probs: np.ndarray,
        remaining_innings: int,
        seed: int | None = None,
        home_pitcher_params: dict | None = None,
        away_pitcher_params: dict | None = None,
    ) -> SimulationResult:
        """
        Fast in-game dynamic baseline simulation.

        Uses N_SIMS_DYNAMIC (5k) instead of 50k for speed — a fast rest-of-game
        run-expectancy estimate from the current base-out / inning state.
        With numba: ~10-20ms. Without numba: ~200-500ms (still acceptable).
        """
        return simulate_game(
            home_pa_probs, away_pa_probs,
            home_bull_probs, away_bull_probs,
            self._tables_nosteal if self.steal_cfg[0] > 0 else self._tables,
            steal_cfg=self.steal_cfg,
            n_sims=N_SIMS_DYNAMIC, n_innings=remaining_innings, seed=seed,
            home_pitcher_params=home_pitcher_params,
            away_pitcher_params=away_pitcher_params,
            model_path=self._model_path,
        )

    @staticmethod
    def build_bullpen_aggregate(
        bullpen_pa_probs: list[np.ndarray],
        usage_weights:    Optional[list[float]] = None,
    ) -> np.ndarray:
        """
        Build aggregate bullpen PA probability vector from individual reliever probs.

        Each reliever's probs are weighted by their historical usage share (IP-weighted).
        If usage_weights is None, assumes equal weighting.

        Args:
            bullpen_pa_probs: list of (N_OUTCOMES,) float32 arrays, one per reliever
            usage_weights:    list of usage shares (unnormalised), same length

        Returns:
            aggregate: (N_OUTCOMES,) float32 summing to 1
        """
        if not bullpen_pa_probs:
            # Fallback: league-average bullpen
            return LEAGUE_AVG_PA.copy()
        n = len(bullpen_pa_probs)
        if usage_weights is None:
            weights = np.ones(n, dtype=np.float32)
        else:
            weights = np.array(usage_weights, dtype=np.float32)
        weights /= weights.sum()
        agg = sum(w * p for w, p in zip(weights, bullpen_pa_probs))
        agg = np.asarray(agg, dtype=np.float32)
        agg /= agg.sum()
        return agg


# ═══════════════════════════════════════════════════════════════
# SMOKE TEST  (python game_simulation.py)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  GAME SIMULATION ENGINE — SMOKE TEST")
    print("=" * 60)

    # ── Test 1: Base running tables ──────────────────────────
    print("\n[1] Building base running tables...")
    tables = build_baserunning_tables()
    new_state_t, runs_t, stoc_ns_t, stoc_r_t, stoc_p_t = tables
    assert new_state_t.shape == (24, 9), f"Bad shape: {new_state_t.shape}"
    assert runs_t.shape == (24, 9)
    assert stoc_p_t.shape == (24, 9)
    assert stoc_p_t.dtype == np.float32

    # Sanity: HR from any state should clear the bases
    for s in range(24):
        outs = s >> 3
        b1   = (s & 1); b2 = (s & 2) >> 1; b3 = (s & 4) >> 2
        hr_new   = new_state_t[s, OUTCOME_TO_IDX["HR"]]
        hr_runs  = runs_t[s, OUTCOME_TO_IDX["HR"]]
        expected_runs = b1 + b2 + b3 + 1
        expected_new  = 0 + outs * 8   # bases clear, same outs
        assert hr_new == expected_new, f"HR state wrong: state={s}, got {hr_new}, expected {expected_new}"
        assert hr_runs == expected_runs, f"HR runs wrong: state={s}, got {hr_runs}, expected {expected_runs}"
    print("  ✓ HR transitions correct for all 24 states")

    # Sanity: strikeout adds one out, no runners advance
    for s in range(24):
        outs = s >> 3
        k_new = new_state_t[s, OUTCOME_TO_IDX["K"]]
        if outs == 2:
            assert k_new == 24, f"K with 2 outs should end inning: state={s}"
        else:
            assert (k_new >> 3) == outs + 1, f"K should add one out: state={s}"
    print("  ✓ Strikeout transitions correct for all 24 states")

    # ── Test 2: Engine initialisation ────────────────────────
    print("\n[2] Initialising simulation engine...")
    engine = SimulationEngine()

    # ── Test 3: League-average game ──────────────────────────
    print("\n[3] Simulating league-average game (10k sims)...")
    lineup_probs = np.tile(LEAGUE_AVG_PA, (9, 1))
    bull_probs   = LEAGUE_AVG_PA.copy()

    result = engine.run_pregame(
        lineup_probs, lineup_probs,
        bull_probs, bull_probs,
        n_sims=10_000,
        seed=123,
    )
    print(f"  {result.summary()}")

    # Sanity checks
    assert 0.45 < result.p_home_win < 0.55, \
        f"League-avg game should be ~50/50: {result.p_home_win:.3f}"
    mean_total = (result.score_dist[:, 0] + result.score_dist[:, 1]).mean()
    assert 7.0 < mean_total < 11.0, \
        f"League-avg total should be ~8-10 runs: {mean_total:.2f}"
    print(f"  ✓ P(home win) = {result.p_home_win:.3f}  (expect ~0.50)")
    print(f"  ✓ Mean total  = {mean_total:.2f}  (expect ~8-10)")
    print(f"  ✓ Markets computed: {len(result.markets)} market lines")

    # ── Test 4: High-K pitcher effect ────────────────────────
    print("\n[4] Testing pitcher effect (elite K pitcher vs avg)...")
    # Elite pitcher: high K, low walks and HR
    elite_pitcher_away_batter = np.array(
        [0.350, 0.060, 0.008, 0.200, 0.220, 0.110, 0.030, 0.005, 0.017],
        dtype=np.float32
    )
    elite_lineup = np.tile(elite_pitcher_away_batter, (9, 1))

    result_elite = engine.run_pregame(
        lineup_probs,  # home lineup vs league-avg away pitcher
        elite_lineup,  # away lineup vs elite home pitcher (fewer runs for away)
        bull_probs, bull_probs,
        n_sims=10_000,
        seed=456,
    )
    print(f"  {result_elite.summary()}")
    assert result_elite.p_home_win > 0.55, \
        f"Elite home pitcher should give home team edge: {result_elite.p_home_win:.3f}"
    print(f"  ✓ Elite pitcher: P(home win) = {result_elite.p_home_win:.3f}  (expect >0.55)")

    # ── Test 5: First-five markets ────────────────────────────
    print("\n[5] Testing first-five inning simulation...")
    result_f5 = engine.run_first_five(
        lineup_probs, lineup_probs,
        bull_probs, bull_probs,
        n_sims=10_000,
        seed=789,
    )
    mean_f5_total = (result_f5.score_dist[:, 0] + result_f5.score_dist[:, 1]).mean()
    print(f"  Mean F5 total: {mean_f5_total:.2f}  (expect ~4.5-5.5)")
    assert 3.5 < mean_f5_total < 6.5, f"F5 total out of range: {mean_f5_total:.2f}"
    print(f"  ✓ First-five total = {mean_f5_total:.2f}")

    # ── Test 6: Dynamic baseline speed ───────────────────────
    print("\n[6] Dynamic baseline speed test (5k sims, 4 remaining innings)...")
    t0 = time.time()
    result_dyn = engine.run_dynamic_baseline(
        lineup_probs, lineup_probs,
        bull_probs, bull_probs,
        remaining_innings=4,
        seed=999,
    )
    elapsed = time.time() - t0
    print(f"  5k sims in {elapsed*1000:.1f}ms  (target: <500ms without numba, <50ms with)")
    if result_dyn.used_numba:
        assert elapsed < 0.5, f"Numba should be <500ms: {elapsed:.2f}s"
    print(f"  ✓ Dynamic baseline: {result_dyn.summary()}")

    # ── Test 7: Bullpen aggregate ─────────────────────────────
    print("\n[7] Testing bullpen aggregate builder...")
    reliever1 = np.array([0.26, 0.08, 0.01, 0.20, 0.22, 0.14, 0.04, 0.01, 0.04], dtype=np.float32)
    reliever2 = np.array([0.20, 0.09, 0.01, 0.23, 0.24, 0.14, 0.04, 0.01, 0.04], dtype=np.float32)
    agg = SimulationEngine.build_bullpen_aggregate(
        [reliever1, reliever2],
        usage_weights=[60, 40],   # 60% reliever1, 40% reliever2
    )
    assert abs(agg.sum() - 1.0) < 1e-5, f"Bullpen aggregate should sum to 1: {agg.sum()}"
    print(f"  ✓ Bullpen aggregate sums to {agg.sum():.6f}")


    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED ✓")
    print(f"  numba: {'available ✓' if _NUMBA_AVAILABLE else 'not installed (install for 20-40x speedup)'}")
    print("=" * 60)
