"""
build_bullpen_profiles.py
=========================
Phase-1 bullpen model: a team-specific, availability-aware reliever blend that
replaces the flat LEAGUE_AVG_PA vector the simulation uses after the starter is
removed.

DESIGN (consistent with the rest of the model, by construction)
---------------------------------------------------------------
A reliever is just a pitcher. We therefore reuse Level 3's existing pitcher-rate
machinery via Level3Predictor.precompute_game():
  • qualified relievers  → posterior a_hat_p / o_p
  • non-qualified arms   → per-outcome-prior-adjusted league rates
…log5-combined against the actual opposing lineup, park-adjusted, and γ-scaled —
exactly like the starter path. No separate/bolted-on rate model.

The single bullpen PA vector for a team = a blend of its available relievers'
matchup vectors, weighted by recent usage × an availability factor.

AVAILABILITY (grounded in the fatigue literature)
--------------------------------------------------
Relievers lose ~0.5 mph on back-to-back days and ~1.5 mph on three straight, and
managers now rarely use arms on zero rest (~16% of appearances). We down-weight
(not hard-zero) recently/heavily-used arms so the blend leans on the rested,
likely-available relievers — which is exactly what a manager's choice reflects.
(Effectiveness *degradation* of tired arms is a phase-2 refinement; phase 1 only
re-weights availability.)

OFFLINE
-------
  python build_bullpen_profiles.py
    → builds data/processed/reliever_appearances.parquet from pa_level_dataset
      + pitcher_start_stats (a relief appearance = any appearance that is not a
      recorded start).

RUNTIME (used by pregame_pipeline)
----------------------------------
  app = load_appearances(proc_dir)
  pen = team_bullpen(app, team_code, game_date)          # current depth chart
  vec = bullpen_pa_vector(predictor, pen, opposing_lineup, park_effects, gamma)
        → (9,) PA-outcome vector (OUTCOMES order) or None to fall back to league avg
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from game_simulation import build_pa_prob_arrays, LEAGUE_AVG_PA, N_OUTCOMES

# Tunables
WINDOW_DAYS   = 45     # look-back window. Widened 30→45 so the SMOOTH recency decay
                       # below (not a hard cliff) governs which arms count — an
                       # appearance ~30 days ago now fades gradually instead of
                       # vanishing abruptly at the boundary (a role change near the
                       # old edge no longer disappears overnight).
USAGE_HALFLIFE_DAYS = 14.0  # usage_weight = Σ bf · 0.5**(days_ago/14): a 2-week half-
                       # life so the depth-chart blend leans on a reliever's CURRENT
                       # role/workload, matching how fast bullpen roles actually churn.
MIN_REL_BF    = 2      # ignore arms with <2 relief BF in the window (spot/position players)
MAX_RELIEVERS = int(os.environ.get("BULLPEN_MAX_RELIEVERS", "12"))
                       # cap the depth chart. Was 10. MEASURED (2025, 45d team window): the real
                       # available-arm count is mean 13.8 / median 13 — a manager churns through
                       # ~13 arms over 45 days, not 8. The old 10-cap truncated the depth-chart
                       # UNIVERSE before the kernel even saw it; combined with the old 8-arm kernel
                       # cap the sim deployed only its highest-QUALITY (≈highest-K) arms, so the
                       # PA-weighted deployed reliever K-talent ran +7.6% over real managers'
                       # (0.285 vs 0.264). Raising to 12 captures ~85% of available arms and lets
                       # the lower-K depth arms be deployed at their true rate. Must stay ≥
                       # MAX_REL_KERNEL so team_bullpen never starves the kernel's choice set.
                       # Env BULLPEN_MAX_RELIEVERS overrides for A/B + the old-behavior golden master.


# ═══════════════════════════════════════════════════════════════
# OFFLINE — build the relief-appearance log
# ═══════════════════════════════════════════════════════════════

def build_appearances(proc_dir: str) -> str:
    """
    Build a clean relief-appearance log and write it to
    {proc_dir}/reliever_appearances.parquet.

    Columns: pitcher, team, game_date, pitches, bf, inning_min, throws
    A relief appearance = a (game_pk, pitcher) that pitched but is NOT a start.
    """
    pa = pd.read_parquet(
        os.path.join(proc_dir, "pa_level_dataset.parquet"),
        columns=["game_pk", "game_date", "pitcher", "fielding_team",
                 "p_throws", "pitches", "inning"],
    )
    starts = pd.read_parquet(
        os.path.join(proc_dir, "pitcher_start_stats.parquet"),
        columns=["game_pk", "pitcher"],
    ).drop_duplicates()
    starts["_is_start"] = 1

    merged = pa.merge(starts, on=["game_pk", "pitcher"], how="left")
    relief = merged[merged["_is_start"].isna()].copy()

    app = (relief
           .groupby(["game_pk", "pitcher", "fielding_team", "game_date"], as_index=False)
           .agg(pitches=("pitches", "sum"),
                bf=("pitcher", "size"),
                inning_min=("inning", "min"),
                throws=("p_throws", "first")))
    app = app.rename(columns={"fielding_team": "team"})
    # Backstop against starts missing from pitcher_start_stats. A true relief
    # outing: enters after the 1st inning AND has a relief-sized workload.
    # (Measured: 99th-pctile relief outing = 16 BF; a 94-pitch / 23-BF outing is
    # a leaked start, not relief.)
    app = app[(app["inning_min"] >= 2)
              & (app["bf"] <= 14)
              & (app["pitches"].fillna(0) <= 50)].copy()
    app["pitcher"] = app["pitcher"].astype(int)
    app["game_date"] = pd.to_datetime(app["game_date"])

    out = os.path.join(proc_dir, "reliever_appearances.parquet")
    app[["pitcher", "team", "game_date", "pitches", "bf", "inning_min", "throws"]].to_parquet(out, index=False)
    print(f"  wrote {len(app):,} relief appearances "
          f"({app['pitcher'].nunique()} pitchers, {app['team'].nunique()} teams) -> {out}")
    return out


# ═══════════════════════════════════════════════════════════════
# RUNTIME — load + assemble the current bullpen
# ═══════════════════════════════════════════════════════════════

def load_appearances(proc_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(proc_dir, "reliever_appearances.parquet")
    if not os.path.exists(path):
        return None
    try:
        app = pd.read_parquet(path)
        app["game_date"] = pd.to_datetime(app["game_date"])
        return app
    except Exception:
        return None


def _availability_weight(days_ago: list[int], pitches_by_day: dict[int, float]) -> float:
    """
    Availability factor in [0, 1] from recent usage.

    days_ago: list of integer days since each recent appearance (1 = yesterday).
    pitches_by_day: {days_ago: pitches thrown that day}.

    EMPIRICAL deploy probabilities, RELATIVE to the rested baseline (rested=1.0),
    measured by fit_bullpen_empirical.py over 2019-2026 as P(arm appears today |
    rest bucket) / P(appears | rested).  Denominator = team games on which the arm
    was active (pitched within 6 days), so IL/option gaps don't dilute it.
    These replace the prior hand-set guesses (shown in []), which were uniformly
    too generous — real managers sit tired arms far more than I assumed:
      • 3 straight days       -> 0.043   [was 0.10]  (P_abs 2.1%)
      • would-be 3rd straight -> 0.138   [was 0.25]  (P_abs 6.7%)
      • yesterday, >30 pitch  -> 0.065   [was 0.45]  (P_abs 3.2% — a heavy outing
                                          nearly rules the arm out the next day)
      • yesterday only        -> 0.545   [was 0.80]  (P_abs 26.3%)
      • >=2 days rest         -> 1.000               (P_abs 48.3%, the normalizer)
    """
    if 0 in days_ago:
        # SAME-DAY earlier outing (doubleheader game 1, via the same-night statsapi supplement —
        # 2026-07-18): measured second-same-day deploy prob (fit_bullpen_empirical "same_day").
        # Older artifacts lack the key → fall back to the most restrictive MEASURED bucket.
        return _AVAIL_CUR.get("same_day", _AVAIL_CUR["3straight"])
    y1 = 1 in days_ago
    y2 = 2 in days_ago
    y3 = 3 in days_ago
    if y1 and y2 and y3:
        b = "3straight"
    elif y1 and y2:
        b = "b2b"
    elif y1 and pitches_by_day.get(1, 0) > 30:
        b = "yest_30p"
    elif y1:
        b = "yest"
    else:
        b = "rested"
    return _AVAIL_CUR[b]      # mean, or a draw from its empirical sampling dist


def team_bullpen(app: pd.DataFrame, team: str, game_date,
                 window_days: int = WINDOW_DAYS, fp_ts=None, supplement=None,
                 cap_arms: bool = True) -> pd.DataFrame:
    """
    Current bullpen depth chart for `team` as of `game_date`.

    Returns a DataFrame: pitcher, throws, bf, n_app, usage_weight, avail, weight
    where `weight` = normalized usage × availability (the blend weight).
    Empty DataFrame if no recent relief appearances are found.
    """
    gd = pd.Timestamp(game_date).normalize()
    lo = gd - pd.Timedelta(days=window_days)
    w = app[(app.team == team) & (app.game_date >= lo) & (app.game_date < gd)]
    # ── SAME-NIGHT SUPPLEMENT (2026-07-18, availability-lag fix): statsapi-sourced appearances
    # covering the statcast publication lag (1-2 days) AND same-day earlier games (doubleheader
    # game 1 — visible via fp_ts: only appearances from games that STARTED ≥2h before THIS game).
    # Dedup: the statcast log wins on any (pitcher, date) it already covers. throws is inherited
    # from the log (an arm with no log history wouldn't clear MIN_REL_BF anyway → dropped). ──
    if supplement is not None and len(supplement):
        sup = supplement[(supplement.team == team) & (supplement.game_date >= lo)
                         & (supplement.game_date <= gd)].copy()
        if len(sup):
            same_day = sup.game_date == gd
            if fp_ts is not None:
                sup = sup[~same_day | (sup.game_ts <= float(fp_ts) - 7200)]
            else:
                sup = sup[~same_day]                       # no first-pitch ts → same-day rows unusable
            have = set(zip(w.pitcher.astype(int), w.game_date))
            sup = sup[[(int(p), d) not in have for p, d in zip(sup.pitcher, sup.game_date)]]
            if len(sup):
                thr = app[app.team == team].groupby("pitcher").throws.last()
                sup = sup[sup.pitcher.isin(thr.index)]
                if len(sup):
                    sup = sup.assign(throws=sup.pitcher.map(thr),
                                     inning_min=np.nan)    # entry inning unknown → excluded from role means
                    w = pd.concat([w, sup[["pitcher", "team", "game_date", "pitches",
                                           "bf", "inning_min", "throws"]]], ignore_index=True)
    if w.empty:
        return w.assign(usage_weight=[], avail=[], weight=[])

    rows = []
    for pid, g in w.groupby("pitcher"):
        bf = int(g.bf.sum())
        if bf < MIN_REL_BF:
            continue
        days_ago = [int((gd - d).days) for d in g.game_date]
        pitches_by_day = (g.assign(da=[(gd - d).days for d in g.game_date])
                            .groupby("da").pitches.sum().to_dict())
        avail = _availability_weight(days_ago, pitches_by_day)
        # Recency-decayed usage: weight each appearance's BF by 0.5**(days_ago/HL) so
        # the depth-chart blend reflects the reliever's CURRENT role, not a flat
        # 45-day count. (Raw `bf` is kept for the MIN_REL_BF gate + stamina inference.)
        _dec = np.power(0.5, np.asarray(days_ago, float) / USAGE_HALFLIFE_DAYS)
        usage_weight = float((g.bf.to_numpy(float) * _dec).sum())
        rows.append({
            "pitcher": int(pid),
            "throws": str(g.throws.iloc[-1]),
            "bf": bf,
            "n_app": int(g.game_pk.nunique()) if "game_pk" in g else len(g),
            "usage_weight": usage_weight,
            "avail": float(avail),
            # ── role/stamina inference inputs ──
            # stamina = typical batters this arm covers in an outing (≥3 for the
            # 3-batter minimum; capped at 8 for long men). avg_entry = the inning
            # he usually enters; late_entries = # of 9th-inning-or-later entries (a
            # data-driven save/closer signal — the real closer enters the 9th most).
            "stamina":     int(np.clip(round(float(g.bf.mean())), 3, 8)),
            "avg_entry":   float(g.inning_min.mean()),
            "late_entries": int((g.inning_min >= 9).sum()),
        })
    pen = pd.DataFrame(rows)
    if pen.empty:
        return pen
    pen = pen.sort_values("bf", ascending=False)
    # cap_arms=False returns the UNCAPPED universe (still bf-sorted) so a caller that filters for
    # eligibility can truncate AFTERWARDS — "the top-N ELIGIBLE arms" rather than "the eligible
    # members of the top-N". Capping here first and filtering later silently re-truncates the
    # depth-chart universe, which is the exact failure the 10→12 raise above was made to fix:
    # measured 2026-08-07, the live roster filter left a mean of 7.2 arms (min 4) — below the 8 that
    # drove deployed reliever K-talent +7.6% over real managers. Filtering first restores 8.8 (min 7).
    if cap_arms:
        pen = pen.head(MAX_RELIEVERS)
    pen = pen.reset_index(drop=True)
    raw = pen.usage_weight * pen.avail
    pen["weight"] = (raw / raw.sum()) if raw.sum() > 0 else (1.0 / len(pen))
    return pen


def _handed(bats: str, throws: str) -> int:
    """Same as pregame_pipeline._batter_handedness (avoid a circular import)."""
    if bats == "S":
        return 0
    return 1 if bats == throws else 0


# Linear weights (≈ runs above average per PA) for ranking reliever quality.
# OUTCOMES order: K, BB, HBP, GO, FO, 1B, 2B, 3B, HR.
_LINEAR_WEIGHTS = np.array(
    [-0.27, 0.33, 0.35, -0.27, -0.27, 0.47, 0.78, 1.09, 1.40], dtype=np.float64
)
N_HIGH_LEV = 3   # number of top arms that form the high-leverage blend


def _reliever_vectors(predictor, pen: pd.DataFrame, opposing_lineup: list[dict],
                      park_effects, gamma: float, month: int = None) -> list[dict]:
    """
    Per-reliever matchup vectors vs the opposing lineup.

    For each available reliever: precompute_game (same log5 + park + γ as the
    starter), averaged over the 9 batters, with handedness recomputed per
    reliever. Returns a list of {pitcher, weight, avail, vec, quality} where
    `quality` = −E[runs/PA] (higher = better run suppression).
    """
    out = []
    # Role-aware season recal (#45): these are RELIEVER matchups — swap the predictor's active
    # recal weights to w_rp for the precompute loop (relievers' K is already calibrated; the
    # starter-fit global w over-penalizes them), then restore. The per-reliever try/except below
    # swallows errors, so the loop never propagates and the restore at the end always runs.
    _saved_w = getattr(predictor, "_season_w", None)
    _rp_w = getattr(predictor, "_season_w_rp", None)
    if _rp_w is not None:
        predictor._season_w = _rp_w
    for _, r in pen.iterrows():
        pid, throws = int(r["pitcher"]), str(r["throws"])
        w, avail = float(r["weight"]), float(r.get("avail", 1.0))
        if w <= 0:
            continue
        lineup_r = [
            {"batter_id": b["batter_id"],
             "handedness": _handed(b.get("bats", "R"), throws),
             "batting_order": b.get("batting_order", 0)}
            for b in opposing_lineup
        ]
        try:
            gp = predictor.precompute_game(pid, lineup_r,
                                           park_effects=park_effects, gamma=gamma, month=month)
            arr = build_pa_prob_arrays(lineup_r, gp)      # (n_batters, 9)
        except Exception:
            continue
        # Per-slot matchup matrix (each batter faces this reliever with the correct
        # handedness split — captures the 20-40 wOBA-point platoon advantage and
        # batter-quality variance the lineup average erases). Pad/truncate to 9.
        slots = np.asarray(arr, dtype=np.float64)
        if slots.shape[0] >= 9:
            slots = slots[:9]
        else:
            pad = np.tile(slots.mean(axis=0, keepdims=True), (9 - slots.shape[0], 1))
            slots = np.vstack([slots, pad])
        vec = slots.mean(axis=0)
        out.append({"pitcher": pid, "throws": throws, "weight": w, "avail": avail,
                    "vec": vec, "vec_slots": slots,
                    "quality": -float(vec @ _LINEAR_WEIGHTS)})
    predictor._season_w = _saved_w                        # restore starter recal (#45)
    return out


def _blend(vecs: list[np.ndarray], weights: list[float]) -> Optional[np.ndarray]:
    if not vecs:
        return None
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(vecs))
    acc = sum(wi * v for wi, v in zip(w, vecs))
    vec = np.asarray(acc, dtype=np.float64)
    s = vec.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return (vec / s).astype(np.float32)


def bullpen_pa_vector(predictor, pen: pd.DataFrame, opposing_lineup: list[dict],
                      park_effects, gamma: float, month: int = None) -> Optional[np.ndarray]:
    """General (usage×availability-weighted) bullpen vector. None -> league-avg fallback."""
    rv = _reliever_vectors(predictor, pen, opposing_lineup, park_effects, gamma, month=month)
    if not rv:
        return None
    return _blend([r["vec"] for r in rv], [r["weight"] for r in rv])


def bullpen_tiered_vectors(predictor, pen: pd.DataFrame, opposing_lineup: list[dict],
                           park_effects, gamma: float,
                           n_high: int = N_HIGH_LEV, n_low: int = N_HIGH_LEV,
                           month: int = None
                           ) -> tuple[Optional[np.ndarray], Optional[np.ndarray],
                                      Optional[np.ndarray]]:
    """
    Leverage-tiered bullpen vectors: (general, high_leverage, low_leverage).

    general       = usage×availability-weighted blend of the whole pen (phase 1).
    high_leverage = blend of the best  n_high arms by quality (closer/setup) —
                    deployed late & close.
    low_leverage  = blend of the worst n_low  arms by quality (long/mop-up) —
                    deployed in blowouts (the run-total tail).

    Any element may be None (empty pen). high/low fall back to general when the
    pen is too small to form a distinct tier.
    """
    rv = _reliever_vectors(predictor, pen, opposing_lineup, park_effects, gamma, month=month)
    if not rv:
        return None, None, None
    general = _blend([r["vec"] for r in rv], [r["weight"] for r in rv])
    ranked = sorted(rv, key=lambda d: d["quality"], reverse=True)
    top = ranked[:max(1, n_high)]
    bot = ranked[-max(1, n_low):]
    high = _blend([r["vec"] for r in top], [max(r["avail"], 1e-3) for r in top])
    # mop-up arms are typically always available (they're the unused ones)
    low  = _blend([r["vec"] for r in bot], [max(r["avail"], 1e-3) for r in bot])
    return general, (high if high is not None else general), (low if low is not None else general)


# ═══════════════════════════════════════════════════════════════
# STATE-OF-THE-ART: individual sequenced relievers for the kernel manager
# ═══════════════════════════════════════════════════════════════

MAX_REL_KERNEL   = int(os.environ.get("BULLPEN_MAX_REL_KERNEL", "12"))
                          # depth-chart size passed to the sim manager. Was 8. The kernel deployed
                          # only its top-8 arms by QUALITY out of a real ~13-arm pen, concentrating
                          # PAs into its highest-K arms (deployed reliever K-talent +7.6% over real,
                          # 0.285 vs 0.264 — the dominant source of the bullpen-K over-prediction).
                          # Raised to 12 to match real bullpen DEPTH (MEASURED: ~13 available arms
                          # in a 45d team window). The learned reliever-choice model is RE-FIT at
                          # this same cap (build_reliever_choice_model.py imports MAX_REL_KERNEL),
                          # so its conditional-logit softmax re-calibrates the deployed K-talent onto
                          # the realistic 12-arm choice set by construction (the MLE first-order
                          # condition balances K when the deployment universe == the training
                          # universe). MUST equal the cap the clf was trained under — changing one
                          # without re-fitting the other re-introduces the gap.
MIN_DEPLOY_AVAIL = 0.10   # arms below this (gassed) are scratched from the pen.

# ── POSITION-PLAYER ARM (2026-07-26) ─────────────────────────────────────────────────────────
# MIN_REL_BF above deliberately drops "spot/position players" from every modeled pen. Real managers
# DO use them once a game is conceded, and they are ~48% worse per PA than a real pitcher
# (build_position_player_arm.py: K .030 vs .228, HR .061 vs .032), so excluding them leaves the sim's
# run-total distribution with too thin a RIGHT TAIL. Measured consequence: game_total at lines 11+
# under-claims by +0.128 in blowouts ([[project_game_total_blowout_tail]]).
#
# We append ONE league-average position-player arm to every pen rather than hand-writing a "blowout
# state": the LEARNED reliever-choice model already carries the score-state features that decide when
# a manager reaches for one (q_x_deficit = -0.536 = conserve good arms when behind). Giving it the
# OPTION is principled; legislating WHEN would be a band-aid.
#
# ⚠ It is appended AFTER the quality cap on purpose — it is the worst arm by construction, so a
# quality sort + truncate would always delete it, which is the very bug we are fixing.
# ⚠ TRAIN/SERVE: adding an arm changes the conditional-logit choice set. build_reliever_choice_model
# MUST be re-fit with this arm present (same rule that forced a re-fit for the 8->12 cap change).
POS_ARM_ON = os.environ.get("BULLPEN_POS_PLAYER_ARM", "0") != "0"   # default OFF ⇒ byte-identical
POS_ARM_PID = -99

# The kernel MIRRORS the 5.10(g) thresholds (njit bakes globals in at compile time, and
# game_simulation must not import this builder). Fail loudly if the two ever drift apart —
# a silent mismatch would mean the model trains under one rule and serves under another.
def _assert_pp_rule_mirrored():
    import build_position_player_arm as _bpa, game_simulation as _gs
    if (_bpa.PP_LOSING_BY, _bpa.PP_WINNING_BY) != (_gs.PP_LOSING_BY, _gs.PP_WINNING_BY):
        raise RuntimeError(
            f"MLB 5.10(g) thresholds DRIFTED: builder=({_bpa.PP_LOSING_BY},{_bpa.PP_WINNING_BY}) "
            f"kernel=({_gs.PP_LOSING_BY},{_gs.PP_WINNING_BY}). Fix before serving.")
                                                   # sentinel, cannot collide
_POS_ARM_CACHE: dict | None = None


def load_position_player_arm() -> dict | None:
    """League position-player PA vector from the nightly artifact. None if absent/disabled."""
    global _POS_ARM_CACHE
    if not POS_ARM_ON:
        return None
    if _POS_ARM_CACHE is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "processed", "position_player_arm.json")
        try:
            import json as _json
            _POS_ARM_CACHE = _json.load(open(p))
        except Exception:
            _POS_ARM_CACHE = {}
    return _POS_ARM_CACHE or None


def append_position_player_arm(usable: list) -> list:
    """Append the league position-player arm to a (already capped) pen. No-op when disabled."""
    art = load_position_player_arm()
    if not art or not usable:
        return usable
    import numpy as _np
    vec = _np.asarray(art["vec"], dtype=float)
    if vec.shape[0] != 9 or not _np.isfinite(vec).all():
        return usable
    vec = vec / max(float(vec.sum()), 1e-12)
    tmpl = usable[0]
    slots = _np.vstack([vec] * 9)                     # no measured platoon split ⇒ same every slot
    arm = dict(tmpl)                                  # inherit the row shape the kernel expects
    arm.update({
        "pitcher": POS_ARM_PID,
        "vec_slots": _np.asarray(slots, dtype=_np.asarray(tmpl["vec_slots"]).dtype),
        "vec": vec,
        "quality": -float(vec @ _LINEAR_WEIGHTS),
        "krate": float(vec[0]),                       # K is OUTCOMES[0]
        "avail": 1.0,                                 # a bench bat is ALWAYS available
        "weight": 0.0,                                # never used in the usage-weighted blend
        "stamina": 1,
        "throws": "R",
        "is_position_player": True,
    })
    return list(usable) + [arm]
                          # Lowered 0.20→0.10 for the new empirical avail scale:
                          # scratches the truly-unavailable (3-straight ≈0.043,
                          # 30+pitch-yesterday ≈0.065) while keeping a single
                          # back-to-back (≈0.138/0.545) deployable-but-degraded.

# Role codes used by the kernel manager.
ROLE_CLOSER, ROLE_SETUP, ROLE_MIDDLE, ROLE_LONG = 0, 1, 2, 3

# ─────────────────────────────────────────────────────────────────
# Learned reliever-selection model (conditional logit) — built by
# build_reliever_choice_model.py to mirror real-manager deployment.
# Loaded lazily; absent → None (kernel falls back to the deterministic
# _pick_reliever, golden-master safe). Weights are β/sd applied to RAW
# features (the standardization constant cancels in the softmax, so μ
# is not needed). _CLF_INT_ORDER MUST match _clf_choose in game_simulation.py.
# ─────────────────────────────────────────────────────────────────
# BULLPEN_CLF_NAME selects a clf artifact BY BASENAME, resolved against THIS module's directory.
# BULLPEN_CLF_PATH is absolute and therefore cannot address a walk-forward run, where every snapshot
# must load its OWN leak-truncated artifact. The basename form lets one env var select the A/B arm
# across all snapshots without destructively swapping files mid-run.
_CLF_MODEL_PATH = os.environ.get("BULLPEN_CLF_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/processed",
    os.environ.get("BULLPEN_CLF_NAME", "reliever_choice_model.json"))
_CLF_MAIN_ORDER = ["quality", "late_entries", "avail", "usage_weight", "stamina",
                   "is_closer", "is_setup", "is_long",
                   "is_pos"]      # alternative-specific constant (0 for every real arm)
_CLF_INT_ORDER  = ["inn_prox", "closer_x_save", "closer_x_late9",
                   "q_x_li_protect", "q_x_li_tied", "q_x_li_trail",
                   "q_x_protect", "q_x_deficit", "q_x_blowout", "stamina_x_blowout",
                   "hand_match", "hand_x_closelate", "le_x_late9", "krate_x_jam",
                   "is_pos_x_extras",   # managers do NOT use a position player in tied extras
                   "is_pos_x_protect", "is_pos_x_deficit"]   # ASC gradient (leader/trailer)
# per-club random-effect axes; ORDER must match build_manager_effects.DEV_FEATS and the kernel's
# rel_clf columns 7,8,9.
_CLF_DEV_ORDER  = ["q_x_li_protect", "q_x_deficit", "closer_x_save"]
_CLF = None   # None=not loaded, False=absent, dict=loaded


def _load_clf():
    global _CLF
    if _CLF is None:
        if not os.path.exists(_CLF_MODEL_PATH):
            _CLF = False
        else:
            with open(_CLF_MODEL_PATH) as f:
                m = json.load(f)
            # The conditional-logit softmax is calibrated ONLY on the choice-set size it was
            # FIT under (max_arms). Deploying it on a DIFFERENT-size pen extrapolates and
            # re-introduces the K over-deployment. Warn loudly on a cap mismatch so a stale clf
            # (e.g. an old cap-8 model left in place after MAX_REL_KERNEL was raised) is visible.
            _trained_cap = int(m.get("max_arms", MAX_REL_KERNEL))
            if _trained_cap != MAX_REL_KERNEL:
                import warnings
                warnings.warn(
                    f"reliever_choice_model.json was fit at max_arms={_trained_cap} but the "
                    f"kernel deploys MAX_REL_KERNEL={MAX_REL_KERNEL}. The clf softmax is only "
                    f"K-calibrated on its training cap — re-run build_reliever_choice_model.py "
                    f"so the deployed reliever K-talent stays calibrated.",
                    RuntimeWarning, stacklevel=2)
            feats = m["features"]; beta = np.asarray(m["beta"]); sd = np.asarray(m["sd"])
            w = {fe: float(beta[i] / sd[i]) for i, fe in enumerate(feats)}
            # .get(...,0.0): an artifact fit BEFORE a feature existed carries no weight for it.
            # Defaulting to 0 makes an OLD clf load cleanly and behave exactly as it did — the
            # LIVE server still serves a pre-position-player artifact.
            _CLF = {"w_main": {fe: w.get(fe, 0.0) for fe in _CLF_MAIN_ORDER},
                    "w_int":  np.array([w.get(fe, 0.0) for fe in _CLF_INT_ORDER],
                                       dtype=np.float64)}
    return _CLF


def clf_int_weights():
    """(17,) conditional-logit interaction-weight vector for the kernel, or None."""
    m = _load_clf()
    return m["w_int"] if m else None

# Empirical fatigue signature (fit_bullpen_empirical.py): the WITHIN-PITCHER
# per-PA outcome shift for a reliever pitching on zero days' rest (pitched
# yesterday) vs rested — 563 arms, 58,835 paired PA, 2019-2026.  Each arm is its
# own control, so this is fatigue, not the arm-quality confound that masks it in
# a pooled comparison.  The signature is the velocity-loss fingerprint: fewer
# whiffs AND fewer grounders, with the contact going into the air and harder
# (FO/1B/HR up) — NOT into walks (the old hand-set K→BB/1B split was wrong).
# Order matches OUTCOMES = [K, BB, HBP, GO, FO, 1B, 2B, 3B, HR].
FATIGUE_DELTA = np.array([-0.0059, +0.0008, 0.0000, -0.0050,
                          +0.0058, +0.0030, -0.0004, -0.0006, +0.0022],
                         dtype=np.float64)
FATIGUE_REF = 0.455   # (1−avail) at the modal "pitched-yesterday" state (avail≈0.545)
                      # where Δ was measured: f=1 reproduces the measured shift.
FATIGUE_CAP = 2.0     # cap the multiple for extreme fatigue (3-straight / 30+ pitch)

# Fallback availability means (used only if the params JSON is missing).
_AVAIL_FALLBACK = {"3straight": 0.043, "b2b": 0.138, "yest_30p": 0.065,
                   "yest": 0.545, "rested": 1.0}


def _load_empirical_params():
    """Load the empirically-fit bullpen params + their SAMPLING SEs, written by fit_bullpen_empirical.py.
    Returns (avail_mean, avail_se, delta_mean, delta_se). DEFAULT ON + artifact REQUIRED: when
    BULLPEN_EMPIRICAL != 0 the bullpen_empirical_params.json artifact is HARD-REQUIRED (missing/malformed
    RAISES — no silent degradation to in-code constants). Explicit BULLPEN_EMPIRICAL=0 = golden-master escape
    hatch → the in-code fallback means with zero SE (set_bullpen_param_draw then injects no bullpen uncertainty)."""
    if os.environ.get("BULLPEN_EMPIRICAL", "1") == "0":
        return (dict(_AVAIL_FALLBACK), {k: 0.0 for k in _AVAIL_FALLBACK},
                FATIGUE_DELTA.copy(), np.zeros(N_OUTCOMES))
    path = os.path.join(os.path.dirname(__file__),
                        "data/processed/bullpen_empirical_params.json")
    if not os.path.exists(path):
        # This runs at MODULE IMPORT, so it must NOT raise — build_reliever_choice_model /
        # build_reliever_deploy_basis import this module for structural helpers and run BEFORE
        # fit_bullpen_empirical writes the artifact within the same finalize. "Required" is enforced
        # by the completeness gate (unified_wf verify_snapshot + refit step-16), which checks the FINAL
        # state; by SIM time the artifact always exists. Warn + fall back to constants at import only.
        warnings.warn(
            f"bullpen_empirical_params.json absent at import ({path}); using in-code constants for now. "
            "The completeness gate hard-requires it in the served model (BULLPEN_EMPIRICAL=0 to opt out).",
            RuntimeWarning)
        return (dict(_AVAIL_FALLBACK), {k: 0.0 for k in _AVAIL_FALLBACK},
                FATIGUE_DELTA.copy(), np.zeros(N_OUTCOMES))
    with open(path) as f:
        d = json.load(f)
    am = {k: float(v[0]) for k, v in d["availability"].items()}
    ase = {k: float(v[1]) for k, v in d["availability"].items()}
    return (am, ase, np.array(d["fatigue_delta"], np.float64),
            np.array(d["fatigue_delta_se"], np.float64))


_AVAIL_MEAN, _AVAIL_SE, _FATIGUE_DELTA_MEAN, _FATIGUE_DELTA_SE = _load_empirical_params()

# "Current" params the functions read.  Default = the posterior/point MEAN
# (unchanged behavior).  set_bullpen_param_draw(rng) swaps in a sampled draw.
_AVAIL_CUR = dict(_AVAIL_MEAN)
_FATIGUE_DELTA_CUR = _FATIGUE_DELTA_MEAN.copy()


def set_bullpen_param_draw(rng: np.random.Generator) -> None:
    """Draw the bullpen parameters (availability weights, fatigue Δ) from their
    EMPIRICAL sampling distributions (mean ± SE from fit_bullpen_empirical.py).

    This mirrors Level3Predictor.set_posterior_draw for the matchup rates: it lets
    the simulator integrate over the uncertainty in these ESTIMATED parameters
    instead of plugging in the point estimate.  Call it alongside the predictor's
    draw, then re-assemble the bullpen (team_bullpen + assemble_bullpen_arrays) so
    the sampled values take effect.  Default (no draw) leaves behavior unchanged.
    """
    global _AVAIL_CUR, _FATIGUE_DELTA_CUR
    _AVAIL_CUR = {k: float(np.clip(rng.normal(_AVAIL_MEAN[k], _AVAIL_SE[k]), 0.0, 1.2))
                  for k in _AVAIL_MEAN}
    _FATIGUE_DELTA_CUR = _FATIGUE_DELTA_MEAN + rng.standard_normal(
        _FATIGUE_DELTA_MEAN.shape) * _FATIGUE_DELTA_SE


def clear_bullpen_param_draw() -> None:
    """Restore the point-estimate (mean) bullpen parameters."""
    global _AVAIL_CUR, _FATIGUE_DELTA_CUR
    _AVAIL_CUR = dict(_AVAIL_MEAN)
    _FATIGUE_DELTA_CUR = _FATIGUE_DELTA_MEAN.copy()


def _degrade_for_fatigue(vec: np.ndarray, avail: float) -> np.ndarray:
    """Apply the EMPIRICAL fatigue shift, scaled by how tired the arm is.

    f = (1−avail)/FATIGUE_REF, clipped to [0, FATIGUE_CAP]: a rested arm (avail=1)
    is unchanged (f=0); the modal back-to-back arm (avail≈0.545) gets f≈1 (the
    measured Δ); a 3-straight / 30+pitch arm gets up to 2× the measured shift.
    Uses _FATIGUE_DELTA_CUR (the mean, or a posterior-predictive draw of Δ).
    """
    f = min(FATIGUE_CAP, max(0.0, (1.0 - float(avail)) / FATIGUE_REF))
    if f <= 0.0:
        return vec
    v = np.array(vec, dtype=np.float64) + f * _FATIGUE_DELTA_CUR
    v = np.clip(v, 1e-6, None)
    s = v.sum()
    return (v / s) if s > 0 else vec


def _infer_roles(arms: list[dict]) -> list[int]:
    """Assign closer/setup/middle/long. `arms` is sorted best→worst by quality.

    Closer = the arm with the most 9th-inning-or-later entries (the real save guy),
    restricted to the top-5 quality arms and broken toward higher quality; if no arm
    has any late entries, fall back to the latest-entering top-3 arm. Other top arms
    are setup; high-stamina lower-half arms are long men.
    """
    n = len(arms)
    roles = [ROLE_MIDDLE] * n
    if n == 0:
        return roles
    cand = list(range(min(5, n)))
    if any(arms[i].get("late_entries", 0) > 0 for i in cand):
        closer = max(cand, key=lambda i: (arms[i].get("late_entries", 0), -i))
    else:
        top3 = list(range(min(3, n)))
        closer = max(top3, key=lambda i: arms[i]["avg_entry"])
    roles[closer] = ROLE_CLOSER
    for i in range(min(3, n)):
        if i != closer and roles[i] == ROLE_MIDDLE:
            roles[i] = ROLE_SETUP
    for i in range(n):
        if i >= n // 2 and arms[i]["stamina"] >= 6 and roles[i] == ROLE_MIDDLE:
            roles[i] = ROLE_LONG
    return roles


_MGR = None    # None=not loaded, False=absent, dict=loaded


def _load_mgr():
    """Per-club bullpen random effects (build_manager_effects.py). Absent ⇒ all-zero deviations
    ⇒ byte-identical to the pooled model."""
    global _MGR
    if _MGR is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "processed", "manager_effects.json")
        try:
            m = json.load(open(p))
            _MGR = m if m.get("features") == _CLF_DEV_ORDER else False
            if _MGR is False:
                import warnings
                warnings.warn(f"manager_effects.json features {m.get('features')} != expected "
                              f"{_CLF_DEV_ORDER}; club effects DISABLED", RuntimeWarning)
        except Exception:
            _MGR = False
    return _MGR


def assemble_bullpen_arrays(predictor, pen: pd.DataFrame, opposing_lineup: list[dict],
                            park_effects, gamma: float, rho=None, def_adjust=None,
                            month: int = None, team: str = None):
    """Build the per-reliever arrays the kernel bullpen manager consumes.

    Returns (rel_cum, rel_role, rel_stamina, rel_quality, rel_throws, rel_clf) or None:
      rel_cum     (n, 9, 9) float32 — RAW per-slot outcome vectors, fatigue-degraded
                                   AND defense-tilted (rho), ordered BEST→worst.
      rel_role    (n,)   int8    — ROLE_CLOSER/SETUP/MIDDLE/LONG.
      rel_stamina (n,)   int8    — batters this arm covers before a change.
      rel_quality (n,)   float32 — −E[runs/PA] (higher = better).
      rel_throws  (n,)   int8    — L=0/R=1 (platoon-aware / conditional-logit pick).
    month: optional calendar month → within-season K/BB shape on each matchup vector.
      rel_clf     (n, 6) float64 — [clf_base, quality, avg_entry, late_entries,
                                   stamina, is_closer] for the learned selection model.

    rho/def_adjust: optional fielding-defense tilt for the fielding team. When given,
    each reliever's per-slot vector is tilted exactly like the lineup/blend vectors
    (def_adjust(vec, None, None, rho)) — fixes the prior asymmetry where only the
    blended tiers got the defense tilt. rho=None / all-zero ⇒ identity.
    """
    rv = _reliever_vectors(predictor, pen, opposing_lineup, park_effects, gamma, month=month)
    if not rv:
        return None
    _tilt = (def_adjust is not None and rho is not None and np.any(rho))
    meta = pen.set_index("pitcher")
    for r in rv:
        m = meta.loc[r["pitcher"]]
        r["stamina"]      = int(m["stamina"]) if "stamina" in m else 4
        r["avg_entry"]    = float(m["avg_entry"]) if "avg_entry" in m else 7.0
        r["late_entries"] = int(m["late_entries"]) if "late_entries" in m else 0
        r["avail"]        = float(m.get("avail", 1.0)) if hasattr(m, "get") else float(r.get("avail", 1.0))
        r["usage_weight"] = float(m["usage_weight"]) if "usage_weight" in m else 0.0

    # scratch gassed arms (manager wouldn't use them) unless that empties the pen
    usable = [r for r in rv if r["avail"] >= MIN_DEPLOY_AVAIL] or rv
    for r in usable:
        # fatigue-degrade every per-slot matchup vector, then defense-tilt it (rho),
        # then refresh quality from the fully-adjusted vector.
        slots = []
        for s in range(r["vec_slots"].shape[0]):
            v = _degrade_for_fatigue(r["vec_slots"][s], r["avail"])
            if _tilt:
                # pass the reliever's pitcher_id so the K/BB framing de-confound (his own
                # catcher-exposure) applies to relievers too, not just starters (parity with
                # _adj_lineup); also makes the BIP spray reliever-GB-aware.
                v = np.asarray(def_adjust(np.asarray(v, dtype=float), None, int(r["pitcher"]), rho),
                               dtype=v.dtype)
            slots.append(v)
        r["vec_slots"] = np.vstack(slots)
        r["vec"] = r["vec_slots"].mean(axis=0)
        r["quality"] = -float(np.asarray(r["vec"]) @ _LINEAR_WEIGHTS)
    usable.sort(key=lambda d: d["quality"], reverse=True)
    usable = usable[:MAX_REL_KERNEL]
    usable = append_position_player_arm(usable)

    roles = _infer_roles(usable)
    # rel_cum is (n_rel, 9 batting slots, 9 outcomes) RAW probs — the kernel indexes
    # [reliever, current batter slot] and applies the base-state tilt + draw itself.
    rel_cum     = np.array([np.asarray(r["vec_slots"], dtype=np.float64)
                            for r in usable], dtype=np.float32)
    rel_role    = np.array(roles, dtype=np.int8)
    rel_stamina = np.array([r["stamina"] for r in usable], dtype=np.int8)
    rel_quality = np.array([r["quality"] for r in usable], dtype=np.float32)
    # throws-hand for the platoon-aware selection: L=0, R=1 (the kernel prefers an
    # arm matching the majority hand of the upcoming ≤3 batters in non-priority spots).
    rel_throws  = np.array([0 if str(r.get("throws", "R")) == "L" else 1
                            for r in usable], dtype=np.int8)
    # ── learned-selection feature bundle (conditional logit) ──
    # rel_clf (n, 11) = [clf_base, quality, avg_entry, late_entries, stamina, is_closer, krate,
    #                    dev_q_x_li_protect, dev_q_x_deficit, dev_closer_x_save, is_pos_player]
    # cols 7-9 are this CLUB's random-effect deviations (same value on every row — a club property,
    # carried here so the kernel needs no per-team clf_w plumbing). Zero when the artifact is absent.
    # clf_base = the state-INDEPENDENT part of the utility (precomputed here);
    # the kernel adds the arm×state interactions via clf_int_weights(). All-zero
    # when no model is present ⇒ clf_on=0 ⇒ deterministic path (golden-master safe).
    clf = _load_clf()
    rel_clf = np.zeros((len(usable), 11), dtype=np.float64)
    # col 10 = IS_POSITION_PLAYER identity flag. Set OUTSIDE the `if clf:` block below: it is not a
    # model feature, it is the marker the kernel uses to enforce MLB 5.10(g), and the DETERMINISTIC
    # (clf-off) path needs it too — its mop-up branch takes the WORST available arm, which would be
    # this one in every blowout, legal or not.
    for _i, _r in enumerate(usable):
        if int(_r.get("pitcher", 0)) == POS_ARM_PID:
            rel_clf[_i, 10] = 1.0
    _mg = _load_mgr()
    _dev = np.zeros(3)
    if _mg and team:
        _dev = np.asarray(_mg["raw"].get(str(team), [0.0, 0.0, 0.0]), dtype=np.float64)
    if clf:
        wm = clf["w_main"]
        for i, (r, rr) in enumerate(zip(usable, roles)):
            # LEAGUE-NEUTRAL quality (vs avg batter, h-averaged) — byte-matches the
            # `quality` feature used to FIT the model (build_reliever_choice_model._quality).
            # NOT r["quality"] (that is the matchup/park/γ/fatigue-adjusted value the kernel
            # uses for ordering + the cum vectors; using it here would mis-scale the q-terms).
            pid = int(r["pitcher"])
            _pv0 = np.asarray(predictor._raw_probs_array(pid, -1, 0, 3))
            _pv1 = np.asarray(predictor._raw_probs_array(pid, -1, 1, 3))
            q  = -float((0.5 * (_pv0 + _pv1)) @ _LINEAR_WEIGHTS)
            # league-neutral K rate — byte-matches build_reliever_choice_model._quality's krate,
            # feeding the krate_x_jam term (strikeout arm for an inherited-runner jam).
            kr = float(0.5 * (_pv0[0] + _pv1[0]))
            le = float(r["late_entries"])
            av = float(r.get("avail", 1.0)); uw = float(r.get("usage_weight", 0.0))
            st = float(r["stamina"])
            isc = 1.0 if rr == ROLE_CLOSER else 0.0
            iss = 1.0 if rr == ROLE_SETUP  else 0.0
            isl = 1.0 if rr == ROLE_LONG   else 0.0
            base = (wm["quality"]*q + wm["late_entries"]*le + wm["avail"]*av
                    + wm["usage_weight"]*uw + wm["stamina"]*st
                    + wm["is_closer"]*isc + wm["is_setup"]*iss + wm["is_long"]*isl
                    + wm["is_pos"]*rel_clf[i, 10])      # ASC; col 10 is set above, 0 for real arms
            rel_clf[i, :10] = (base, q, float(r["avg_entry"]), le, st, isc, kr,
                               _dev[0], _dev[1], _dev[2])      # col 10 (is_pos) set above
    return rel_cum, rel_role, rel_stamina, rel_quality, rel_throws, rel_clf


# ═══════════════════════════════════════════════════════════════
# CLI / smoke test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    proc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    print("Building relief-appearance log...")
    build_appearances(proc)

    app = load_appearances(proc)
    print("\nSmoke test — BAL bullpen as of 2026-05-22:")
    pen = team_bullpen(app, "BAL", _dt.date(2026, 5, 22))
    print(pen[["pitcher", "throws", "bf", "n_app", "avail", "weight"]].to_string(index=False))
    print(f"  {len(pen)} relievers, weights sum = {pen.weight.sum():.3f}")
