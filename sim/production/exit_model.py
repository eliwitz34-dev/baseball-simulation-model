#!/usr/bin/env python3
"""
exit_model.py — serving layer for the starter outing-length removal hazard (train_exit_hazard.py).

The fitted LightGBM hazard is a function of fixed pregame features + live in-game state. Because the
pregame features are constant within a start, each start is BAKED host-side to a small lookup table; the
sim kernel then does O(1) lookups per plate appearance (no model in the hot loop). Outing-length VARIANCE
is produced by the sim generating diverse state trajectories the hazard reacts to — not by injected noise.

FACTORED bake (fast — ~0.1s/start). DEFAULT (live score_diff axis, EXIT_SCORE_DIFF=1): a dense LUT (in
logit) over the top interacting axes
    pitch_count 0..130 (131) × earned_runs 0..8 (9) × score_diff clip(sd,−4,4)+4 0..8 (9) × is_eoi 0..1 (2)
which is conditioned LIVE on the in-game score every PA. (Legacy layout, EXIT_SCORE_DIFF=0: drop the
score_diff axis = 2358 dense cells, score_diff fixed per cell to its empirical conditional mean — biased
because the hazard is non-monotone in score and E[sd|er] is er-confounded; kept only for golden-master.)
plus two ADDITIVE logit-deviation vectors for the secondary axes (each computed at a reference state,
the other held at 0):
    base_out 0..23 (24)   — current jam   (the old model also treated base-out additively)
    accrued_baserunners 0..6 (7) — traffic / trouble build-up
Packed into ONE flat array [dense_logit | bo_adj | br_adj] (21253 live-sd / 2389 legacy). The kernel
auto-detects the layout from the LUT length and reconstructs logit_h = dense + bo_adj[base_out] +
br_adj[accrued_br];  h = sigmoid(logit_h).
`logit_shift` (a per-start additive logit term) carries the global BF-mean offset (EXIT_LOGIT_ADJ, default 0)
+ the bullpen-fatigue tilt (pen_fatigue_z·β, set by the sim) + the age random-intercept (age_tilt, folded in at
bake time). It does NOT carry the per-pitcher leash — that enters via the GBM features avg_bf/avg_pitches/k_pct,
looked up by pitcher_id (see _leash). bf/inning/TTO/bf_vs derived from pitch_count.
"""
import os, pickle
from functools import lru_cache
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL = os.path.join(_DIR, "data", "processed", "exit_hazard.pkl")

# ── grid contract (shared with the kernel: same order, same caps, same offsets) ──
PC_N, ER_N, EOI_N, BO_N, BR_N = 131, 9, 2, 24, 7
PC_MAX, ER_MAX, BR_MAX = 130, 8, 6
DENSE_N = PC_N * ER_N * EOI_N           # 2358
BO_OFF = DENSE_N                        # 2358 — base_out adj block start
BR_OFF = DENSE_N + BO_N                 # 2382 — accrued_br adj block start
LUT_SIZE = DENSE_N + BO_N + BR_N        # 2389
# ── live score_diff axis (EXIT_SCORE_DIFF=1): dense becomes pitch_count × er × score_diff × is_eoi, with
#    score_diff bucketed to clip(sd,−4,+4)+4 ∈ 0..8. Replaces the conditional-mean plug-in, which is biased
#    because the hazard is non-monotone in score_diff and E[sd|er] is er-confounded. Kernels auto-detect the
#    layout from the LUT length (>LUT_SIZE ⇒ live-sd), so OFF is byte-identical to the legacy bake. ──
SD_N, SD_OFF = 9, 4
DENSE_N_SD = PC_N * ER_N * SD_N * EOI_N   # 21222
BO_OFF_SD = DENSE_N_SD                     # 21222 — base_out adj block start (live-sd layout)
BR_OFF_SD = DENSE_N_SD + BO_N             # 21246 — accrued_br adj block start (live-sd layout)
LUT_SIZE_SD = DENSE_N_SD + BO_N + BR_N    # 21253
# ── v2 EXACT layout (2026-07-21, [[project-bf-tail-defect]]): the additive br deviation vector and
#    the single-reference bo deviation were measured to LEAK hazard vs the GBM they were baked from
#    (−21% at 60-75 pitches, −11% at 85-95 on 52,745 real-state replays) — the direct mechanism of
#    the sim's fat deep-outing tail (P(BF≥28) served 1.56x reality). v2 removes the approximation:
#    accrued_br joins the DENSE grid (it interacts with depth; additivity was the error), and the
#    base_out deviation becomes PITCH-COUNT-CONDITIONED (bo is a transient within-PA jam — additive
#    is fine, but only at a depth-matched reference). One trailing cell carries h_shock, the
#    competing-risk injury hazard (train_exit_hazard), applied by the kernel as
#    h = h_shock + (1−h_shock)·sigmoid(L). Selected automatically iff the pkl carries the v2 key;
#    kernels detect the layout from LUT length, so old artifacts stay byte-identical.
#    MICRO-PARITY LESSONS (same day): base_out is not additively separable either — the jam effect
#    interacts with accrued_br/er AND score_diff, and any logit-delta trick is further scrambled by
#    the isotonic calibrator's plateaus (deltas computed on one plateau, applied on another). Every
#    partial factorization tried left 5-20% serve error somewhere. So v2 approximates NOTHING:
#    base_out joins the dense grid. Size is tamed by a structural fact of baseball, not by an
#    approximation — a half-inning STARTS with bases empty, so eoi=1 co-occurs only with bo=0
#    (regular innings) or the extras ghost runner, which the kernel encodes as exactly state=2.
#    Block A: eoi=0, full (pc × er × sd × br × bo).  Block B: eoi=1, (pc × er × sd × br) × bo∈{0,2}.
#    BR CAP LESSON (the final leak, same day): the GBM trains on UNCAPPED accrued_baserunners and
#    is monotone-increasing in it, but the original grid clamped br at 6 — and 54-60% of deep-zone
#    PAs (85-105 pitches) carry MORE than 6 accrued baserunners, so the serve under-called exactly
#    where removals happen. The v2 axis spans the training distribution (p99.9 = 13-14): cap 14.
BR_N_V2 = 15                                         # accrued_baserunners 0..14 (training p99.9)
BR_MAX_V2 = 14
DENSE_N_V2 = PC_N * ER_N * SD_N * BR_N_V2 * BO_N     # 3,819,960  block A (eoi=0)
EOI_OFF_V2 = DENSE_N_V2                              # block B start (eoi=1)
EOI_N_V2 = PC_N * ER_N * SD_N * BR_N_V2 * 2          # 318,330    block B: ghost slot 0=empty, 1=runner-on-2nd
SHOCK_OFF_V2 = DENSE_N_V2 + EOI_N_V2                 # 4,138,290 — h_shock (stored as a PROBABILITY)
LUT_SIZE_V2 = SHOCK_OFF_V2 + 1                       # 4,138,291
# reference state for the additive deviation vectors (decision-zone, mid-inning)
_REF_PC, _REF_ER, _REF_EOI = 90.0, 2.0, 0.0

_MODEL = None
_MISSING_MSG = (
    f"starter exit hazard model is missing: {_PKL}. "
    "Run build_exit_dataset.py and train_exit_hazard.py before simming starter K/BF markets."
)


def _load():
    global _MODEL
    if _MODEL is None:
        _MODEL = pickle.load(open(_PKL, "rb")) if os.path.exists(_PKL) else False
    return _MODEL


_LEASH = None


def _leash():
    """Per-pitcher recency-shrunk current leash {pid: (avg_bf, avg_pitches, k_pct)} from build_exit_dataset.
    The bake uses these IDENTICAL-to-training values (looked up by pitcher_id) so serve can't drift from train."""
    global _LEASH
    if _LEASH is None:
        p = os.path.join(_DIR, "data", "processed", "exit_leash.parquet")
        if os.path.exists(p):
            import pandas as pd
            d = pd.read_parquet(p)
            _LEASH = {int(i): (float(r.avg_bf), float(r.avg_pitches), float(r.k_pct)) for i, r in d.iterrows()}
        else:
            _LEASH = {}
    return _LEASH


def estab_ttop():
    """Set of established starters (>= 12 starts) for the fringe-TTOP membership test (re-homed here from
    the retired starter-exit GAM npz). Pitchers absent from this set — callups / unproven arms — get the
    steeper fringe times-through-order curve in run_pregame; established arms get the global curve."""
    m = _load()
    return (m.get("estab_ttop") if m else None) or set()


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _features(pc, er, eoi, base_out, accrued_br, avg_bf, avg_pitches, k_pct, age, sd_tab, feats, score_diff=None,
              ramp_budget_delta=0.0, il_first=0.0, il_arm_first=0.0):
    """Assemble the GBM feature matrix for given live-axis arrays + this start's pregame values.
    score_diff: explicit array (live-sd bake) or None → conditional-mean sd_tab[pc,er] (legacy bake)."""
    n = pc.size
    eff = (avg_bf / avg_pitches) if avg_pitches > 0 else (1.0 / 3.9)   # batters per pitch
    bf = pc * eff
    inning = np.clip(np.floor(bf / 4.3) + 1.0, 1.0, 9.0)
    tto = np.clip(np.floor(bf / 9.0), 0.0, 3.0)
    bf_vs = np.clip(bf / max(avg_bf, 1.0), 0.0, 3.0)
    pitch_vs = np.clip(pc / max(avg_pitches, 1.0), 0.0, 3.0)
    if score_diff is None:
        score_diff = (sd_tab[pc.astype(int), er.astype(int)] if sd_tab is not None else np.zeros(n))
    cols = {
        "batters_faced": bf, "pitch_count": pc, "inning": inning, "earned_runs": er,
        "score_diff": score_diff, "is_end_of_inning": eoi, "times_through_order": tto,
        "accrued_baserunners": accrued_br, "base_out_state": base_out,
        "pitcher_avg_bf": np.full(n, avg_bf), "pitcher_avg_pitches": np.full(n, avg_pitches),
        "pitcher_k_pct": np.full(n, k_pct), "bf_vs_avg": bf_vs, "pitch_vs_avg": pitch_vs,
        "age": np.full(n, age),
        "ramp_budget_delta": np.full(n, ramp_budget_delta),   # per-start ramp/injury (constant across LUT cells)
        "il_first": np.full(n, il_first), "il_arm_first": np.full(n, il_arm_first),
    }
    return np.column_stack([cols[f] for f in feats]).astype(np.float32)


def bake_start_lut(avg_bf, avg_pitches, k_pct, logit_shift=0.0, pitcher_id=None,
                   ramp_budget_delta=0.0, il_first=0.0, il_arm_first=0.0, leash_scale=1.0):
    """Bake one start to a flat (LUT_SIZE,) packed logit table. Cached on rounded inputs so the live/resume
    path re-bakes each start only once. Returned array is read-only to the kernel. The starter's age is
    looked up by pitcher_id (latest observed), defaulting to the league median. ramp_budget_delta / il_first /
    il_arm_first are pregame-constant ramp/injury features (0 = neutral) the GBM uses iff trained with them."""
    m = _load()
    if pitcher_id is not None:                  # serving leash (recency-shrunk, training-consistent) by pid
        lk = _leash().get(int(pitcher_id))
        if lk is not None:
            avg_bf, avg_pitches, k_pct = lk
    if leash_scale != 1.0:                       # market-implied leash override (STARTER_MKT_LEASH) — AFTER the
        avg_bf = float(avg_bf) * leash_scale     # pid-leash lookup so it actually takes effect on the bake
        avg_pitches = float(avg_pitches) * leash_scale
    if m and pitcher_id is not None:
        age = m.get("age_latest", {}).get(int(pitcher_id), m.get("age_default", 28.0))
    else:
        age = (m.get("age_default", 28.0) if m else 28.0)
    # data-calibrated age random-intercept (centered logit tilt the GBM under-weights) → folded into the shift
    age_tilt = m.get("age_adj", {}).get(int(round(age)), 0.0) if m else 0.0
    sd_axis = os.environ.get("EXIT_SCORE_DIFF", "1") == "1"   # live score_diff hazard axis (DEFAULT ON; set =0 for legacy/golden-master)
    return _bake_cached(round(float(avg_bf), 2), round(float(avg_pitches), 1),
                        round(float(k_pct), 4), round(float(logit_shift) + age_tilt, 4), round(float(age), 1), sd_axis,
                        round(float(ramp_budget_delta), 2), float(il_first), float(il_arm_first))


# v2 LUTs are 33MB each (4,138,291 f64). A full slate needs ≤ ~36 (17 games × 2 starters + churn),
# so the default cap of 40 bounds the cache at ~1.3GB — safe on the production server (the quoting service and
# the in-game service each hold one). Backtest boxes with more RAM can raise EXIT_LUT_CACHE; this is an
# OPS sizing knob (memory ceiling), not a model parameter.
_LUT_CACHE_N = int(os.environ.get("EXIT_LUT_CACHE", "40"))

# ── bake threading (2026-07-22, PORTFOLIO_ARCH_STREAMLINE §4b): the v2 exact bake is ~1.93M GBM
# evaluations per start, and eval harnesses pin OMP_NUM_THREADS=1 for the SIM hot loop — which also
# strangled booster.predict to one thread (model_eval 653 games: 17min → 4h15m). LightGBM accepts a
# per-call num_threads that overrides the OpenMP default, so the bake threads independently of the
# pin. EXIT_BAKE_THREADS: explicit count; unset/0 → ncpu // EVAL_WORKERS_ACTIVE (set by parallel
# harnesses so workers×threads ≈ cores), or ncpu//2 for a single process (live server: leaves
# headroom for the other services).
def _bake_threads():
    v = int(os.environ.get("EXIT_BAKE_THREADS", "0"))
    if v > 0:
        return v
    ncpu = os.cpu_count() or 1
    workers = int(os.environ.get("EVAL_WORKERS_ACTIVE", "0"))
    return max(1, ncpu // workers) if workers > 0 else max(1, ncpu // 2)


@lru_cache(maxsize=_LUT_CACHE_N)
def _bake_cached(avg_bf, avg_pitches, k_pct, logit_shift, age, sd_axis=False,
                 ramp_budget_delta=0.0, il_first=0.0, il_arm_first=0.0):
    m = _load()
    if not m:
        raise FileNotFoundError(_MISSING_MSG)
    booster, iso, feats, sd_tab = m["booster"], m["isotonic"], m["features"], m.get("score_diff_table")
    nit = m.get("best_iteration") or 0
    _nt = _bake_threads()
    _hz = lambda X: iso.predict(booster.predict(X, num_iteration=nit, num_threads=_nt))
    _rf = dict(ramp_budget_delta=ramp_budget_delta, il_first=il_first, il_arm_first=il_arm_first)  # per-start ramp/injury

    # additive deviation vectors at the reference state (other axis held at 0). sd_ref=None ⇒ conditional-mean
    # score_diff via sd_tab (legacy); sd_ref=0.0 ⇒ live-sd layout (these deviations are ~score_diff-separable).
    def _ref(name, vals, sd_ref=None):
        k = vals.size
        base = dict(pc=np.full(k, _REF_PC), er=np.full(k, _REF_ER), eoi=np.full(k, _REF_EOI),
                    base_out=np.zeros(k), accrued_br=np.zeros(k))
        base[name] = vals.astype(np.float64)
        sdv = None if sd_ref is None else np.full(k, sd_ref)
        X = _features(base["pc"], base["er"], base["eoi"], base["base_out"], base["accrued_br"],
                      avg_bf, avg_pitches, k_pct, age, sd_tab, feats, score_diff=sdv, **_rf)
        lg = _logit(_hz(X))
        return lg - lg[0]

    if m.get("v2"):
        # ── v2 EXACT bake: NO factorization anywhere. Block A = every mid-inning state
        # (pc × er × sd × br × bo, eoi=0); block B = inning-start states (eoi=1), which by the
        # rules of baseball carry only bases-empty or the extras ghost runner (kernel state==2).
        # ~1.93M GBM evaluations per start, chunked; lru-cached so the cost is once per starter.
        def _grid_logit(eoi_val, bo_vals):
            pc, er, sd, br, bo = np.meshgrid(np.arange(PC_N), np.arange(ER_N), np.arange(SD_N),
                                             np.arange(BR_N_V2), np.asarray(bo_vals), indexing="ij")
            pc = pc.ravel().astype(np.float64); er = er.ravel().astype(np.float64)
            sd = sd.ravel().astype(np.float64); br = br.ravel().astype(np.float64)
            bo = bo.ravel().astype(np.float64)
            out = np.empty(pc.size, np.float64)
            CH = 250_000                                     # chunk the predict: bounded peak memory
            for s0 in range(0, pc.size, CH):
                sl = slice(s0, min(s0 + CH, pc.size))
                X = _features(pc[sl], er[sl], np.full(sl.stop - sl.start, float(eoi_val)),
                              bo[sl], br[sl], avg_bf, avg_pitches, k_pct, age, sd_tab, feats,
                              score_diff=(sd[sl] - SD_OFF), **_rf)
                out[sl] = _logit(_hz(X))
            return out + logit_shift
        dense = _grid_logit(0, np.arange(BO_N))              # block A: eoi=0, all 24 base-out states
        eoi_blk = _grid_logit(1, np.array([0, 2]))           # block B: eoi=1, empty / ghost-on-2nd
        shock = np.array([float(m.get("h_shock", 0.0))])
        return np.ascontiguousarray(np.concatenate([dense, eoi_blk, shock]), np.float64)

    if sd_axis:
        # dense (pitch_count × earned_runs × score_diff × is_eoi); score_diff is the LIVE value (bucket−SD_OFF
        # ∈ −4..+4), evaluated per cell — no conditional-mean plug-in, no Jensen/confound bias.
        pc, er, sd, eoi = np.meshgrid(np.arange(PC_N), np.arange(ER_N), np.arange(SD_N), np.arange(EOI_N), indexing="ij")
        pc = pc.ravel().astype(np.float64); er = er.ravel().astype(np.float64)
        sd = sd.ravel().astype(np.float64); eoi = eoi.ravel().astype(np.float64)
        z = np.zeros(pc.size)
        dense = _logit(_hz(_features(pc, er, eoi, z, z, avg_bf, avg_pitches, k_pct, age, sd_tab, feats,
                                     score_diff=(sd - SD_OFF), **_rf))) + logit_shift
        bo_adj = _ref("base_out", np.arange(BO_N), sd_ref=0.0)
        br_adj = _ref("accrued_br", np.arange(BR_N), sd_ref=0.0)
        return np.ascontiguousarray(np.concatenate([dense, bo_adj, br_adj]), np.float64)

    # ── legacy bake (conditional-mean score_diff via sd_tab; byte-identical to pre-axis) ──
    pc, er, eoi = np.meshgrid(np.arange(PC_N), np.arange(ER_N), np.arange(EOI_N), indexing="ij")
    pc = pc.ravel().astype(np.float64); er = er.ravel().astype(np.float64); eoi = eoi.ravel().astype(np.float64)
    z = np.zeros(pc.size)
    dense = _logit(_hz(_features(pc, er, eoi, z, z, avg_bf, avg_pitches, k_pct, age, sd_tab, feats, **_rf))) + logit_shift
    bo_adj = _ref("base_out", np.arange(BO_N))
    br_adj = _ref("accrued_br", np.arange(BR_N))
    return np.ascontiguousarray(np.concatenate([dense, bo_adj, br_adj]), np.float64)


def hazard(lut, pitch_count, er, accrued_br, base_out, is_eoi, score_diff=0):
    """Reference per-PA hazard from a packed LUT — the exact computation the kernel inlines.
    Layout auto-detect by length: v2 exact (151,699: br in the dense grid, pc-conditioned bo_adj,
    trailing h_shock mixed in probability space) > live-sd (21,253) > legacy (2,389)."""
    pc = int(min(max(pitch_count, 0), PC_MAX))
    e = int(min(er, ER_MAX)); b = int(min(accrued_br, BR_MAX))
    if lut.shape[0] == LUT_SIZE_V2:                   # v2 exact layout + competing-risk shock
        sd = int(min(max(score_diff, -SD_OFF), SD_OFF)) + SD_OFF
        b2 = int(min(accrued_br, BR_MAX_V2))          # v2 spans the TRAINING br range (cap 14, not 6)
        if int(is_eoi) == 1:                          # inning start: bases empty or extras ghost (bo==2)
            L = lut[EOI_OFF_V2 + (((pc * ER_N + e) * SD_N + sd) * BR_N_V2 + b2) * 2
                    + (1 if int(base_out) == 2 else 0)]
        else:
            L = lut[((((pc * ER_N + e) * SD_N + sd) * BR_N_V2 + b2) * BO_N) + int(base_out)]
        hs = lut[SHOCK_OFF_V2]
        return hs + (1.0 - hs) * (1.0 / (1.0 + np.exp(-L)))
    if lut.shape[0] > LUT_SIZE:                       # live score_diff axis
        sd = int(min(max(score_diff, -SD_OFF), SD_OFF)) + SD_OFF
        L = lut[((pc * ER_N + e) * SD_N + sd) * EOI_N + int(is_eoi)] + lut[BO_OFF_SD + int(base_out)] + lut[BR_OFF_SD + b]
    else:
        L = lut[(pc * ER_N + e) * EOI_N + int(is_eoi)] + lut[BO_OFF + int(base_out)] + lut[BR_OFF + b]
    return 1.0 / (1.0 + np.exp(-L))
