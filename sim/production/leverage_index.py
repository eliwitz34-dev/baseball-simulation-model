#!/usr/bin/env python3
"""leverage_index.py — shared Leverage Index (Tango) lookup for train AND serve.

ONE definition of the LI key, imported by both build_reliever_choice_model.py (training) and
game_simulation.py (the kernel), so a state indexes to the same LI on both sides. The table is
built by build_leverage_index.py from our own statcast games (no pinned constants).

KEY (pitching-team frame, matching e["lead"] in training and home_lead/away_lead in the kernel):
    li[min(inning,11)-1, is_bot, outs*8 + base_code, clip(pitcher_lead,-8,8)+8]

⚠ LI measures IMPORTANCE, not DIRECTION — it is near-symmetric in the lead's sign. It does NOT
replace the signed protect/deficit features; both are required.
"""
from __future__ import annotations
import os
import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "processed", "leverage_index.npy")
N_INN, N_HALF, N_STATE, N_SD = 11, 2, 24, 17
SD_OFF, SD_MAX = 8, 8

_TABLE = None


def load_li():
    """(11,2,24,17) LI table, or an all-ones fallback if the artifact is absent (⇒ LI is inert
    and the model degrades to its other features rather than crashing)."""
    global _TABLE
    if _TABLE is None:
        if os.path.exists(_PATH):
            _TABLE = np.ascontiguousarray(np.load(_PATH), dtype=np.float64)
        else:
            _TABLE = np.ones((N_INN, N_HALF, N_STATE, N_SD), dtype=np.float64)
    return _TABLE


def li_key(inning0: int, is_bot: int, base: int, outs: int, lead: int):
    """(i, h, s, d) indices. inning0 is 0-BASED; base 0-7; outs 0-2; lead pitcher-perspective."""
    i = inning0 if inning0 < N_INN else N_INN - 1
    if i < 0:
        i = 0
    h = 1 if is_bot else 0
    o = 0 if outs < 0 else (2 if outs > 2 else outs)
    b = base & 7
    s = o * 8 + b
    d = lead
    if d < -SD_MAX:
        d = -SD_MAX
    elif d > SD_MAX:
        d = SD_MAX
    return i, h, s, d + SD_OFF


def li_lookup(inning0: int, is_bot: int, base: int, outs: int, lead: int) -> float:
    i, h, s, d = li_key(inning0, is_bot, base, outs, lead)
    return float(load_li()[i, h, s, d])
