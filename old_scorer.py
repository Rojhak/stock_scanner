from __future__ import annotations

import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None

# /Users/fehmikatar/Desktop/Stock/leader_scan/scorer.py
"""
Scoring utilities
~~~~~~~~~~~~~~~~~
Isolate the logic that converts a row of boolean/float metrics into a single
integer **Leadership Score** (0‒5).  This decouples the scoring model from
`main.py`, so you can tweak thresholds or add features without touching the
orchestrator.

Default rule‐set (one point each)
---------------------------------
* stage2           – confirmed up‑trend
* drawdown_ok      – ≤ CONFIG["drawdown_max"]
* rs_new_hi        – RS‑line makes fresh 52‑wk high
* vol_thrust       – today's volume ≥ 1.4 × 50‑day avg
* fundamental_ok   – EPS & Sales growth above CONFIG minima

The functions here do **not** fetch data; they expect a *row* already populated
with those keys (see `_score_symbol` in main.py).

Example
-------
>>> from leader_scan.scorer import compute_row_score
>>> row = {"stage2": True, "drawdown_ok": True, "rs_new_hi": False,
...        "vol_thrust": True, "fundamental_ok": True}
>>> compute_row_score(row)
4
"""

from typing import Dict, Mapping

from .config import CONFIG

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
_BOOL_RULES = (
    "stage2",
    "drawdown_ok",
    "rs_new_hi",
    "vol_thrust",
    "fundamental_ok",
)


def _bool_flags(row: Mapping[str, object]) -> Dict[str, bool]:
    """
    Ensure each rule name exists in `row`; cast to bool for scoring.

    Missing keys default to False so the score never raises a KeyError.
    """
    return {name: bool(row.get(name, False)) for name in _BOOL_RULES}


def compute_row_score(row: Mapping[str, object]) -> int:
    """
    Return 0‒5 score for a single row of metrics (dict‑like).

    The calling code is responsible for writing back the score if desired:
        df.loc[ticker, "score"] = compute_row_score(df.loc[ticker])
    """
    flags = _bool_flags(row)
    return sum(flags.values())


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #
def score_dataframe(df):
    """
    Vectorised version that appends/overwrites a 'score' column.

    Parameters
    ----------
    df : pandas.DataFrame
        Needs at least the boolean columns listed in `_BOOL_RULES`.

    Returns
    -------
    DataFrame with a new 'score' column.
    """
    import pandas as pd  # local import to avoid hard dependency elsewhere

    bool_df = df.reindex(columns=_BOOL_RULES).fillna(False).astype(bool)
    df = df.copy()
    df["score"] = bool_df.sum(axis=1)
    return df


# --------------------------------------------------------------------------- #
# __all__
# --------------------------------------------------------------------------- #
__all__ = ["compute_row_score", "score_dataframe"]