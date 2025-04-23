# leader_scan/indicators.py
"""
Technical indicator calculation functions using pandas.
Aims to provide common indicators without a hard dependency on libraries like TA-Lib.
Uses lowercase column names internally ('open', 'high', 'low', 'close', 'volume').
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

# --- Helper Function for Safe Division ---
def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = np.nan) -> pd.Series:
    """Performs division, replacing 0/0 or x/0 with the default value."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator.replace(0, np.nan)
    return result.fillna(default)

# --- Moving Averages ---
def ma(series: pd.Series, window: int = 50) -> pd.Series:
    """Simple Moving Average (SMA)."""
    if not isinstance(series, pd.Series): log.error("MA input must be a pandas Series."); return pd.Series(dtype=float)
    if window <= 0: log.error("MA window must be positive."); return pd.Series(dtype=float)
    return series.rolling(window, min_periods=max(1, window // 2)).mean()

def ema(series: pd.Series, span: int = 21) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    if not isinstance(series, pd.Series): log.error("EMA input must be a pandas Series."); return pd.Series(dtype=float)
    if span <= 0: log.error("EMA span must be positive."); return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False, min_periods=span // 2).mean()

# --- True Range & ATR ---
def _true_range(df: pd.DataFrame) -> pd.Series: # Expects df with 'high', 'low', 'close'
    """Calculates the True Range component using lowercase columns."""
    # *** USE LOWERCASE COLUMN NAMES ***
    high_col = 'high'
    low_col = 'low'
    close_col = 'close'

    # Check if required columns exist
    if not all(c in df.columns for c in [high_col, low_col, close_col]):
         missing = [c for c in [high_col, low_col, close_col] if c not in df.columns]
         log.error(f"_true_range missing required columns: {missing}")
         return pd.Series(dtype=float, index=df.index).fillna(np.nan)

    HIGH = df[high_col]
    LOW = df[low_col]
    CLOSE = df[close_col]
    prev_close = CLOSE.shift(1)

    tr1 = (HIGH - LOW).abs()
    tr2 = (HIGH - prev_close).abs()
    tr3 = (LOW - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    return tr

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR). Uses Wilder's smoothing (alpha = 1 / window).
    Expects DataFrame with lowercase 'high', 'low', 'close' columns.

    Args:
        df: DataFrame with lowercase 'high', 'low', 'close' columns.
        window: Smoothing period (default: 14).

    Returns:
        A pandas Series containing the ATR values.
    """
    # Check passed df columns (already lowercase from _calculate...)
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        log.error(f"ATR calculation requires lowercase columns: {required_cols}. Found: {df.columns.tolist()}")
        return pd.Series(dtype=float, index=df.index).fillna(np.nan)

    try:
        tr = _true_range(df) # Pass the df directly
        # Use EMA with alpha=1/window for Wilder's smoothing
        atr_series = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        return atr_series
    except Exception as e:
        log.error(f"Error calculating ATR: {e}")
        return pd.Series(dtype=float, index=df.index).fillna(np.nan)


# --- Relative Strength (ratio vs benchmark) ---
def rs_line(
    target_close: pd.Series, # Assumes 'close' series
    bench_close: pd.Series,
    smooth: Optional[int] = None,
) -> pd.Series:
    """
    Calculates the Price Relative Strength line (target price / benchmark price).
    Aligns series by index before calculation.
    """
    if not isinstance(target_close, pd.Series) or not isinstance(bench_close, pd.Series):
        log.error("RS Line inputs must be pandas Series."); return pd.Series(dtype=float)
    try:
        aligned_target, aligned_bench = target_close.align(bench_close, join='inner')
        if aligned_bench.empty or aligned_target.empty:
            log.warning("RS Line: No overlapping data between target and benchmark.")
            return pd.Series(dtype=float, index=target_close.index).reindex(target_close.index)
        rs = _safe_divide(aligned_target, aligned_bench, default=np.nan)
        if smooth is not None and smooth > 0: rs = ema(rs, span=smooth)
        return rs.reindex(target_close.index)
    except Exception as e:
        log.error(f"Error calculating RS Line: {e}")
        return pd.Series(dtype=float, index=target_close.index).fillna(np.nan)


def rs_new_high(rs_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Identifies where the Relative Strength series makes a new high over a lookback period."""
    if not isinstance(rs_series, pd.Series): log.error("rs_new_high input must be Series."); return pd.Series(dtype=bool)
    if lookback <= 0: log.error("rs_new_high lookback must be positive."); return pd.Series(dtype=bool)
    try:
        rolling_max = rs_series.rolling(lookback, min_periods=lookback).max()
        is_new_high = rs_series >= rolling_max.shift(1)
        return is_new_high.fillna(False)
    except Exception as e:
        log.error(f"Error calculating RS New High: {e}")
        return pd.Series(dtype=bool, index=rs_series.index).fillna(False)

# --- Volatility / Volume Contraction Pattern (VCP) Helpers ---
def atr_multiple(price: pd.Series, atr_series: pd.Series) -> pd.Series: # Assumes 'close' series for price
    """Calculates ATR as a percentage of the closing price."""
    if not isinstance(price, pd.Series) or not isinstance(atr_series, pd.Series):
        log.error("atr_multiple inputs must be Series."); return pd.Series(dtype=float)
    try:
        aligned_price, aligned_atr = price.align(atr_series, join='inner')
        if aligned_price.empty: return pd.Series(dtype=float, index=price.index).reindex(price.index)
        atr_mult = _safe_divide(aligned_atr, aligned_price, default=np.nan) * 100.0
        return atr_mult.reindex(price.index)
    except Exception as e:
        log.error(f"Error calculating ATR Multiple: {e}")
        return pd.Series(dtype=float, index=price.index).fillna(np.nan)


def vcp_dry_up(volume: pd.Series, lookback: int = 10, thresh: float = 0.4) -> pd.Series: # Assumes 'volume' series
    """Minervini-style Volume Dry-Up check for VCP."""
    if not isinstance(volume, pd.Series): log.error("vcp_dry_up input must be Series."); return pd.Series(dtype=bool)
    if lookback <= 0: log.error("vcp_dry_up lookback must be positive."); return pd.Series(dtype=bool)
    try:
        sum_2n = volume.rolling(lookback * 2, min_periods=lookback * 2).sum()
        sum_prev_n = sum_2n.shift(lookback)
        sum_recent_n = sum_2n - sum_prev_n
        ratio = _safe_divide(sum_recent_n, sum_prev_n, default=np.inf)
        is_dry = ratio < thresh
        return is_dry.fillna(False)
    except Exception as e:
        log.error(f"Error calculating VCP Dry Up: {e}")
        return pd.Series(dtype=bool, index=volume.index).fillna(False)

# --- Explicit Export List ---
__all__ = ["ma", "ema", "atr", "rs_line", "rs_new_high", "atr_multiple", "vcp_dry_up"]