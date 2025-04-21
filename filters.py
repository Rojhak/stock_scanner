# leader_scan/filters.py
"""
Functions to check for specific chart patterns or conditions.
Expects DataFrames with Capitalized column names ('Close', 'ATR', etc.)
"""
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

def is_compressed(df: pd.DataFrame, index_loc: int, lookback: int = 10, threshold: float = 0.03) -> bool:
    """Checks if price range (%ATR) is compressed over a lookback period."""
    close_col, atr_col = 'Close', 'ATR' # Use Capitalized
    if not all(c in df.columns for c in [close_col, atr_col]): log.debug(f"Compression check skipped: Missing cols"); return False
    if index_loc < lookback: return False
    try:
        recent_df = df.iloc[index_loc - lookback + 1 : index_loc + 1]
        if recent_df.empty: return False
        atr_percent = (recent_df[atr_col] / recent_df[close_col].replace(0, np.nan)).fillna(0)
        is_comp = atr_percent.mean() < threshold
        log.debug(f"Compression check at {df.index[index_loc]}: ATR% mean={atr_percent.mean():.4f}, Thresh={threshold}, Result={is_comp}")
        return is_comp
    except Exception as e: log.warning(f"Error compression check: {e}"); return False

def has_volume_spike(df: pd.DataFrame, index_loc: int, lookback: int = 50, factor: float = 1.5) -> bool:
    """Checks if volume is significantly higher than average."""
    vol_col = 'Volume' # Use Capitalized
    if vol_col not in df.columns: log.debug("Volume spike check: No Volume col"); return False
    if index_loc < lookback: return False
    try:
        current_volume = df[vol_col].iloc[index_loc]
        avg_volume = df[vol_col].iloc[index_loc - lookback : index_loc].mean() # Avg of previous N
        is_spike = pd.notna(current_volume) and pd.notna(avg_volume) and avg_volume > 0 and current_volume > (avg_volume * factor)
        log.debug(f"Volume spike check at {df.index[index_loc]}: Vol={current_volume:.0f}, Avg={avg_volume:.0f}, Factor={factor}, Result={is_spike}")
        return is_spike
    except Exception as e: log.warning(f"Error volume spike check: {e}"); return False

def momentum_confirmed(df: pd.DataFrame, index_loc: int, lookback: int = 5) -> bool:
    """Checks if short-term price change is positive."""
    close_col = 'Close' # Use Capitalized
    if close_col not in df.columns: log.debug("Momentum check: No Close col"); return False
    if index_loc < lookback: return False
    try:
        price_change = df[close_col].diff(lookback).iloc[index_loc]
        is_mom_up = pd.notna(price_change) and price_change > 0
        log.debug(f"Momentum check at {df.index[index_loc]}: Diff({lookback})={price_change:.2f}, Result={is_mom_up}")
        return is_mom_up
    except Exception as e: log.warning(f"Error momentum check: {e}"); return False

def in_base(df: pd.DataFrame, index_loc: int, lookback: int = 50, threshold: float = 0.15) -> bool:
    """Checks if price is within a consolidation range (base)."""
    high_col, low_col = 'High', 'Low' # Use Capitalized
    if not all(c in df.columns for c in [high_col, low_col]): log.debug("Base check: Missing High/Low"); return False
    if index_loc < lookback: return False
    try:
        recent_df = df.iloc[index_loc - lookback + 1 : index_loc + 1]
        period_high = recent_df[high_col].max(); period_low = recent_df[low_col].min()
        is_in_base = pd.notna(period_high) and pd.notna(period_low) and period_low > 0 and ((period_high - period_low) / period_low) < threshold
        log.debug(f"Base check at {df.index[index_loc]}: High={period_high:.2f}, Low={period_low:.2f}, Thresh={threshold*100:.1f}%, Result={is_in_base}")
        return is_in_base
    except Exception as e: log.warning(f"Error in_base check: {e}"); return False

def rsi_confirmed(df: pd.DataFrame, index_loc: int, threshold: float = 50) -> bool:
    """Checks if RSI is above a threshold."""
    rsi_col = 'RSI' # Use Capitalized
    if rsi_col not in df.columns: log.debug("RSI check: No RSI col"); return False
    try:
        current_rsi = df[rsi_col].iloc[index_loc]
        is_rsi_ok = pd.notna(current_rsi) and current_rsi > threshold
        log.debug(f"RSI check at {df.index[index_loc]}: RSI={current_rsi:.1f}, Threshold={threshold}, Result={is_rsi_ok}")
        return is_rsi_ok
    except Exception as e: log.warning(f"Error rsi check: {e}"); return False