# leader_scan/scorer.py
"""
Stock scoring engine. Works with CAPITALIZED column names ('Close', 'ATR', etc.).
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

from .config import CONFIG
# Filters must also expect Capitalized column names now
from .filters import (
    is_compressed, has_volume_spike, momentum_confirmed,
    in_base, rsi_confirmed
)

log = logging.getLogger(__name__)

DEFAULT_STOP_ATR_MULT = 1.5
DEFAULT_TARGET_MIN_MULT = CONFIG.get("min_r_multiple", 2.5)
MIN_SCORE_THRESHOLD = 3.0

# --- Helper Functions ---
def _determine_setup_type(flags: Dict[str, bool]) -> str:
    """Determines setup type based on boolean flags."""
    # Check in order of priority
    if flags.get('in_stage2', False): return 'STAGE2'
    if flags.get('ma_cross', False) and flags.get('compressed', False): return 'MA_CROSS'
    if flags.get('rs_rising', False) and flags.get('above_ma50', False): return 'LEADER'
    return 'NONE'

def _calculate_risk_reward(
    row_data: pd.Series, setup_type: str,
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT, target_min_mult: float = DEFAULT_TARGET_MIN_MULT
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Calculates RR metrics. Expects CAPITALIZED column names in row_data."""
    stop, target, r_multiple, risk_pct = None, None, None, None
    try:
        price = row_data.get('Close'); atr_val = row_data.get('ATR'); low_price = row_data.get('Low')
        if pd.isna(price) or price <= 0: log.debug("RR Calc: Invalid entry price."); return None, None, None, None
        if pd.isna(atr_val) or atr_val <= 0: atr_val = price * 0.03; log.debug(f"RR Calc: Using default ATR: {atr_val:.4f}")
        if pd.isna(low_price): low_price = price

        if setup_type == 'STAGE2':
            fib_61_8 = row_data.get('Fib_61.8') # Assumes this name if exists
            if pd.notna(fib_61_8): stop = fib_61_8 * 0.99
            else: stop = low_price - (stop_atr_mult * atr_val)
        elif setup_type == 'MA_CROSS':
            ma20 = row_data.get('MA20'); atr_stop = low_price - (stop_atr_mult * atr_val)
            if pd.notna(ma20): stop = min(ma20 * 0.995, atr_stop)
            else: stop = atr_stop
        else: stop = low_price - (stop_atr_mult * atr_val)

        if stop >= price: stop = price - (0.5 * atr_val); log.warning(f"RR Calc: Stop >= entry, adjusted to {stop:.2f}")
        if stop >= price: log.error("RR Calc: Stop still >= entry."); return None, None, None, None

        risk_abs = price - stop
        if risk_abs <= 0: log.warning("RR Calc: Non-positive risk."); return None, None, None, None
        risk_pct = (risk_abs / price) * 100.0
        reward_abs = risk_abs * target_min_mult; target = price + reward_abs
        r_multiple = reward_abs / risk_abs if risk_abs > 0 else np.inf
        log.debug(f"RR Calc Results: Entry={price:.2f}, Stop={stop:.2f}, Target={target:.2f}, Risk={risk_pct:.2f}%, R={r_multiple:.2f}")
        return round(stop, 4), round(target, 4), round(r_multiple, 2), round(risk_pct, 2)
    except Exception as e: log.error(f"Error calculating R/R: {e}", exc_info=False); return None, None, None, None

def score_symbol(df: pd.DataFrame, idx: int = -1) -> Dict:
    """Scores a single row. Expects df with CAPITALIZED columns."""
    try:
        row_iloc = idx if idx >= 0 else len(df) + idx
        if not (0 <= row_iloc < len(df)): log.warning(f"Index {idx} out of bounds."); return {}
        row = df.iloc[row_iloc] # Row has Capitalized columns
    except Exception as e: log.error(f"Error accessing row index {idx}: {e}"); return {}
    if not isinstance(row, pd.Series): log.error("Could not extract Series row."); return {}

    flags: Dict[str, bool] = {}
    try:
        # --- Use Capitalized column names ---
        close_val = row.get('Close'); ma50_val = row.get('MA50'); ma200_val = row.get('MA200')
        flags['above_ma50'] = pd.notna(close_val) and pd.notna(ma50_val) and close_val > ma50_val
        flags['above_ma200'] = pd.notna(close_val) and pd.notna(ma200_val) and close_val > ma200_val

        # Filters called with Capitalized df
        flags['compressed'] = is_compressed(df, row_iloc)
        flags['volume_spike'] = has_volume_spike(df, row_iloc)
        flags['momentum_up'] = momentum_confirmed(df, row_iloc)
        flags['in_base'] = in_base(df, row_iloc)
        flags['rsi_bullish'] = rsi_confirmed(df, row_iloc, threshold=50)

        rs_slope_val = row.get('RS_slope'); flags['rs_rising'] = pd.notna(rs_slope_val) and rs_slope_val > 0
        ma10_val = row.get('MA10'); ma20_val = row.get('MA20')
        flags['ma_cross'] = pd.notna(ma10_val) and pd.notna(ma20_val) and pd.notna(ma50_val) and ma10_val > ma20_val > ma50_val

        fib_38_2 = row.get('Fib_38.2'); fib_61_8 = row.get('Fib_61.8') # Assumes Capitalized if present
        flags['in_stage2'] = pd.notna(fib_38_2) and pd.notna(fib_61_8) and pd.notna(close_val) and fib_61_8 <= close_val <= fib_38_2
        if flags['in_stage2']: flags['in_stage2'] = flags['volume_spike']

    except Exception as e: log.error(f"Error calculating flags index {idx}: {e}", exc_info=False); return {}

    setup_type = _determine_setup_type(flags)

    # Composite Score (logic unchanged)
    score = 0.0; score += 1.0 if flags.get('above_ma50', False) else 0.0; score += 1.0 if flags.get('above_ma200', False) else 0.0
    score += 1.0 if flags.get('volume_spike', False) else 0.0; score += 1.0 if flags.get('rsi_bullish', False) else 0.0
    score += 1.0 if (flags.get('in_base', False) or flags.get('compressed', False)) else 0.0
    score += 1.0 if flags.get('rs_rising', False) else 0.0
    if setup_type == 'STAGE2': score += 2.0
    elif setup_type == 'MA_CROSS': score += 1.0
    elif setup_type == 'LEADER': score += 1.0

    # Calculate Risk/Reward using Capitalized row data
    stop, target, r_multiple, risk_pct = _calculate_risk_reward(row, setup_type)

    # Assemble Output Dict with CAPITALIZED keys
    score_data = {
        'date': row.name, 'Close': row.get('Close'), 'Open': row.get('Open'), 'High': row.get('High'), 'Low': row.get('Low'),
        'Volume': row.get('Volume'), 'Adj Close': row.get('Adj Close'),
        'above_ma50': flags.get('above_ma50', False), 'above_ma200': flags.get('above_ma200', False),
        'compressed': flags.get('compressed', False), 'volume_spike': flags.get('volume_spike', False),
        'momentum_up': flags.get('momentum_up', False), 'in_base': flags.get('in_base', False),
        'rsi_bullish': flags.get('rsi_bullish', False), 'rs_rising': flags.get('rs_rising', False),
        'ma_cross': flags.get('ma_cross', False), 'in_stage2': flags.get('in_stage2', False),
        'MA10': row.get('MA10'), 'MA20': row.get('MA20'), 'MA50': row.get('MA50'), 'MA200': row.get('MA200'),
        'ATR': row.get('ATR'), 'RSI': row.get('RSI'), 'RS_Line': row.get('RS_Line'), 'RS_slope': row.get('RS_slope'),
        'setup_type': setup_type, 'score': round(score, 2), 'stop': stop,
        'target': target, 'r_multiple': r_multiple, 'risk_pct': risk_pct }
    # Clean None/NaN values
    score_data_cleaned = {k: v for k, v in score_data.items() if v is not None and pd.notna(v)}
    # Add back essential keys if they were None/NaN
    for essential in ['date', 'Close', 'setup_type', 'score', 'stop', 'target', 'r_multiple', 'risk_pct']:
         if essential not in score_data_cleaned: score_data_cleaned[essential] = score_data.get(essential)
    return score_data_cleaned


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Scores latest row. Expects df with CAPITALIZED columns."""
    if df.empty or len(df) < 2: log.debug("score_dataframe: Input empty/short."); return pd.DataFrame()
    try:
        latest_score_data_dict = score_symbol(df, idx=-1) # Pass original Capitalized df
        if not latest_score_data_dict: log.debug("score_dataframe: Scoring failed."); return pd.DataFrame()

        # Filter results
        score = latest_score_data_dict.get('score', 0.0); setup_type = latest_score_data_dict.get('setup_type', 'NONE')
        r_multiple = latest_score_data_dict.get('r_multiple')
        is_valid_setup = setup_type != 'NONE'; meets_min_score = score >= MIN_SCORE_THRESHOLD
        meets_min_r = (r_multiple is not None) and (r_multiple >= DEFAULT_TARGET_MIN_MULT)

        if is_valid_setup and meets_min_score and meets_min_r:
            log.debug(f"Valid setup: Score={score:.2f}, R={r_multiple:.2f}, Type={setup_type}")
            result_df = pd.DataFrame([latest_score_data_dict])
            if 'date' in result_df.columns: result_df = result_df.set_index('date')
            return result_df
        else:
            log.debug(f"Latest row failed criteria: Valid={is_valid_setup}, ScoreOK={meets_min_score}({score:.2f}), R_OK={meets_min_r}({r_multiple})")
            return pd.DataFrame()
    except Exception as e: log.error(f"Error scoring/filtering DataFrame: {e}", exc_info=True); return pd.DataFrame()