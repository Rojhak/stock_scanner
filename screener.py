# leader_scan/screener.py
"""
Dedicated scanner for finding stocks potentially entering Stage 2 after a pullback,
based on Fibonacci retracements and volume confirmation.

Note: This logic is also partially integrated into the main scorer.py. This
file can serve as a standalone runner for just this specific setup or be deprecated
if the main scorer/advanced_screener covers the use case sufficiently.
"""
import os
import sys
import pandas as pd
import numpy as np
import traceback
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from typing import Union

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not found. Install with `pip install yfinance`.", file=sys.stderr)
    yf = None
try:
    from scipy.signal import argrelextrema
except ImportError:
    print("Warning: scipy not found. Install with `pip install scipy`.", file=sys.stderr)
    argrelextrema = None

# Ensure correct relative imports
try:
    from .config import CONFIG # Assuming config handles its loading
    from .data import get_price_data, load_universe # Use robust data functions
except ImportError:
    print("Error: Could not import config/data modules from leader_scan.", file=sys.stderr)
    # Fallback config or exit
    CONFIG = {"drawdown_max": 0.25, "volume_thrust_multiple": 1.4} # Minimal fallback
    sys.exit(1) # Exit if core components missing

# Define base directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent

# --- Helper Functions ---

def safe_scalar(value: Union[pd.Series, np.number, float, int, None]) -> Optional[float]:
    """Safely extract a scalar float value from various inputs."""
    if isinstance(value, pd.Series):
        if len(value) > 0 and pd.notna(value.iloc[0]):
            try:
                return float(value.iloc[0])
            except (ValueError, TypeError):
                return None
        else:
            return None
    elif isinstance(value, (np.number, float, int)) and pd.notna(value):
        return float(value)
    else:
        return None

def compute_sma_slope(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Compute the SMA and its slope for the given window. Handles NaNs."""
    df_out = df.copy()
    sma_col = f"SMA_{window}"
    slope_col = f"SMA_slope_{window}"
    try:
        if 'Close' not in df_out.columns:
            print(f"Warning: 'Close' column missing for SMA calculation.", file=sys.stderr)
            df_out[sma_col] = np.nan
            df_out[slope_col] = np.nan
            return df_out

        # Calculate SMA, handling potential NaNs introduced by rolling window
        df_out[sma_col] = df_out["Close"].rolling(window, min_periods=window // 2).mean()

        # Calculate slope: difference between current SMA and SMA 'window-1' periods ago
        # More robust slope calculation: change over the last N periods (e.g., 10)
        n_slope = 10 # Lookback for slope calculation
        df_out[slope_col] = df_out[sma_col].diff(n_slope)

    except Exception as e:
        print(f"Error computing SMA/slope (window {window}): {e}", file=sys.stderr)
        df_out[sma_col] = np.nan
        df_out[slope_col] = np.nan
    return df_out

def filter_downtrend(df: pd.DataFrame, window: int = 200, min_days: int = 100) -> bool:
    """Check if the stock is in a confirmed downtrend."""
    sma_col = f"SMA_{window}"
    slope_col = f"SMA_slope_{window}" # Using the N-period slope

    if sma_col not in df.columns or slope_col not in df.columns or 'Close' not in df.columns:
        # print(f"Warning: Missing columns for downtrend check ({sma_col}, {slope_col}, Close).")
        return False

    if len(df) < min_days:
        return False

    try:
        # Consider the last 'min_days' for the trend check
        tail_df = df.tail(min_days)

        # Check conditions (SMA slope negative, Close below SMA)
        # Requires the *majority* of the lookback period to satisfy conditions
        is_slope_down = tail_df[slope_col] < 0
        is_below_sma = tail_df["Close"] < tail_df[sma_col]

        # More robust check: majority of days meet the criteria
        required_true_ratio = 0.75 # e.g., 75% of the days must meet criteria
        slope_down_ratio = is_slope_down.mean() # Mean of boolean series gives ratio of True
        below_sma_ratio = is_below_sma.mean()

        # Check if *both* conditions are met sufficiently often
        return slope_down_ratio >= required_true_ratio and below_sma_ratio >= required_true_ratio

    except Exception as e:
        print(f"Error in filter_downtrend: {e}", file=sys.stderr)
        return False

def find_swings(df: pd.DataFrame, lookback: int = 20) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Find the last significant high and low points."""
    if argrelextrema is None:
        print("Error: scipy.signal.argrelextrema not available.", file=sys.stderr)
        return None, None
    if 'Close' not in df.columns or len(df) < lookback * 2: # Need enough data
        return None, None

    try:
        prices = df["Close"].values
        # Find relative maxima and minima indices
        high_indices = argrelextrema(prices, np.greater_equal, order=lookback)[0]
        low_indices = argrelextrema(prices, np.less_equal, order=lookback)[0]

        if low_indices.size > 0 and high_indices.size > 0:
            # Find the index of the most recent low
            last_low_idx = low_indices[-1]

            # Find the most recent high *before* the last low
            relevant_highs = high_indices[high_indices < last_low_idx]
            if relevant_highs.size > 0:
                last_high_idx = relevant_highs[-1]
                # Return the corresponding dates (timestamps) from the DataFrame index
                return df.index[last_high_idx], df.index[last_low_idx]

    except Exception as e:
        print(f"Error finding swings: {e}", file=sys.stderr)

    return None, None # Return None if swings aren't found

def compute_fib_levels(df: pd.DataFrame, high_idx: pd.Timestamp, low_idx: pd.Timestamp) -> Optional[Dict[str, float]]:
    """Compute Fibonacci retracement levels based on swing high/low dates."""
    try:
        # Get Close prices at the specific dates (indices)
        high_price = safe_scalar(df.loc[high_idx, "Close"])
        low_price = safe_scalar(df.loc[low_idx, "Close"])

        if high_price is None or low_price is None or high_price <= low_price:
            # print(f"Invalid high/low prices for Fib calculation: High={high_price}, Low={low_price}")
            return None

        price_diff = high_price - low_price
        return {
            "high_price": high_price,
            "low_price": low_price,
            "38.2": round(high_price - 0.382 * price_diff, 4),
            "50.0": round(high_price - 0.500 * price_diff, 4),
            "61.8": round(high_price - 0.618 * price_diff, 4),
        }
    except KeyError:
        print(f"Error accessing price data at index {high_idx} or {low_idx} for Fib calculation.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error computing Fibonacci levels: {e}", file=sys.stderr)
        return None

def check_stage2_entry(df: pd.DataFrame, fib_levels: Dict[str, float]) -> bool:
    """Check if the current price is within the 38.2% - 61.8% Fib retracement zone."""
    if not fib_levels or '38.2' not in fib_levels or '61.8' not in fib_levels:
        return False
    try:
        current_price = safe_scalar(df["Close"].iloc[-1])
        fib_38_2 = fib_levels["38.2"]
        fib_61_8 = fib_levels["61.8"]

        if current_price is None:
            return False

        # Check if price is between the two levels (inclusive)
        return fib_61_8 <= current_price <= fib_38_2
    except IndexError:
        print("Error: Cannot access last row price for Stage 2 check (DataFrame likely empty).", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error in check_stage2_entry: {e}", file=sys.stderr)
        return False

def filter_volume_thrust(df: pd.DataFrame, window: int = 50, multiplier: float = CONFIG.get("volume_thrust_multiple", 1.4)) -> bool:
    """Check if the latest volume is significantly above its moving average."""
    if 'Volume' not in df.columns or len(df) < window + 1:
        return False
    try:
        # Calculate average volume excluding the latest bar
        avg_vol = df["Volume"].iloc[-window-1:-1].mean()
        latest_vol = safe_scalar(df["Volume"].iloc[-1])

        if latest_vol is None or pd.isna(avg_vol) or avg_vol == 0:
            return False

        return latest_vol >= avg_vol * multiplier
    except IndexError:
         print("Error: Cannot access volume data for volume filter (DataFrame too short?).", file=sys.stderr)
         return False
    except Exception as e:
        print(f"Error in filter_volume_thrust: {e}", file=sys.stderr)
        return False

def calculate_stage2_score(df: pd.DataFrame, fib_levels: Dict[str, float], vol_window: int = 50) -> float:
    """Calculate a score (0-100) indicating the quality of the Stage 2 setup."""
    score = 0.0
    if not fib_levels:
        return score

    try:
        price = safe_scalar(df["Close"].iloc[-1])
        if price is None: return score

        middle_fib = fib_levels.get("50.0", 0.0)
        fib_38_2 = fib_levels.get("38.2", 0.0)
        fib_61_8 = fib_levels.get("61.8", 0.0)
        fib_range = fib_38_2 - fib_61_8

        # 1. Retracement Score (Max 50 points, highest near 50% Fib)
        if fib_range > 0:
            # Normalize distance from 50% level (0=at 50%, 1=at 38.2/61.8)
            dist_from_mid = abs(price - middle_fib) / (fib_range / 2.0)
            retracement_score = max(0.0, 50.0 * (1.0 - dist_from_mid))
            score += retracement_score

        # 2. Volume Score (Max 30 points)
        avg_vol = df["Volume"].rolling(vol_window, min_periods=vol_window//2).mean().iloc[-1]
        current_vol = safe_scalar(df["Volume"].iloc[-1])
        if current_vol is not None and not pd.isna(avg_vol) and avg_vol > 0:
            volume_ratio = current_vol / avg_vol
            # Scale score based on ratio (e.g., 1x avg = 10 pts, 2x avg = 20 pts, >=3x avg = 30 pts)
            volume_score = min(30.0, 10.0 * volume_ratio)
            score += volume_score

        # 3. Recent Momentum Score (Max 20 points - e.g., price > MA10)
        if 'Close' in df.columns and 'MA10' not in df.columns: # Calculate MA10 if missing
             df['MA10'] = df['Close'].rolling(10, min_periods=5).mean()

        ma10 = safe_scalar(df["MA10"].iloc[-1]) if 'MA10' in df.columns else None
        if ma10 is not None and price > ma10:
            score += 20.0

    except Exception as e:
        print(f"Error calculating stage2 score: {e}", file=sys.stderr)
        # traceback.print_exc() # Uncomment for debugging
        return 0.0 # Return 0 on error

    return round(score, 2)


# --- Main Screening Function ---

def screen_stage2_candidates(
    tickers: List[str],
    period: str = "2y", # Data download period
    interval: str = "1d",
    downtrend_window: int = 200,
    downtrend_days: int = 100,
    swing_lookback: int = 20,
    vol_window: int = 50, # Consistent with volume thrust filter
    vol_mult: float = CONFIG.get("volume_thrust_multiple", 1.4)
) -> List[Tuple[str, float, Optional[Dict]]]:
    """
    Screen tickers for Stage 2 pullback setups with volume confirmation.

    Returns:
        List of tuples: (ticker, score, fib_levels_dict or None)
    """
    if yf is None or argrelextrema is None:
        print("Error: Missing yfinance or scipy. Cannot run Stage 2 scan.", file=sys.stderr)
        return []

    print(f"Starting Stage 2 scan for {len(tickers)} tickers...")
    # Use robust data fetching
    data = get_price_data(tickers, period=period, interval=interval)

    if data is None or data.empty:
        print("Error: Failed to fetch any price data for Stage 2 scan.", file=sys.stderr)
        return []

    candidates = []
    symbols_in_data = data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else tickers

    for t in symbols_in_data:
        try:
            # Extract symbol data
            if isinstance(data.columns, pd.MultiIndex):
                df = data[t].copy()
            else:
                df = data.copy() if t == symbols_in_data[0] else pd.DataFrame()

            df.columns = [col.strip() for col in df.columns] # Clean column names
            df = df.dropna(subset=['Close']) # Ensure Close price exists

            if len(df) < max(downtrend_window, downtrend_days, swing_lookback * 2):
                continue # Skip if insufficient data

            # 1. Check for prior downtrend
            df = compute_sma_slope(df, downtrend_window)
            if not filter_downtrend(df, downtrend_window, downtrend_days):
                continue

            # 2. Find relevant swing points
            high_idx, low_idx = find_swings(df, swing_lookback)
            if not high_idx or not low_idx:
                continue

            # 3. Calculate Fibonacci levels
            fib_levels = compute_fib_levels(df, high_idx, low_idx)
            if not fib_levels:
                continue

            # 4. Check if current price is in the Stage 2 entry zone
            if not check_stage2_entry(df, fib_levels):
                continue

            # 5. Check for volume confirmation (thrust)
            if not filter_volume_thrust(df, vol_window, vol_mult):
                continue

            # If all conditions met, calculate score and add to candidates
            score = calculate_stage2_score(df, fib_levels, vol_window)
            if score > 50: # Set a minimum quality score threshold
                 # Include Fib levels in output for potential analysis/plotting
                 candidates.append((t, score, fib_levels))

        except Exception as e:
            print(f"Error processing {t} for Stage 2: {e}", file=sys.stderr)
            # traceback.print_exc() # Uncomment for debugging
            continue

    # Sort candidates by score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"Stage 2 scan complete. Found {len(candidates)} potential candidates.")
    return candidates


def find_new_stage2_stocks(
    tickers: Optional[Union[List[str], pd.DataFrame]] = None,
    top: Optional[int] = 5, # Default to return top 5
    **kwargs # Pass other parameters to screen_stage2_candidates
) -> pd.DataFrame:
    """
    Wrapper to find new Stage 2 stocks, load tickers if needed, and return a DataFrame.
    """
    tickers_list: List[str] = []
    if tickers is None:
        # Load default universe if none provided
        default_universe = CONFIG.get("universe", "sp500")
        print(f"No tickers provided, loading default universe: {default_universe}")
        try:
            # Use the robust loader from data.py
            tickers_list = load_universe(default_universe)
        except ValueError as e:
             print(f"Error loading default universe {default_universe}: {e}. Aborting Stage 2 scan.", file=sys.stderr)
             return pd.DataFrame()
    elif isinstance(tickers, pd.DataFrame):
        # Handle DataFrame input (e.g., from main scanner)
        if 'symbol' in tickers.columns:
            tickers_list = tickers['symbol'].astype(str).unique().tolist()
        elif tickers.index.name is not None and tickers.index.name.lower() == 'ticker':
             tickers_list = tickers.index.astype(str).unique().tolist()
        else:
             print("Warning: Cannot determine tickers from DataFrame input.", file=sys.stderr)
             return pd.DataFrame()
    elif isinstance(tickers, list):
        tickers_list = list(set(str(t) for t in tickers)) # Ensure unique strings
    else:
        print(f"Error: Invalid type for 'tickers': {type(tickers)}", file=sys.stderr)
        return pd.DataFrame()

    if not tickers_list:
         print("No valid tickers to scan for Stage 2 setups.", file=sys.stderr)
         return pd.DataFrame()

    # Get scored candidates list [ (ticker, score, fib_dict), ... ]
    candidates_list = screen_stage2_candidates(tickers_list, **kwargs)

    # Create DataFrame from the list of tuples
    if candidates_list:
        # Extract data, handling potential None for fib_dict
        symbols = [c[0] for c in candidates_list]
        scores = [c[1] for c in candidates_list]
        # Optionally extract key fib levels if needed in the output DataFrame
        fib_61_8 = [c[2]['61.8'] if c[2] else np.nan for c in candidates_list]
        fib_38_2 = [c[2]['38.2'] if c[2] else np.nan for c in candidates_list]

        df = pd.DataFrame({
            'symbol': symbols,
            'stage2_score': scores,
            'fib_61.8': fib_61_8,
            'fib_38.2': fib_38_2
        })
        # Ensure score column name is distinct if merging later
    else:
        df = pd.DataFrame({'symbol': [], 'stage2_score': []}) # Return empty DF with columns

    # Limit to top N if specified
    if top is not None and len(df) > 0:
        # Already sorted by screen_stage2_candidates
        return df.head(top).reset_index(drop=True)
    elif len(df) > 0:
        return df.reset_index(drop=True) # Return all if top is None
    else:
        return df # Return empty DF


# --- Example Execution ---
if __name__ == "__main__":
    print("Running Standalone Stage 2 Screener...")
    # Load tickers from a default source (e.g., SP500)
    try:
         # Use robust loader from data.py
         default_tickers = load_universe(CONFIG.get("universe", "sp500"))
    except ValueError as e:
         print(f"Failed to load default tickers: {e}", file=sys.stderr)
         default_tickers = ["AAPL", "MSFT", "GOOGL"] # Fallback

    if default_tickers:
        # Get top 5 candidates using default parameters
        stage2_hits = find_new_stage2_stocks(default_tickers, top=5)

        if not stage2_hits.empty:
            print("\nTop 5 Stage 2 + High-Volume Candidates:")
            # Adjust printing based on columns available in stage2_hits DataFrame
            print(stage2_hits.to_string(index=False))
        else:
            print("\nNo Stage 2 + High-Volume candidates found matching criteria.")
    else:
        print("Could not load any tickers to scan.")
