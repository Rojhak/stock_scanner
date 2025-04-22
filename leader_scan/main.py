# leader_scan/main.py
"""
Main orchestrator for the Leader Scan package.
Provides high-level functions to run scans and the LeadershipScanner class.
Can scan single universes or all universes found in the resources directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
import logging
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path # Import needed
import os
import numpy as np

# Setup basic logging
log_level = logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
log = logging.getLogger(__name__)

# --- Imports from within the package ---
try:
    from .config import CONFIG
    from .data import get_price_data, get_fundamentals, load_universe
    from .scorer import score_dataframe, score_symbol # Scorer expects Capitalized cols
    from .alert import dispatch
    from .indicators import atr, ma, ema, rs_line, rs_new_high # Indicators expect lowercase cols
except ImportError as e:
    log.critical(f"Failed to import necessary modules from leader_scan package: {e}. Exiting.")
    sys.exit(1)

# Define package root relative to this file for resource loading
_PACKAGE_ROOT = Path(__file__).resolve().parent

# --- Internal Helper Functions ---

def _calculate_required_indicators(df: pd.DataFrame, bench_close: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Calculate indicators. Input df has Capitalized columns.
    Standardizes to lowercase internally, then renames back to Capitalized.
    """
    if df.empty: return df
    df_out = df.copy()
    df_out.columns = [str(col).lower() for col in df_out.columns]
    log.debug(f"Calculating indicators. Lowercase columns: {df_out.columns.tolist()}")
    required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
    close_col, high_col, low_col = 'close', 'high', 'low'
    if not all(col in df_out.columns for col in required_cols_lower):
         missing_cols = [rc for rc in required_cols_lower if rc not in df_out.columns]
         log.warning(f"Missing required OHLCV columns (lowercase): {missing_cols}.")

    try:
        # Moving Averages
        for window in [10, 20, 50, 200]:
            ma_col = f"ma{window}"
            if ma_col not in df_out.columns:
                if close_col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[close_col]): df_out[ma_col] = ma(df_out[close_col], window)
                else: df_out[ma_col] = np.nan
        # ATR
        atr_col = 'atr'
        if atr_col not in df_out.columns:
            if all(c in df_out.columns for c in [high_col, low_col, close_col]) and not df_out[[high_col, low_col, close_col]].isnull().all().all():
                 ohlc_data_for_atr = df_out[[high_col, low_col, close_col]].dropna()
                 if not ohlc_data_for_atr.empty: df_out[atr_col] = atr(ohlc_data_for_atr, window=14).reindex(df_out.index)
                 else: df_out[atr_col] = np.nan
            else: df_out[atr_col] = np.nan
        # RSI
        rsi_col = 'rsi'
        if rsi_col not in df_out.columns:
            if close_col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[close_col]) and not df_out[close_col].isnull().all():
                 delta = df_out[close_col].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
                 ma_up = up.ewm(com=13, adjust=False).mean(); ma_down = down.ewm(com=13, adjust=False).mean()
                 rs = ma_up / ma_down.replace(0, np.nan)
                 df_out[rsi_col] = 100.0 - (100.0 / (1.0 + rs)); df_out[rsi_col] = df_out[rsi_col].fillna(50)
            else: df_out[rsi_col] = np.nan
        # Relative Strength
        rs_line_col = 'rs_line'; rs_slope_col = 'rs_slope'
        if bench_close is not None and not bench_close.empty and close_col in df_out.columns and not df_out[close_col].isnull().all():
             log.debug(f"Calculating RS_Line using benchmark data...")
             if rs_line_col not in df_out.columns:
                  aligned_bench = bench_close.reindex(df_out.index)
                  if not aligned_bench.isnull().all(): df_out[rs_line_col] = rs_line(df_out[close_col], aligned_bench, smooth=5)
                  else: df_out[rs_line_col] = np.nan; log.debug("Aligned benchmark all NaNs.")
             else: df_out[rs_line_col].fillna(method='ffill', inplace=True)
             if rs_line_col in df_out.columns and rs_slope_col not in df_out.columns:
                  if not df_out[rs_line_col].isnull().all(): df_out[rs_slope_col] = df_out[rs_line_col].diff(5)
                  else: df_out[rs_slope_col] = np.nan
        else:
            log.debug(f"Skipping RS Calcs (Benchmark valid: {bench_close is not None and not bench_close.empty}, Close valid: {close_col in df_out.columns and not df_out[close_col].isnull().all()})")
            df_out[rs_line_col] = np.nan; df_out[rs_slope_col] = np.nan
        log.debug(f"Finished indicators. Lowercase cols: {df_out.columns.tolist()}")
    except Exception as e: log.warning(f"Error indicator calc: {e}", exc_info=True)

    # --- Rename columns back to Capitalized ---
    rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'adj close': 'Adj Close',
                  'ma10': 'MA10', 'ma20': 'MA20', 'ma50': 'MA50', 'ma200': 'MA200', 'atr': 'ATR', 'rsi': 'RSI',
                  'rs_line': 'RS_Line', 'rs_slope': 'RS_slope'}
    final_rename_map = {k: v for k, v in rename_map.items() if k in df_out.columns}
    df_out.rename(columns=final_rename_map, inplace=True)
    log.debug(f"Cols after renaming back: {df_out.columns.tolist()}")
    return df_out

# Corrected _fetch_and_prepare_data with robust benchmark handling
def _fetch_and_prepare_data(symbols: List[str], benchmark: str = "SPY", period: str = '2y', interval: str = '1d', universe_name: Optional[str] = None) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[pd.Series]]:
    """Fetches price data, prepares benchmark close. Handles MultiIndex benchmark."""
    log.info(f"Fetching data: {len(symbols)} symbols (Universe: {universe_name or 'N/A'}), Benchmark: {benchmark}...")
    today = dt.date.today(); start_date = today - dt.timedelta(days=CONFIG.get("cache_days", 730))
    # --- This is the CORRECTED code block ---
    if 'y' in period:
        try:
            years = int(period.replace('y', ''))
            start_date = today - dt.timedelta(days=years*365)
        except ValueError:
            # Handle error or do nothing
            pass
    elif 'm' in period:
        try:
            months = int(period.replace('m', ''))
            # Using 30 days per month is an approximation, consider if more precision is needed
            start_date = today - dt.timedelta(days=months*30)
        except ValueError:
            # Handle error or do nothing
            pass
    # --- End of CORRECTED code block ---

    # Fetch Benchmark Data
    bench_df = get_price_data(benchmark, start_date=start_date, end_date=today, interval=interval, universe_name_for_cache=benchmark)
    bench_close = None
    if bench_df is None or bench_df.empty: log.warning(f"Failed fetch ({benchmark}).")
    else:
        log.debug(f"Benchmark ({benchmark}) cols: {bench_df.columns}"); log.debug(f"Benchmark ({benchmark}) head:\n{bench_df.head(2)}")
        close_col_id = None
        if isinstance(bench_df.columns, pd.MultiIndex):
             log.debug("Benchmark df has MultiIndex cols.")
             if benchmark in bench_df.columns.get_level_values(0):
                 price_level_values = bench_df[benchmark].columns
                 log.debug(f"Benchmark price level columns: {price_level_values}")
                 found_col_name = next((c for c in price_level_values if str(c) == 'Close'), None) or \
                                  next((c for c in price_level_values if str(c).lower() == 'close'), None)
                 if found_col_name: close_col_id = (benchmark, found_col_name); log.debug(f"Identified MultiIndex Close ID: {close_col_id}")
                 else: log.warning(f"No 'Close'/'close' in price level of benchmark MultiIndex.")
             else: log.warning(f"Benchmark ticker {benchmark} not in MultiIndex level 0.")
        else: # Flat columns
             log.debug("Benchmark df has flat columns.")
             found_col_name = next((c for c in bench_df.columns if str(c) == 'Close'), None) or \
                              next((c for c in bench_df.columns if str(c).lower() == 'close'), None)
             if found_col_name: close_col_id = found_col_name; log.debug(f"Identified flat Close ID: {close_col_id}")
             else: log.warning("No 'Close'/'close' in flat benchmark columns.")

        if close_col_id is not None:
            try:
                log.debug(f"Attempting extraction using ID: {close_col_id}")
                extracted_series = bench_df[close_col_id]
                log.debug(f"Extracted series type: {type(extracted_series)}, dtype: {extracted_series.dtype}")
                log.debug(f"Extracted series head:\n{extracted_series.head(3)}")
                is_null = extracted_series.isnull().all(); is_numeric = pd.api.types.is_numeric_dtype(extracted_series)
                log.debug(f"Validation: Is Null = {is_null}, Is Numeric = {is_numeric}")
                if is_null: log.warning(f"Benchmark column ({close_col_id}) all NaNs."); bench_close = None
                elif not is_numeric: log.warning(f"Benchmark column ({close_col_id}) not numeric."); bench_close = None
                else: bench_close = extracted_series; log.info(f"OK: Benchmark Close extracted from: {close_col_id}")
            except Exception as e: log.error(f"Error extracting benchmark column ({close_col_id}): {e}", exc_info=True); bench_close = None
        else: log.error(f"Could not find 'Close' identifier in benchmark {benchmark}.")

    if bench_close is None: log.warning(f"Proceeding without benchmark data. RS calcs skipped.")

    # Fetch Symbol Data
    all_data = get_price_data(symbols, start_date=start_date, end_date=today, interval=interval, universe_name_for_cache=universe_name)
    if all_data is None or all_data.empty: log.error(f"Failed symbol fetch for universe '{universe_name}'."); return None, bench_close

    # Structure Symbol Data
    data_dict: Dict[str, pd.DataFrame] = {}; successful_tickers = []
    def standardize_columns(df):
        new_cols = []; df.columns = [str(col).strip() for col in df.columns]
        for col in df.columns:
            if isinstance(col, str): new_cols.append(col.capitalize())
            else: new_cols.append(col)
        df.columns = new_cols
        return df

    if isinstance(all_data.columns, pd.MultiIndex):
        successful_tickers = list(all_data.columns.levels[0])
        log.debug(f"Data fetched for: {', '.join(successful_tickers)} in universe '{universe_name}'")
        for sym in successful_tickers:
            if sym in symbols:
                 try:
                     df_sym = all_data[sym].copy(); df_sym = standardize_columns(df_sym)
                     if 'Close' in df_sym.columns: data_dict[sym] = df_sym.dropna(subset=['Close'])
                     else: log.warning(f"Symbol {sym} missing 'Close'. Cols: {df_sym.columns}")
                 except Exception as e: log.error(f"Error processing {sym}: {e}")
    elif not all_data.empty and len(symbols) == 1 and symbols[0] in symbols:
         sym = symbols[0]; successful_tickers.append(sym); df_sym = all_data.copy()
         df_sym = standardize_columns(df_sym)
         if 'Close' in df_sym.columns: data_dict[sym] = df_sym.dropna(subset=['Close'])
         else: log.warning(f"Single symbol {sym} missing 'Close'. Cols: {df_sym.columns}")
    else: log.warning(f"Unexpected data structure/empty result for universe '{universe_name}'. Cols: {all_data.columns}")
    log.info(f"OK: Prepared data for {len(data_dict)} symbols in universe '{universe_name}'.")
    return data_dict, bench_close


def _process_and_score_symbol(symbol: str, df: pd.DataFrame, bench_close: Optional[pd.Series]) -> Optional[pd.Series]:
    """Calculates indicators and scores a single symbol's DataFrame."""
    log.debug(f"Processing symbol: {symbol}")
    required_rows = CONFIG.get("min_data_rows", 50)
    if df.empty or len(df) < required_rows: log.debug(f"Skipping {symbol}: Insufficient data."); return None
    try:
        df_with_indicators = _calculate_required_indicators(df, bench_close) # Returns Capitalized
        if 'ATR' not in df_with_indicators or df_with_indicators['ATR'].isnull().all(): log.warning(f"ATR failed {symbol}, skipping scoring."); return None

        scored_results = score_dataframe(df_with_indicators) # Scorer expects Capitalized
        if not scored_results.empty:
            if isinstance(scored_results, pd.DataFrame): latest_result = scored_results.iloc[-1].copy(); latest_result['date'] = scored_results.index[-1]
            elif isinstance(scored_results, pd.Series): latest_result = scored_results.copy(); latest_result['date'] = scored_results.name
            else: log.error(f"Unexpected type from score_dataframe: {type(scored_results)}"); return None
            latest_result['symbol'] = symbol
            log.debug(f"Valid setup found for {symbol} on {latest_result['date']} score {latest_result.get('score', np.nan):.2f}")
            return latest_result
        else: log.debug(f"No valid setup for {symbol} after scoring."); return None
    except Exception as e: log.error(f"Error processing {symbol}: {e}", exc_info=False); return None

# --- Public API ---
class LeadershipScanner:
    """Orchestrates the stock scanning process."""
    def __init__(self, universe: str = CONFIG.get("universe", "sp500"), benchmark: str = "SPY"):
        log.info(f"Initializing Scanner: Universe='{universe}', Benchmark='{benchmark}'.")
        self.universe_name = universe; self.benchmark_symbol = benchmark; self.symbols: List[str] = []
        self.data: Dict[str, pd.DataFrame] = {}; self.benchmark_close: Optional[pd.Series] = None; self.results: Optional[pd.DataFrame] = None
        try: self.symbols = load_universe(self.universe_name)
        except ValueError as e: log.error(f"Failed load '{self.universe_name}': {e}")
        if not self.symbols: log.warning(f"No symbols loaded for universe '{self.universe_name}'.")

    def run(self, top: int = 20, min_score: float = 3.0, min_r: Optional[float] = None) -> pd.DataFrame:
        log.info(f"Starting scan run for universe '{self.universe_name}'...")
        if not self.symbols: log.warning("No symbols loaded."); return pd.DataFrame()
        data_period = CONFIG.get("cache_days", 730); data_interval = CONFIG.get("price_interval", "1d")
        fetched_data, benchmark_close_series = _fetch_and_prepare_data(
            self.symbols, self.benchmark_symbol, period=f"{data_period}d",
            interval=data_interval, universe_name=self.universe_name )
        self.benchmark_close = benchmark_close_series # Store benchmark data (or None)

        if not fetched_data: log.error("Data fetching failed."); return pd.DataFrame()
        self.data = fetched_data; all_results = []
        log.info(f"Scoring {len(self.data)} symbols for universe '{self.universe_name}'...")
        for symbol, df in self.data.items():
             result_series = _process_and_score_symbol(symbol, df, self.benchmark_close)
             if result_series is not None: all_results.append(result_series)
        if not all_results: log.info(f"No qualifying setups for '{self.universe_name}'."); self.results = pd.DataFrame(); return self.results
        try:
            results_df = pd.DataFrame(all_results)
            if 'date' in results_df.columns: results_df = results_df.set_index('date')
            if 'symbol' not in results_df.columns and results_df.index.name != 'symbol': log.warning("'symbol' column missing.") # Check if index is symbol
            elif results_df.index.name == 'symbol': results_df.reset_index(inplace=True) # Make symbol a column if index
        except Exception as e: log.error(f"Error creating results DF: {e}", exc_info=True); return pd.DataFrame()
        log.info(f"Found {len(results_df)} potential setups for '{self.universe_name}'.")
        final_min_r = min_r if min_r is not None else CONFIG.get("min_r_multiple", 2.5)
        if 'score' not in results_df.columns: results_df['score'] = np.nan
        if 'r_multiple' not in results_df.columns: results_df['r_multiple'] = np.nan
        filtered_df = results_df[(results_df['score'].fillna(0) >= min_score) & (results_df['r_multiple'].fillna(0) >= final_min_r)].copy()
        if filtered_df.empty: log.info(f"No stocks met final criteria for '{self.universe_name}'."); self.results = pd.DataFrame(); return self.results
        sort_cols = ['score', 'r_multiple']; ascending_order = [False, False]
        if 'score' not in filtered_df.columns: sort_cols.remove('score'); ascending_order.pop(0)
        if 'r_multiple' not in filtered_df.columns: sort_cols.remove('r_multiple'); ascending_order.pop(1)
        if not sort_cols: leaders = filtered_df.head(top)
        else: leaders = filtered_df.sort_values(sort_cols, ascending=ascending_order).head(top)
        log.info(f"Scan complete for '{self.universe_name}'. Identified {len(leaders)} leaders.")
        self.results = leaders; return leaders.reset_index() # Always return with symbol col

def run_daily_scan(
    universe: str = CONFIG.get("universe", "sp500"), top: int = 20, alert: bool = False,
    silent: bool = False, benchmark: str = "SPY") -> pd.DataFrame:
    """Convenience one-liner function to run the daily scan for a SINGLE universe."""
    log.info(f"Executing run_daily_scan: universe='{universe}', top={top}, alert={alert}, silent={silent}, benchmark={benchmark}")
    scanner = LeadershipScanner(universe=universe, benchmark=benchmark); leaders_df = scanner.run(top=top)
    if leaders_df is not None and not leaders_df.empty:
        if not silent:
            print(f"\n--- Leader Scan Results ({universe.upper()}) ---")
            display_cols = ['symbol', 'date', 'setup_type', 'score', 'r_multiple', 'Close', 'stop', 'target']
            if 'date' not in leaders_df.columns and isinstance(leaders_df.index, (pd.DatetimeIndex, pd.Index)): leaders_df.reset_index(inplace=True)
            if 'Close' not in leaders_df.columns:
                 closes = []; date_col = 'date'
                 if date_col not in leaders_df.columns: log.error("No 'date' col for Close lookup."); closes = [np.nan] * len(leaders_df)
                 else:
                     for idx in leaders_df.index:
                         row = leaders_df.loc[idx]; sym = row['symbol']; event_date = pd.to_datetime(row[date_col])
                         close_val = np.nan
                         try:
                             if sym in scanner.data and event_date in scanner.data[sym].index:
                                 close_col_lookup = 'Close' # Expect Capitalized
                                 if close_col_lookup in scanner.data[sym].columns: close_val = scanner.data[sym].loc[event_date, close_col_lookup]
                                 else: log.debug(f"Col 'Close' not found for {sym}")
                             closes.append(close_val)
                         except Exception as e: log.debug(f"Could not get cached Close {sym} {event_date}: {e}"); closes.append(np.nan)
                 leaders_df['Close'] = closes
            final_display_cols = []
            for col in display_cols:
                if col in leaders_df.columns: final_display_cols.append(col)
                else: log.warning(f"Display column '{col}' missing."); leaders_df[col] = pd.NA; final_display_cols.append(col)
            if 'symbol' not in leaders_df.columns: log.error("CRITICAL: 'symbol' missing."); return leaders_df
            if 'date' in leaders_df.columns: leaders_df['date'] = pd.to_datetime(leaders_df['date']).dt.strftime('%Y-%m-%d')
            print(leaders_df[final_display_cols].to_string(index=False, float_format="%.2f"))
        if alert:
            log.info("Dispatching alerts..."); today_str = dt.date.today().strftime('%Y-%m-%d'); subject = f"Leader Scan ({today_str}) - {len(leaders_df)} Matches ({universe.upper()})"
            try:
                final_display_cols = [col for col in display_cols if col in leaders_df.columns]; alert_df = leaders_df.copy()
                if 'date' in alert_df.columns: alert_df['date'] = pd.to_datetime(alert_df['date']).dt.strftime('%Y-%m-%d')
                body = alert_df[final_display_cols].to_string(index=False, float_format="%.2f")
                dispatch(subject, body); log.info("Alerts dispatched.")
            except Exception as e: log.error(f"Failed dispatch: {e}", exc_info=True)
    elif not silent: log.info(f"No leaders found for universe '{universe}'."); print(f"\n--- No leaders found for {universe.upper()}. ---")
    return leaders_df if leaders_df is not None else pd.DataFrame()

# --- Function to Scan All Universes (Modified to return results) ---
def run_scan_for_all_universes(top_per_universe: int = 5, alert: bool = False, benchmark: str = "SPY", return_results: bool = False):
    """Scans all CSV files in the resources directory."""
    resources_dir = _PACKAGE_ROOT / "resources"
    if not resources_dir.is_dir(): log.error(f"Resources dir not found: {resources_dir}"); return {} if return_results else None
    all_results = {}; found_universes = []
    log.info(f"Scanning all universes in {resources_dir}...")
    resource_files = sorted([item for item in os.listdir(resources_dir) if item.lower().endswith(".csv")])
    for item in resource_files:
        if item.startswith(('.', '~')): continue
        universe_name = item[:-4]; log.info(f"--- Processing Universe: {universe_name} ---")
        try:
            leaders_df = run_daily_scan(universe=universe_name, top=top_per_universe, alert=False, silent=True, benchmark=benchmark)
            if leaders_df is not None and not leaders_df.empty: all_results[universe_name] = leaders_df; found_universes.append(universe_name)
            else: log.info(f"No results found for universe: {universe_name}")
        except ValueError as ve: log.error(f"Skipping '{universe_name}' loading error: {ve}")
        except Exception as e: log.error(f"Failed scan '{universe_name}': {type(e).__name__} - {e}", exc_info=False)

    if return_results: log.info("Multi-universe scan complete. Returning results."); return all_results

    # --- Original printing/alerting logic ---
    log.info("\n--- Multi-Universe Scan Summary ---")
    if not all_results: print("No leaders found in any universe."); return
    for universe_name in found_universes:
        if universe_name in all_results:
             df = all_results[universe_name]; print(f"\n=== Top {len(df)} Leaders for {universe_name.upper()} ===")
             display_cols = ['symbol', 'date', 'setup_type', 'score', 'r_multiple', 'Close', 'stop', 'target']
             final_display_cols = []
             for col in display_cols:
                  if col in df.columns: final_display_cols.append(col)
                  else: df[col] = pd.NA; final_display_cols.append(col)
             if 'date' in df.columns:
                try: df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                except Exception as date_err: log.warning(f"Date format warning {universe_name}: {date_err}")
             print(df[final_display_cols].to_string(index=False, float_format="%.2f"))
    if alert: # Combined alert logic
        log.info("Sending combined alert..."); combined_body = ""; total_leaders = 0
        for universe_name in found_universes:
             if universe_name in all_results:
                 df = all_results[universe_name]; display_cols = ['symbol', 'date', 'setup_type', 'score', 'r_multiple', 'Close', 'stop', 'target']
                 final_display_cols = [col for col in display_cols if col in df.columns]; alert_df = df.copy();
                 if 'date' in alert_df.columns:
                     try: alert_df['date'] = pd.to_datetime(alert_df['date']).dt.strftime('%Y-%m-%d')
                     except: pass
                 combined_body += f"\n=== Top {len(df)} Leaders {universe_name.upper()} ===\n"
                 combined_body += alert_df[final_display_cols].to_string(index=False, float_format="%.2f") + "\n"
                 total_leaders += len(df)
        if total_leaders > 0:
            today_str = dt.date.today().strftime('%Y-%m-%d'); subject = f"Multi-Universe Scan ({today_str}) - {total_leaders} Matches"
            try: dispatch(subject, combined_body); log.info("Combined alert dispatched.")
            except Exception as e: log.error(f"Failed dispatch: {e}", exc_info=True)

    return None # Return None if printing/alerting

# --- Command-Line Interface ---
def _cli():
    """Command-line interface setup and execution."""
    parser = argparse.ArgumentParser(description="Run the Leader Stock Scanner.")
    parser.add_argument("--universe", type=str, default="ALL", help="Stock universe name (e.g., sp500) or 'ALL'. Default: ALL")
    parser.add_argument("--top", type=int, default=None, help="Number of top leaders (default: 5 for ALL, 20 for single).")
    parser.add_argument("--alert", action="store_true", help="Send alerts.")
    parser.add_argument("--silent", action="store_true", help="Suppress console output (single universe only).")
    parser.add_argument("--force-download", action="store_true", help="Force data download, ignore cache.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark symbol (default: SPY).")

    args = parser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logging.getLogger('leader_scan').setLevel(log_level); log.setLevel(log_level)
    if args.debug: log.debug(f"Debug logging enabled.\nArgs: {args}\nConfig: {CONFIG}")
    if args.force_download: log.info("Force download enabled."); CONFIG['force_download_flag'] = True
    if args.universe.upper() == "ALL": top_n = args.top if args.top is not None else 5; run_scan_for_all_universes(top_per_universe=top_n, alert=args.alert, benchmark=args.benchmark)
    else: top_n = args.top if args.top is not None else 20; run_daily_scan(universe=args.universe, top=top_n, alert=args.alert, silent=args.silent, benchmark=args.benchmark)
    if 'force_download_flag' in CONFIG: del CONFIG['force_download_flag']

if __name__ == "__main__":
    _cli()