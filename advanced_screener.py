# leader_scan/advanced_screener.py
"""
Advanced stock screener that combines multiple setup types.
Can scan single universes or all universes found in the resources directory.
Uses lowercase internally for columns during indicator calculation.
"""
import os
import pandas as pd
import argparse
import logging
from typing import List, Dict, Optional, Set
from pathlib import Path # Import pathlib
import datetime as dt # Import datetime
import numpy as np # Import numpy

# Use absolute imports for clarity within the package
from .scorer import score_dataframe # Scorer expects Capitalized cols
from .data import get_price_data, load_universe
from .config import CONFIG
# Indicators expect lowercase cols
from .indicators import atr, ma, ema, rs_line, rs_new_high, _safe_divide

# Setup logger for this module
log = logging.getLogger(__name__)
_PACKAGE_ROOT = Path(__file__).resolve().parent # Path relative to this file

# --- Helper Functions ---
def _calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates RSI using EMA. Assumes lowercase 'close' column."""
    close_col = 'close'; rsi_col = 'rsi'
    if rsi_col not in df.columns:
        log.debug(f"Calculating RSI...")
        if close_col in df.columns and pd.api.types.is_numeric_dtype(df[close_col]):
            delta = df[close_col].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
            ma_up = up.ewm(com=window - 1, adjust=False).mean(); ma_down = down.ewm(com=window - 1, adjust=False).mean()
            rs = _safe_divide(ma_up, ma_down, default=np.inf); rs_adjusted = rs.replace([np.inf, -np.inf], np.nan).fillna(1e9)
            df[rsi_col] = 100.0 - (100.0 / (1.0 + rs_adjusted)); df[rsi_col] = df[rsi_col].fillna(50)
        else: log.warning(f"Cannot calculate RSI, '{close_col}' missing/non-numeric."); df[rsi_col] = np.nan
    return df

def find_top_setups(
    symbols: Optional[List[str]] = None, universe: Optional[str] = None,
    benchmark: str = "SPY", period: str = '2y', interval: str = '1d',
    setup_type: Optional[str] = None, min_r: Optional[float] = None, top_n: int = 5
) -> pd.DataFrame:
    """Finds top setups for a given list or SINGLE universe."""
    if symbols is None:
        if universe is None or universe.upper() == 'ALL': log.error("Requires specific universe/symbols."); return pd.DataFrame()
        log.info(f"Loading universe: {universe}")
        try: symbols_to_scan = load_universe(universe)
        except ValueError as e: log.error(f"Failed load '{universe}': {e}. Aborting."); return pd.DataFrame()
    else:
        symbols_to_scan = list(set(str(s).strip().upper() for s in symbols)); universe = "custom_list"
    if not symbols_to_scan: log.warning(f"No symbols for scanning ({universe})."); return pd.DataFrame()

    log.info(f"Adv Scan: Processing {len(symbols_to_scan)} symbols from '{universe}'...");
    final_min_r = min_r if min_r is not None else CONFIG.get("min_r_multiple", 2.5)
    today = dt.date.today(); start_date = today - dt.timedelta(days=730)

    # --- Fetch Benchmark Data (Corrected Cache Key & MultiIndex Handling) ---
    log.debug(f"Fetching benchmark {benchmark}...")
    bench_df = get_price_data(benchmark, start_date=start_date, end_date=today, interval=interval, universe_name_for_cache=benchmark)
    bench_close = None
    if bench_df is None or bench_df.empty: log.warning(f"Failed benchmark fetch ({benchmark}).")
    else:
        log.debug(f"Benchmark ({benchmark}) DataFrame columns: {bench_df.columns}"); log.debug(f"Benchmark ({benchmark}) DataFrame head:\n{bench_df.head(2)}")
        close_col_id = None
        if isinstance(bench_df.columns, pd.MultiIndex):
             log.debug(f"Benchmark df has MultiIndex columns.")
             if benchmark in bench_df.columns.get_level_values(0):
                 price_level_values = bench_df[benchmark].columns
                 log.debug(f"Benchmark price level columns: {price_level_values}")
                 found_col_name = next((c for c in price_level_values if str(c) == 'Close'), None) or \
                                  next((c for c in price_level_values if str(c).lower() == 'close'), None)
                 if found_col_name: close_col_id = (benchmark, found_col_name); log.debug(f"Identified MultiIndex Close column ID: {close_col_id}")
                 else: log.warning(f"Could not find 'Close'/'close' in price level of benchmark MultiIndex.")
             else: log.warning(f"Benchmark ticker {benchmark} not found in MultiIndex level 0.")
        else:
             log.debug(f"Benchmark df has flat columns.")
             found_col_name = next((c for c in bench_df.columns if str(c) == 'Close'), None) or \
                              next((c for c in bench_df.columns if str(c).lower() == 'close'), None)
             if found_col_name: close_col_id = found_col_name; log.debug(f"Identified flat Close column ID: {close_col_id}")
             else: log.warning(f"Could not find 'Close'/'close' in flat benchmark columns.")

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
        else: log.error(f"Could not find 'Close' column identifier in benchmark {benchmark}.")

    if bench_close is None: log.warning(f"Proceeding without benchmark {benchmark}. RS calculations skipped.")

    # --- Fetch symbol data ---
    log.debug(f"Fetching price data for {len(symbols_to_scan)} symbols...")
    data = get_price_data(symbols_to_scan, start_date=start_date, end_date=today, interval=interval, universe_name_for_cache=universe)
    if data is None or data.empty: log.error("Failed price data fetch."); return pd.DataFrame()

    results = []; processed_symbols: Set[str] = set(); symbols_in_data = []
    if isinstance(data.columns, pd.MultiIndex): symbols_in_data = list(data.columns.levels[0])
    elif not data.empty and len(symbols_to_scan) == 1 : symbols_in_data = symbols_to_scan

    for sym in symbols_in_data:
        if sym in processed_symbols: continue
        try:
            if isinstance(data.columns, pd.MultiIndex): df = data[sym].copy().dropna(subset=['Close'])
            else: df = data.copy().dropna(subset=['Close'])
            df.columns = [str(c).lower() for c in df.columns] # Lowercase for internal use

            required_cols_lower = ['open','high','low','close','volume']
            if not all(c in df.columns for c in required_cols_lower):
                 missing = [rc for rc in required_cols_lower if rc not in df.columns]; log.warning(f"Skipping {sym}: Missing: {missing}."); continue
            if len(df) < 50: continue

            df = _calculate_rsi(df)
            for window in [10, 20, 50, 200]: df[f'ma{window}'] = ma(df['close'], window)
            if 'atr' not in df.columns: df['atr'] = atr(df[['high', 'low', 'close']], window=14) # Pass lowercase
            if bench_close is not None and 'rs_line' not in df.columns:
                 aligned_bench = bench_close.reindex(df.index)
                 if not aligned_bench.isnull().all():
                      df['rs_line'] = rs_line(df['close'], aligned_bench, smooth=5)
                      df['rs_slope'] = df['rs_line'].diff(5) if 'rs_line' in df.columns else np.nan
                 else: df['rs_line'], df['rs_slope'] = np.nan, np.nan
            else: df['rs_line'], df['rs_slope'] = np.nan, np.nan

            rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'adj close': 'Adj Close',
                          'ma10': 'MA10', 'ma20': 'MA20', 'ma50': 'MA50', 'ma200': 'MA200', 'atr': 'ATR', 'rsi': 'RSI',
                          'rs_line': 'RS_Line', 'rs_slope': 'RS_slope'}
            final_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
            df.rename(columns=final_rename_map, inplace=True)
            log.debug(f"{sym} columns before scoring: {df.columns.tolist()}")

            scored = score_dataframe(df) # Scorer expects Capitalized

            if not scored.empty:
                best_row = scored.iloc[-1].copy() if isinstance(scored, pd.DataFrame) else scored.copy()
                current_setup_type = best_row.get('setup_type', 'NONE'); current_r = best_row.get('r_multiple', 0.0); current_score = best_row.get('score', 0.0); current_date = best_row.name
                if pd.isna(current_r) or current_r < final_min_r: continue
                if setup_type and setup_type != 'ALL' and current_setup_type != setup_type: continue
                close_val = best_row.get('Close') # Expect Capitalized
                result = {'symbol': sym, 'score': current_score, 'setup_type': current_setup_type, 'entry': close_val,
                          'stop': best_row.get('stop', 0.0), 'target': best_row.get('target', 0.0), 'r_multiple': current_r,
                          'date': pd.to_datetime(current_date).strftime('%Y-%m-%d') if pd.notna(current_date) else None}
                results.append(result)
            processed_symbols.add(sym)
        except Exception as e: log.error(f"Error processing {sym} in advanced scan: {e}", exc_info=False)

    if results:
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(['score', 'r_multiple'], ascending=[False, False])
        return result_df.head(top_n).reset_index(drop=True)
    else: log.info(f"Advanced Scan: No qualifying setups found for '{universe}'."); return pd.DataFrame()


# --- Function to Scan All Universes (Modified to return results) ---
def run_advanced_scan_for_all(top_per_universe: int = 5, setup_type_filter: Optional[str] = None, min_r_filter: Optional[float] = None, benchmark: str = "SPY", return_results: bool = False):
    """Scans all CSV files in the resources directory using the advanced logic."""
    resources_dir = _PACKAGE_ROOT / "resources"
    if not resources_dir.is_dir(): log.error(f"Resources directory not found: {resources_dir}"); return {} if return_results else None
    all_results = {}; log.info(f"Adv Scan: Scanning all universes in {resources_dir}...")
    resource_files = sorted([item for item in os.listdir(resources_dir) if item.lower().endswith(".csv")])
    for item in resource_files:
        if item.startswith(('.', '~')): continue
        universe_name = item[:-4]; log.info(f"--- Processing Universe: {universe_name} ---")
        try:
            leaders_df = find_top_setups(universe=universe_name, setup_type=setup_type_filter,
                min_r=min_r_filter, top_n=top_per_universe, benchmark=benchmark)
            if leaders_df is not None and not leaders_df.empty: all_results[universe_name] = leaders_df
            else: log.info(f"No results found for universe: {universe_name}")
        except ValueError as ve: log.error(f"Skipping '{universe_name}' loading error: {ve}")
        except Exception as e: log.error(f"Failed adv scan '{universe_name}': {type(e).__name__} - {e}", exc_info=False)

    if return_results: log.info("Advanced multi-universe scan complete. Returning results."); return all_results

    log.info("\n--- Multi-Universe Advanced Scan Summary ---")
    if not all_results: print("No leaders found in any universe."); return
    for universe_name, df in all_results.items():
        print(f"\n=== Top {len(df)} Adv Leaders for {universe_name.upper()} ===")
        display_cols = ['symbol', 'date', 'setup_type', 'score', 'r_multiple', 'entry', 'stop', 'target']
        final_display_cols = []
        for col in display_cols:
             if col in df.columns: final_display_cols.append(col)
             else: df[col] = pd.NA; final_display_cols.append(col)
        if 'date' in df.columns:
             try: df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
             except: pass
        print(df[final_display_cols].to_string(index=False, float_format="%.2f"))
    return None

# --- Main Execution Block (Includes --force-download and corrected logic) ---
def main():
    """Run the advanced screener with command-line options"""
    parser = argparse.ArgumentParser(description="Advanced Stock Screener")
    parser.add_argument("--universe", type=str, default="ALL", help="Stock universe name (e.g., sp500) or 'ALL'. Default: ALL")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of specific symbols (overrides universe).")
    parser.add_argument("--type", choices=["ALL", "STAGE2", "MA_CROSS", "LEADER"], default="ALL", help="Setup type filter.")
    parser.add_argument("--min-r", type=float, default=None, help=f"Minimum R multiple. Default: {CONFIG.get('min_r_multiple', 2.5)}")
    parser.add_argument("--top", type=int, default=5, help="Number of top results per universe/list (default: 5).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark symbol (default: SPY).")
    parser.add_argument("--combined", action="store_true", help="Display combined results grouped by type (applies to single universe/list scans).")
    parser.add_argument("--force-download", action="store_true", help="Force data download, ignore cache.") # Defined

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logging.getLogger('leader_scan').setLevel(log_level); log.setLevel(log_level)
    if args.debug: log.debug(f"Debug logging enabled.\nArgs: {args}")

    if args.force_download:
         log.info("Force download enabled globally for this run.")
         CONFIG['force_download_flag'] = True

    symbols_list = args.symbols.split(',') if args.symbols else None
    min_r_val = args.min_r
    setup_type_filter = "ALL" if args.combined else args.type
    top_n_val = args.top
    results = None; scan_target = None

    # --- Corrected Execution Logic ---
    if symbols_list:
        scan_target = "Specific Symbols"; print(f"\n--- Advanced Scan for {scan_target} ---")
        fetch_top_n = top_n_val * 3 if args.combined else top_n_val
        results = find_top_setups(symbols=symbols_list, setup_type=setup_type_filter, min_r=min_r_val, top_n=fetch_top_n, benchmark=args.benchmark)
    elif args.universe.upper() == "ALL":
        log.info("Running multi-universe scan...")
        # Determine the filter type for the multi-scan
        # --combined implies ALL types across universes
        multi_setup_filter = "ALL" if args.combined else (args.type if args.type != "ALL" else None)
        if args.combined and args.type != "ALL":
             log.warning("Ignoring --type filter when --combined and --universe ALL are used.")
        run_advanced_scan_for_all(top_per_universe=top_n_val, setup_type_filter=multi_setup_filter, min_r_filter=min_r_val, benchmark=args.benchmark)
        if 'force_download_flag' in CONFIG: del CONFIG['force_download_flag']; return
    else:
        scan_target = f"Universe: {args.universe.upper()}"; print(f"\n--- Advanced Scan for {scan_target} ---")
        fetch_top_n = top_n_val * 3 if args.combined else top_n_val
        results = find_top_setups(universe=args.universe, setup_type=setup_type_filter, min_r=min_r_val, top_n=fetch_top_n, benchmark=args.benchmark)

    # Display results only if NOT multi-scan
    if args.universe.upper() != "ALL":
         if results is None or results.empty: print(f"No qualifying setups found for the specified target ({scan_target}).")
         elif args.combined:
             print(f"\n=== Combined Scan Results for {scan_target} (Top up to {top_n_val} per type) ===")
             for setup_type_val in ['STAGE2', 'MA_CROSS', 'LEADER']:
                 type_results = results[results['setup_type'] == setup_type_val].head(top_n_val)
                 if not type_results.empty:
                     print(f"\n--- {setup_type_val} Strategy ---"); display_cols = ['symbol', 'date', 'score', 'r_multiple', 'entry', 'stop', 'target']
                     final_display_cols = [col for col in display_cols if col in type_results.columns]
                     if 'date' in type_results.columns: type_results['date'] = pd.to_datetime(type_results['date']).dt.strftime('%Y-%m-%d')
                     print(type_results[final_display_cols].to_string(index=False, float_format="%.2f"))
         else:
             print(f"\n=== Top {len(results)} {args.type} Setups for {scan_target} ===")
             display_cols = ['symbol', 'date', 'setup_type', 'score', 'r_multiple', 'entry', 'stop', 'target']
             final_display_cols = [col for col in display_cols if col in results.columns]
             if 'date' in results.columns: results['date'] = pd.to_datetime(results['date']).dt.strftime('%Y-%m-%d')
             print(results[final_display_cols].to_string(index=False, float_format="%.2f"))

    if 'force_download_flag' in CONFIG: del CONFIG['force_download_flag'] # Clean up flag

if __name__ == "__main__":
    main()