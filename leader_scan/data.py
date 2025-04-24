# leader_scan/data.py
"""
Data ingestion helpers for the leader_scan package.
Includes file-based and in-memory caching, and batch downloading.
"""

import datetime as dt
import pathlib
from pathlib import Path # Ensure Path is imported
import sys
import logging
import hashlib
from typing import Iterable, List, Dict, Any, Optional, Union
import time # Import the time module for delays

import pandas as pd
try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance library not found. Please install it (`pip install yfinance`) to fetch price data.", file=sys.stderr)
    yf = None
# Try importing pyarrow for parquet caching, but don't make it a hard requirement
try:
    import pyarrow
except ImportError:
    print("Warning: pyarrow library not found. File caching (.parquet) will be disabled.", file=sys.stderr)
    pyarrow = None # Set to None if not found

# Import config here to use flags if needed
try:
    from .config import CONFIG
except ImportError:
    CONFIG = {} # Minimal fallback

log = logging.getLogger(__name__)

# --- In-Memory Cache ---
_memory_cache: Dict[str, pd.DataFrame] = {}
# -----------------------

# --------------------------------------------------------------------------- #
# Configuration and File Cache Path
# --------------------------------------------------------------------------- #
_PACKAGE_ROOT = Path(__file__).resolve().parent
_CACHE_DIR = _PACKAGE_ROOT.parent / ".cache" # Place .cache outside package
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure parent dirs exist
except OSError as e:
    log.warning(f"Could not create cache directory at {_CACHE_DIR}: {e}")
    _CACHE_DIR = _PACKAGE_ROOT / ".cache_fallback" # Fallback inside package
    try:
         _CACHE_DIR.mkdir(parents=True, exist_ok=True)
         log.warning(f"Using fallback cache directory: {_CACHE_DIR}")
    except OSError as e_fallback:
         log.error(f"Could not create fallback cache directory: {e_fallback}. File caching disabled.")
         _CACHE_DIR = None

_DEFAULT_CACHE_DAYS = CONFIG.get("cache_days", 730)

# --------------------------------------------------------------------------- #
# Price Data Functions
# --------------------------------------------------------------------------- #
def _generate_ticker_hash(tickers: Iterable[str]) -> str:
    """Generates a SHA256 hash from a sorted list of tickers."""
    ticker_string = "_".join(sorted(str(t) for t in tickers))
    return hashlib.sha256(ticker_string.encode('utf-8')).hexdigest()[:16]

def _get_cache_key(
    tickers: Iterable[str], start_date: dt.date, end_date: dt.date,
    interval: str, universe_name: Optional[str] = None
) -> Optional[Path]:
    """Generates file cache key. Returns None if file caching disabled."""
    if _CACHE_DIR is None or pyarrow is None: log.debug("File cache key skipped."); return None
    date_format = "%Y%m%d"
    ticker_list = list(tickers) if tickers else []
    if universe_name:
         safe_name = "".join(c if c.isalnum() else '_' for c in universe_name); base_name = safe_name[:50]
    elif ticker_list and len(ticker_list) == 1:
         safe_ticker = "".join(c if c.isalnum() else '_' for c in str(ticker_list[0])); base_name = safe_ticker[:50]
    else: base_name = f"tickers_{_generate_ticker_hash(ticker_list)}"
    key = f"{base_name}_{start_date.strftime(date_format)}_{end_date.strftime(date_format)}_{interval}.parquet"
    return _CACHE_DIR / key

def clear_memory_cache():
    """Clears the in-memory data cache."""
    global _memory_cache
    _memory_cache = {}
    log.info("In-memory data cache cleared.")

def get_price_data(
    tickers: Union[str, List[str]], *, start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None, interval: str = "1d",
    force_download: bool = False, universe_name_for_cache: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Downloads OHLCV data using yfinance, with batching, delays, caching, and validation.

    Returns CAPITALIZED column names (Open, High, Low, Close, Volume, Adj Close).
    """
    global _memory_cache

    # --- Input validation and date calculation ---
    if yf is None: log.error("yfinance not installed."); return None
    if isinstance(tickers, str): tickers_list = [s.strip().upper() for s in tickers.replace(",", " ").split() if s.strip()]
    elif isinstance(tickers, list): tickers_list = [s.strip().upper() for s in tickers if isinstance(s, str) and s.strip()]
    else: log.error(f"Invalid type for 'tickers': {type(tickers)}."); return None
    if not tickers_list: log.warning("No valid tickers provided."); return pd.DataFrame()
    today = dt.date.today(); final_end_date = end_date or today
    final_start_date = start_date or (final_end_date - dt.timedelta(days=_DEFAULT_CACHE_DAYS))
    global_force_download = CONFIG.get('force_download_flag', False)
    if global_force_download: force_download = True; log.info("Forcing download due to flag.")

    # --- Create unique key for caching ---
    cache_key_tuple = (tuple(sorted(tickers_list)), final_start_date, final_end_date, interval)
    cache_key_str = str(cache_key_tuple)

    # --- 1. Check In-Memory Cache ---
    if not force_download and cache_key_str in _memory_cache:
        log.debug(f"Loading data from memory cache for key (approx): {tickers_list[:2]}...")
        return _memory_cache[cache_key_str].copy()

    # --- 2. Check File Cache (Parquet) ---
    cache_file = _get_cache_key(tickers_list, final_start_date, final_end_date, interval,
                                universe_name=universe_name_for_cache or (tickers_list[0] if len(tickers_list) == 1 else None))
    if not force_download and cache_file and cache_file.exists() and cache_file.is_file():
        try:
            log.debug(f"Loading cached data from file: {cache_file}")
            df = pd.read_parquet(cache_file)
            if not df.empty:
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if isinstance(df.columns, pd.MultiIndex): actual_cols = list(df.columns.levels[1]) if len(df.columns.levels) > 1 else []
                else: actual_cols = list(df.columns)
                if all(col in actual_cols for col in expected_cols):
                    log.debug(f"File cache validated for {cache_file}.")
                    _memory_cache[cache_key_str] = df # Store in memory
                    return df.copy() # Return copy
                else: log.warning(f"Cached file {cache_file} missing columns ({actual_cols}). Forcing download."); force_download = True
            else: log.warning(f"Cached file {cache_file} is empty. Forcing download."); force_download = True
        except ImportError: log.warning("pyarrow not installed. Cannot read file cache.")
        except Exception as e: log.error(f"Error reading cache file {cache_file}: {e}. Forcing download.", exc_info=False); force_download = True

    # --- 3. Download Data (with Batching) ---
    if force_download or cache_key_str not in _memory_cache:
        log.info(f"Fetching/Downloading price data via yfinance for {len(tickers_list)} tickers...")
        df_downloaded = None; yf_exception = None
        all_batch_data = [] # Store results from successful batches
        batch_size = 50  # Adjust as needed (50-100 is often reasonable)
        delay_between_batches = 2 # Seconds (adjust based on testing/rate limits)

        try:
            num_batches = (len(tickers_list) + batch_size - 1) // batch_size
            for i in range(0, len(tickers_list), batch_size):
                batch = tickers_list[i:i + batch_size]
                current_batch_num = i//batch_size + 1
                log.info(f"Fetching batch {current_batch_num}/{num_batches}: ({len(batch)} tickers starting with {batch[0]}...)")
                try:
                    df_batch = yf.download(tickers=" ".join(batch), start=final_start_date,
                                           end=final_end_date + dt.timedelta(days=1), interval=interval,
                                           group_by="ticker", auto_adjust=False, prepost=False,
                                           threads=True, progress=False, ignore_tz=True)

                    if not df_batch.empty and isinstance(df_batch.columns, pd.MultiIndex):
                        # Filter out tickers that returned only NaNs immediately
                        valid_batch_data = {}
                        for ticker in df_batch.columns.levels[0]:
                            # Ensure the ticker was actually requested in this batch
                            # and yfinance didn't return extra due to similar names
                            if ticker in batch and not df_batch[ticker].isnull().all().all():
                                 valid_batch_data[ticker] = df_batch[ticker]
                        if valid_batch_data:
                             # Concatenate valid data for this batch
                             all_batch_data.append(pd.concat(valid_batch_data, axis=1))
                             log.debug(f"Batch {current_batch_num} successful for {len(valid_batch_data)} tickers.")
                        else: log.warning(f"No valid data obtained in batch {current_batch_num} (started with {batch[0]}).")
                    elif df_batch.empty:
                         log.warning(f"Empty DataFrame returned for batch {current_batch_num} (started with {batch[0]}).")
                    else:
                         log.warning(f"Unexpected non-MultiIndex DataFrame returned for batch {current_batch_num}. Discarding batch.")

                except Exception as e_batch:
                    log.error(f"Download failed for batch {current_batch_num} (started with {batch[0]}): {type(e_batch).__name__} - {e_batch}", exc_info=False) # Log batch error but continue

                # Delay between batches (except after the last one)
                if current_batch_num < num_batches:
                     log.debug(f"Waiting {delay_between_batches}s before next batch...")
                     time.sleep(delay_between_batches)

            # Combine batch results
            if all_batch_data:
                df_downloaded = pd.concat(all_batch_data, axis=1)
                # Sort columns by ticker name (level 0) for consistency
                df_downloaded = df_downloaded.sort_index(axis=1, level=0)
                log.info(f"Successfully combined data for {len(df_downloaded.columns.levels[0])} tickers from {num_batches} batches.")
            else:
                log.error("No valid data downloaded from any batch.")
                df_downloaded = pd.DataFrame() # Ensure empty df if all batches failed

        except Exception as e_concat: # Catch errors during concat or main loop
            yf_exception = e_concat
            log.error(f"Exception during batch processing/concatenation: {type(e_concat).__name__} - {e_concat}", exc_info=True)
            df_downloaded = pd.DataFrame() # Ensure empty df on critical error

        # --- Post-Download Validation & Standardization (Applied to Concatenated DF) ---
        if df_downloaded is None or df_downloaded.empty:
            log.warning(f"No data obtained after batch download attempts for: {', '.join(tickers_list)}")
            df_processed = pd.DataFrame()
        else:
            log.debug("Processing final concatenated DataFrame...")
            successful_tickers = []; df_processed = pd.DataFrame(); expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if isinstance(df_downloaded.columns, pd.MultiIndex):
                valid_tickers_data = {}
                downloaded_tickers = df_downloaded.columns.get_level_values(0).unique()
                for ticker in downloaded_tickers:
                     # Ensure we only keep tickers originally requested
                     if ticker in tickers_list:
                         ticker_df = df_downloaded[ticker].copy()
                         ticker_df.columns = [str(c).strip().capitalize() for c in ticker_df.columns] # Standardize
                         if all(col in ticker_df.columns for col in expected_cols):
                             # Drop rows where Close is NaN *after* standardization
                             valid_ticker_df = ticker_df.dropna(subset=['Close'])
                             if not valid_ticker_df.empty:
                                  valid_tickers_data[ticker] = valid_ticker_df
                                  successful_tickers.append(ticker)
                             else: log.debug(f"{ticker} empty after dropna(Close).")
                         else: log.warning(f"{ticker} missing std cols after batch concat: {ticker_df.columns.tolist()}. Skipping.")
                # Log tickers requested but not in final valid data
                failed_tickers = [t for t in tickers_list if t not in successful_tickers]
                if failed_tickers: log.warning(f"Tickers with no valid/processed data: {', '.join(failed_tickers)}")
                # Re-concatenate only the valid, processed data
                if valid_tickers_data: df_processed = pd.concat(valid_tickers_data, axis=1).sort_index(axis=1, level=0)
                else: log.error("No valid data remains after processing concatenated batches."); df_processed = pd.DataFrame()
            else: log.error(f"Final downloaded df has unexpected structure (not MultiIndex). Cols: {df_downloaded.columns}")

        # --- Cache and Return ---
        if df_processed.empty: log.warning(f"Returning empty DataFrame for {', '.join(tickers_list)} after processing.")
        # Cache the result (even if empty)
        _memory_cache[cache_key_str] = df_processed
        if cache_file:
            try: df_processed.to_parquet(cache_file, compression="snappy")
            except ImportError: log.warning("pyarrow not installed. Cannot save file cache.")
            except Exception as e_save: log.error(f"Error saving cache file {cache_file}: {e_save}", exc_info=False)
        # Return None only if there was a critical exception during download/concat
        return None if yf_exception else df_processed.copy()

    # Fallback if logic path is unexpected (e.g., failed cache read didn't force download)
    log.warning("Returning None from get_price_data (unexpected execution path).")
    return None

# --- Fundamentals Function ---
def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Retrieves basic fundamental data using yfinance Ticker info."""
    if yf is None: log.error("yfinance not installed, cannot get fundamentals."); return {}
    fundamentals = {}; tkr = yf.Ticker(ticker)
    try: info = tkr.info
    except Exception as e: log.warning(f"Could not get Ticker info for {ticker}: {e}"); info = {}
    key_map = {"market_cap": "marketCap", "eps_ttm": "trailingEps", "pe_ratio": "trailingPE",
               "ps_ratio": "priceToSalesTrailing12Months", "sector": "sector", "industry": "industry",
               "currency": "currency", "shares_outstanding": "sharesOutstanding", "beta": "beta",
               "dividend_yield": "dividendYield", "forward_pe": "forwardPE"}
    for f_key, yf_key in key_map.items(): fundamentals[f_key] = info.get(yf_key)
    try:
         mcap = fundamentals.get("market_cap"); ps = fundamentals.get("ps_ratio")
         if mcap is not None and ps is not None and ps != 0: fundamentals["sales_ttm"] = mcap / ps
         else: fundamentals["sales_ttm"] = None
    except Exception: fundamentals["sales_ttm"] = None
    return fundamentals

# --- Universe Loading Function ---
def load_universe(name: str = "sp500") -> List[str]:
    """Loads ticker symbols from a CSV file in resources directory."""
    name_lower = name.lower().strip()
    if not name_lower: raise ValueError("Universe name cannot be empty.")
    csv_filename = f"{name_lower}.csv"
    resources_dir = _PACKAGE_ROOT / "resources"
    csv_path = resources_dir / csv_filename
    log.info(f"Attempting to load universe '{name}' from: {csv_path}")
    if not csv_path.exists() or not csv_path.is_file():
        resources_dir_alt = _PACKAGE_ROOT.parent / "resources" # Fallback check
        csv_path_alt = resources_dir_alt / csv_filename
        if csv_path_alt.exists() and csv_path_alt.is_file(): log.warning(f"Using universe file from parent: {csv_path_alt}"); csv_path = csv_path_alt
        else: raise ValueError(f"Universe CSV file not found at {csv_path} or {csv_path_alt}")

    symbols_list = []; df = pd.DataFrame()
    try:
        try: df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8', on_bad_lines='warn', skipinitialspace=True)
        except UnicodeDecodeError: log.warning(f"UTF-8 failed for {csv_path}, trying latin1."); df = pd.read_csv(csv_path, delimiter=',', encoding='latin1', on_bad_lines='warn', skipinitialspace=True)
        except pd.errors.ParserError as pe: raise ValueError(f"Error parsing CSV {csv_path}: {pe}") from pe
        except Exception as read_err: raise ValueError(f"Error reading CSV {csv_path}: {read_err}") from read_err
        if df.empty: log.warning(f"{csv_path} loaded empty."); return []

        symbol_col = None
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in ['symbol', 'ticker', 'code', 'isin']: symbol_col = col; break
        if symbol_col is None:
            if len(df.columns) > 0: symbol_col = df.columns[0]; log.warning(f"No standard symbol column found. Using first column '{symbol_col}'.")
            else: raise ValueError(f"No columns found in {csv_path}.")

        symbols_raw = df[symbol_col].dropna().astype(str)
        symbols_list_raw = symbols_raw.str.strip().str.upper().tolist()
        symbols_list = []
        skip_prefixes = ("PERF.", "KGV", "MARKT-", "KAUFEN", "VERKAUFEN", "#", "//")
        skip_exact = {"SYMBOL", "TICKER", "ISIN", "NAME", "NAN", ""}
        problem_chars = ('/', '^', '+', '*')
        for s in symbols_list_raw:
            if s in skip_exact or s.startswith(skip_prefixes) or any(char in s for char in problem_chars):
                log.debug(f"Skipping invalid symbol: {s}"); continue
            symbols_list.append(s) # Use symbol directly

        if not symbols_list: log.warning(f"No valid symbols extracted from {csv_path}."); return []
        log.info(f"Successfully loaded {len(symbols_list)} symbols from {csv_path}.")
        return symbols_list
    except KeyError as e: raise ValueError(f"Column '{symbol_col}' not found: {e}") from e
    except Exception as e: log.error(f"Error loading universe {csv_path}: {e}", exc_info=True); raise ValueError(f"Failed loading {csv_path}") from e

# --- Explicit Export List ---
__all__ = ["get_price_data", "get_fundamentals", "load_universe", "clear_memory_cache"]