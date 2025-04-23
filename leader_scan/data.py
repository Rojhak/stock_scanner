# leader_scan/data.py
"""
Data ingestion helpers for the leader_scan package.
"""

import datetime as dt
import pathlib
from pathlib import Path # Ensure Path is imported
import sys
import logging
import hashlib
from typing import Iterable, List, Dict, Any, Optional, Union

import pandas as pd
try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance library not found. Please install it (`pip install yfinance`) to fetch price data.", file=sys.stderr)
    yf = None

# Import config here to use flags if needed
try:
    from .config import CONFIG
except ImportError:
    CONFIG = {} # Minimal fallback

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration and Cache Path
# --------------------------------------------------------------------------- #
_PACKAGE_ROOT = Path(__file__).resolve().parent
_CACHE_DIR = _PACKAGE_ROOT.parent / ".cache" # Place .cache outside package
try:
    _CACHE_DIR.mkdir(exist_ok=True)
except OSError as e:
    log.warning(f"Could not create cache directory at {_CACHE_DIR}: {e}")
    _CACHE_DIR = _PACKAGE_ROOT / ".cache_fallback" # Example fallback
    try:
         _CACHE_DIR.mkdir(exist_ok=True)
         log.warning(f"Using fallback cache directory: {_CACHE_DIR}")
    except OSError as e_fallback:
         log.error(f"Could not create fallback cache directory at {_CACHE_DIR}: {e_fallback}. Caching disabled.")
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
    """Generates cache key. Returns None if caching disabled."""
    if _CACHE_DIR is None: return None
    date_format = "%Y%m%d"
    ticker_list = list(tickers) if tickers else [] # Ensure list
    if universe_name:
         safe_name = "".join(c if c.isalnum() else '_' for c in universe_name); base_name = safe_name[:50]
    elif ticker_list and len(ticker_list) == 1:
         safe_ticker = "".join(c if c.isalnum() else '_' for c in str(ticker_list[0])); base_name = safe_ticker[:50]
    else: base_name = f"tickers_{_generate_ticker_hash(ticker_list)}"
    key = f"{base_name}_{start_date.strftime(date_format)}_{end_date.strftime(date_format)}_{interval}.parquet"
    return _CACHE_DIR / key

def get_price_data(
    tickers: Union[str, List[str]], *, start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None, interval: str = "1d",
    force_download: bool = False, universe_name_for_cache: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Downloads OHLCV data using yfinance, with improved caching and error handling."""
    global_force_download = CONFIG.get('force_download_flag', False)
    if global_force_download: force_download = True; log.info("Forcing download due to flag.")

    if yf is None: log.error("yfinance not installed."); return None

    if isinstance(tickers, str): tickers_list = [s.strip().upper() for s in tickers.replace(",", " ").split() if s.strip()]
    elif isinstance(tickers, list): tickers_list = [s.strip().upper() for s in tickers if isinstance(s, str) and s.strip()]
    else: log.error(f"Invalid type for 'tickers': {type(tickers)}."); return None
    if not tickers_list: log.warning("No valid tickers provided."); return pd.DataFrame()

    today = dt.date.today()
    final_end_date = end_date or today
    final_start_date = start_date or (final_end_date - dt.timedelta(days=_DEFAULT_CACHE_DAYS))

    cache_file = _get_cache_key(tickers_list, final_start_date, final_end_date, interval,
                                universe_name=universe_name_for_cache or (tickers_list[0] if len(tickers_list) == 1 else None))

    # --- Check cache ---
    if not force_download and cache_file and cache_file.exists() and cache_file.is_file():
        try:
            log.debug(f"Loading cached data from: {cache_file}")
            df = pd.read_parquet(cache_file)
            if not df.empty:
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                actual_cols = []
                if isinstance(df.columns, pd.MultiIndex):
                     # Check level 1 contains the expected cols if multi-index non-empty
                     if df.columns.nlevels > 1 and len(df.columns.levels[1]) > 0:
                         actual_cols = list(df.columns.levels[1])
                     else: # Handle potentially empty MultiIndex levels
                         log.warning(f"Cache {cache_file} has empty MultiIndex levels. Forcing download.")
                         force_download = True
                else: # Flat index
                     actual_cols = list(df.columns)

                has_ohlcv = all(col in actual_cols for col in expected_cols)

                if has_ohlcv:
                    log.debug(f"Cache validated for {cache_file}.")
                    # Ensure consistent return type (MultiIndex) even from cache
                    if not isinstance(df.columns, pd.MultiIndex) and len(tickers_list) == 1:
                        log.debug("Converting cached flat DataFrame to MultiIndex for single ticker.")
                        df.columns = pd.MultiIndex.from_product([tickers_list, df.columns])
                    return df
                else:
                     log.warning(f"Cached file {cache_file} missing columns ({actual_cols}). Forcing download.")
                     force_download = True
            else: log.warning(f"Cached file {cache_file} is empty. Forcing download."); force_download = True
        except Exception as e: log.error(f"Error reading cache {cache_file}: {e}. Forcing download.", exc_info=False); force_download = True

    # --- Download data ---
    if force_download or cache_file is None or not cache_file.exists():
        log.info(f"Downloading/fetching price data for: {', '.join(tickers_list)}")
        df_downloaded = None # Initialize df to None
        yf_exception = None # Store potential exception

        try:
            ticker_string_yf = " ".join(tickers_list)
            # --- Try forcing group_by='ticker' even for single tickers ---
            group_by_arg = "ticker" # Always group by ticker? Sometimes helps consistency.

            log.debug(f"Calling yf.download(tickers='{ticker_string_yf}', start='{final_start_date}', end='{final_end_date + dt.timedelta(days=1)}', interval='{interval}', group_by='{group_by_arg}')")

            df_downloaded = yf.download(tickers=ticker_string_yf, start=final_start_date,
                                        end=final_end_date + dt.timedelta(days=1), # End date is exclusive for yf
                                        interval=interval,
                                        group_by=group_by_arg,
                                        auto_adjust=False, prepost=False, threads=True,
                                        progress=False, ignore_tz=True)

        except Exception as e:
            yf_exception = e # Store exception if download call fails
            log.error(f"Exception DIRECTLY from yf.download for {tickers_list}: {type(e).__name__} - {e}", exc_info=True) # Log full traceback

        # --- Enhanced Debugging and Validation AFTER the call ---
        log.debug(f"yf.download completed. Exception caught: {yf_exception is not None}")
        log.debug(f"Raw result type: {type(df_downloaded)}")
        if df_downloaded is not None:
             log.debug(f"Raw result empty: {df_downloaded.empty}")
             if not df_downloaded.empty:
                  log.debug(f"Raw result columns: {df_downloaded.columns}")
                  log.debug(f"Raw result head:\n{df_downloaded.head(2)}")
             else: log.warning("yf.download returned an empty DataFrame.")
        else: log.warning("yf.download returned None.")


        if df_downloaded is None or df_downloaded.empty:
            log.warning(f"No data downloaded (result is None or empty) for tickers: {', '.join(tickers_list)}")
            if cache_file:
                try: pd.DataFrame().to_parquet(cache_file, compression="snappy")
                except Exception as e: log.error(f"Error saving empty cache {cache_file}: {e}")
            # If there was an exception during download, return None to signal critical failure
            # Otherwise, return empty DataFrame for "no data found"
            return None if yf_exception else pd.DataFrame()


        # --- Post-Download Processing & Validation ---
        successful_tickers = []
        df_processed = pd.DataFrame()
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Handle MultiIndex (always group_by='ticker')
        if isinstance(df_downloaded.columns, pd.MultiIndex):
            valid_tickers_data = {}
            # Ensure level 0 contains the tickers we requested
            fetched_tickers = df_downloaded.columns.levels[0]
            for ticker in fetched_tickers:
                if ticker in tickers_list: # Only process tickers we actually asked for
                     try:
                         ticker_df = df_downloaded[ticker].copy() # Extract data for one ticker
                         ticker_df.columns = [str(c).strip().capitalize() for c in ticker_df.columns] # Capitalize price type cols
                         # Check required OHLCV columns exist AND the Close column is not all NaN
                         if all(col in ticker_df.columns for col in expected_cols) and not ticker_df['Close'].isnull().all():
                             valid_tickers_data[ticker] = ticker_df
                             successful_tickers.append(ticker)
                         else:
                             missing_reason = "missing standard cols" if not all(col in ticker_df.columns for col in expected_cols) else "Close column all NaN"
                             log.warning(f"{ticker} data invalid ({missing_reason}): {ticker_df.columns.tolist()}. Skipping.")
                     except Exception as proc_err:
                         log.error(f"Error processing downloaded data for {ticker}: {proc_err}", exc_info=False)

            failed_tickers = [t for t in tickers_list if t not in successful_tickers]
            if failed_tickers: log.warning(f"Failed/Invalid download for: {', '.join(failed_tickers)}")
            if valid_tickers_data:
                # Concatenate valid data, ensuring consistent structure
                df_processed = pd.concat(valid_tickers_data, axis=1)
                # Ensure the resulting DataFrame columns are MultiIndex [Ticker, OHLCV]
                if not isinstance(df_processed.columns, pd.MultiIndex):
                    log.warning("Concatenation did not result in MultiIndex, attempting reconstruction.")
                    try:
                         df_processed.columns = pd.MultiIndex.from_tuples([(ticker, col) for ticker in valid_tickers_data for col in valid_tickers_data[ticker].columns])
                    except Exception as recon_err:
                         log.error(f"Failed to reconstruct MultiIndex columns: {recon_err}. Returning potentially malformed DataFrame.")

            else: log.error("No valid data obtained for requested tickers (MultiIndex case)."); df_processed = pd.DataFrame()
        else: # Should not happen with group_by='ticker', but handle defensively
            log.error(f"Unexpected non-MultiIndex df structure from yfinance for multiple tickers ({tickers_list}). Cols: {df_downloaded.columns}")
            df_processed = pd.DataFrame() # Return empty


        if df_processed.empty:
            log.error(f"No valid data obtained for requested tickers after processing.")
            if cache_file:
                try: pd.DataFrame().to_parquet(cache_file, compression="snappy")
                except Exception as e: log.error(f"Error saving empty cache {cache_file}: {e}")
            return pd.DataFrame()

        # --- Save valid, processed data to cache ---
        if cache_file:
             try:
                 # Ensure columns are correctly formatted before saving
                 if isinstance(df_processed.columns, pd.MultiIndex):
                      df_processed.columns = pd.MultiIndex.from_tuples(df_processed.columns)
                 df_processed.to_parquet(cache_file, compression="snappy")
                 log.info(f"Saved data ({len(successful_tickers)} tickers) to cache: {cache_file}")
             except Exception as e: log.error(f"Error saving cache file {cache_file}: {e}", exc_info=False)

        return df_processed

    # Fallback if logic somehow reaches here without returning
    log.debug("Returning None from get_price_data (unexpected fallthrough).")
    return None

# --- Fundamentals (unchanged from previous versions) ---
def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Retrieves basic fundamental data."""
    if yf is None: log.error("yfinance not installed."); return {}
    fundamentals = {}; tkr = yf.Ticker(ticker)
    try: info = tkr.info
    except Exception: info = {}
    key_map = {"market_cap": "marketCap","eps_ttm": "trailingEps","pe_ratio": "trailingPE","ps_ratio": "priceToSalesTrailing12Months","sector": "sector","industry": "industry","currency": "currency","shares_outstanding": "sharesOutstanding","beta": "beta","dividend_yield": "dividendYield","forward_pe": "forwardPE"}
    for f_key, yf_key in key_map.items(): fundamentals[f_key] = info.get(yf_key)
    try:
         if fundamentals.get("market_cap") and fundamentals.get("ps_ratio") and fundamentals["ps_ratio"] != 0: fundamentals["sales_ttm"] = fundamentals["market_cap"] / fundamentals["ps_ratio"]
         else: fundamentals["sales_ttm"] = None
    except Exception: fundamentals["sales_ttm"] = None
    return fundamentals

# --- Universe Loading (Using symbols directly) ---
def load_universe(name: str = "sp1500") -> List[str]:
    """Loads ticker symbols from a CSV file in resources. Uses symbols directly."""
    name_lower = name.lower().strip()
    if not name_lower: raise ValueError("Universe name cannot be empty.")
    csv_filename = f"{name_lower}.csv"
    resources_dir = _PACKAGE_ROOT / "resources" # Correct path
    csv_path = resources_dir / csv_filename
    log.info(f"Attempting to load universe '{name}' from: {csv_path}")
    if not csv_path.exists() or not csv_path.is_file():
        raise ValueError(f"Universe CSV file not found at: {csv_path}")

    symbols_list = []
    df = pd.DataFrame()
    try:
        # Try reading with different encodings and error handling
        try: df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8', on_bad_lines='skip', skipinitialspace=True)
        except UnicodeDecodeError:
             log.warning(f"UTF-8 failed for {csv_path}, trying latin1.")
             try: df = pd.read_csv(csv_path, delimiter=',', encoding='latin1', on_bad_lines='skip', skipinitialspace=True)
             except Exception as read_err_latin1: raise ValueError(f"Error reading {csv_path} with latin1: {read_err_latin1}") from read_err_latin1
        except pd.errors.ParserError as pe: raise ValueError(f"Error parsing {csv_path}: {pe}") from pe
        except Exception as read_err: raise ValueError(f"General error reading {csv_path}: {read_err}") from read_err

        if df.empty: log.warning(f"{csv_path} loaded empty."); return []

        # Find the symbol column robustly
        symbol_col = None
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower == 'symbol': symbol_col = col; break
            if 'ticker' in col_lower: symbol_col = col # Fallback
            if 'code' in col_lower and not symbol_col: symbol_col = col # Fallback
            if 'isin' in col_lower and not symbol_col: symbol_col = col # Fallback
        if symbol_col is None:
            if len(df.columns) > 0: symbol_col = df.columns[0]; log.warning(f"Could not identify symbol column in {csv_path}, using first column '{symbol_col}'.")
            else: raise ValueError(f"No columns found in {csv_path}.")

        # Extract and clean symbols
        symbols_raw = df[symbol_col].dropna().astype(str)
        symbols_list_raw = symbols_raw.str.strip().str.upper().tolist()
        symbols_list = []
        for s in symbols_list_raw:
            # Skip common header/invalid values
            if s in ["SYMBOL", "TICKER", "ISIN", "NAME", "NAN", ""] or s.startswith(("PERF.", "KGV", "MARKT-", "KAUFEN", "VERKAUFEN", "#", "//")): continue
            s_cleaned = s # Use symbol directly from CSV
            # Skip symbols with characters yfinance often struggles with
            if '/' in s_cleaned or '^' in s_cleaned or '+' in s_cleaned or '*' in s_cleaned:
                 log.debug(f"Skipping potentially problematic symbol format: {s}")
                 continue
            # Add the cleaned symbol
            symbols_list.append(s_cleaned)

        if not symbols_list: log.warning(f"No valid symbols extracted from {csv_path} using column '{symbol_col}'."); return []
        log.info(f"Successfully loaded and cleaned {len(symbols_list)} symbols from {csv_path}.")
        return symbols_list
    except KeyError as e: raise ValueError(f"Column '{symbol_col}' not found in {csv_path}: {e}") from e
    except Exception as e: log.error(f"Unhandled error loading universe {csv_path}: {e}", exc_info=True); raise ValueError(f"Failed loading {csv_path}") from e

__all__ = ["get_price_data", "get_fundamentals", "load_universe"]