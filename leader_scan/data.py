# leader_scan/data.py
"""
Data ingestion helpers for the leader_scan package.
Includes file-based and in-memory caching.
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
# Place .cache outside the package directory (in parent)
_CACHE_DIR = _PACKAGE_ROOT.parent / ".cache"
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
    if _CACHE_DIR is None or pyarrow is None: # Also disable if pyarrow missing
        log.debug("File cache key generation skipped (Dir missing or pyarrow not installed).")
        return None
    date_format = "%Y%m%d"
    ticker_list = list(tickers) if tickers else [] # Ensure list
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
    Downloads OHLCV data using yfinance, with in-memory and file caching.

    Returns CAPITALIZED column names (Open, High, Low, Close, Volume, Adj Close).
    """
    global _memory_cache # Allow access to module-level cache

    # --- Input validation and date calculation ---
    if yf is None: log.error("yfinance not installed."); return None

    if isinstance(tickers, str): tickers_list = [s.strip().upper() for s in tickers.replace(",", " ").split() if s.strip()]
    elif isinstance(tickers, list): tickers_list = [s.strip().upper() for s in tickers if isinstance(s, str) and s.strip()]
    else: log.error(f"Invalid type for 'tickers': {type(tickers)}."); return None
    if not tickers_list: log.warning("No valid tickers provided."); return pd.DataFrame()

    today = dt.date.today()
    final_end_date = end_date or today
    final_start_date = start_date or (final_end_date - dt.timedelta(days=_DEFAULT_CACHE_DAYS))

    # Respect global force download flag
    global_force_download = CONFIG.get('force_download_flag', False)
    if global_force_download: force_download = True; log.info("Forcing download due to flag.")

    # --- Create unique key for caching ---
    cache_key_tuple = (
        tuple(sorted(tickers_list)), # Use sorted tuple of tickers
        final_start_date,
        final_end_date,
        interval
    )
    cache_key_str = str(cache_key_tuple) # Use string representation as dict key

    # --- 1. Check In-Memory Cache ---
    if not force_download and cache_key_str in _memory_cache:
        log.debug(f"Loading data from memory cache for key: {cache_key_str}")
        # Return a copy to prevent modifying the cached DataFrame unintentionally
        return _memory_cache[cache_key_str].copy()

    # --- 2. Check File Cache (Parquet) ---
    cache_file = _get_cache_key(tickers_list, final_start_date, final_end_date, interval,
                                universe_name=universe_name_for_cache or (tickers_list[0] if len(tickers_list) == 1 else None))

    if not force_download and cache_file and cache_file.exists() and cache_file.is_file():
        try:
            log.debug(f"Loading cached data from file: {cache_file}")
            df = pd.read_parquet(cache_file) # Requires pyarrow
            if not df.empty:
                # Basic validation (can be more thorough)
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                actual_cols = list(df.columns.levels[1]) if isinstance(df.columns, pd.MultiIndex) else list(df.columns)
                has_ohlcv = all(col in actual_cols for col in expected_cols)

                if has_ohlcv:
                    log.debug(f"File cache validated for {cache_file}.")
                    # Store in memory cache before returning
                    _memory_cache[cache_key_str] = df
                    return df.copy() # Return a copy
                else:
                    log.warning(f"Cached file {cache_file} missing standard columns ({actual_cols}). Forcing download.")
                    force_download = True # Force download if cache is invalid
            else:
                log.warning(f"Cached file {cache_file} is empty. Forcing download.")
                force_download = True # Force download if cache is empty
        except ImportError:
            log.warning("pyarrow not installed. Cannot read from file cache.")
            # Continue to download attempt
        except Exception as e:
            log.error(f"Error reading cache file {cache_file}: {e}. Forcing download.", exc_info=False)
            force_download = True # Force download on read error

    # --- 3. Download Data ---
    # Condition: force_download is True OR file cache was missed/invalid OR not found in memory
    # (Redundant check for memory cache here, but safe)
    if force_download or cache_key_str not in _memory_cache:
        log.info(f"Fetching/Downloading price data via yfinance for: {', '.join(tickers_list)}")
        df_downloaded = None
        yf_exception = None

        try:
            ticker_string_yf = " ".join(tickers_list)
            # Use group_by='ticker' for consistent MultiIndex output
            group_by_arg = "ticker"
            log.debug(f"Calling yf.download(tickers='{ticker_string_yf}', start='{final_start_date}', end='{final_end_date + dt.timedelta(days=1)}', interval='{interval}', group_by='{group_by_arg}')")

            df_downloaded = yf.download(tickers=ticker_string_yf, start=final_start_date,
                                        end=final_end_date + dt.timedelta(days=1), # yfinance end is exclusive
                                        interval=interval,
                                        group_by=group_by_arg,
                                        auto_adjust=False, prepost=False, threads=True,
                                        progress=False, ignore_tz=True)

        except Exception as e:
            yf_exception = e
            log.error(f"Exception DIRECTLY from yf.download for {tickers_list}: {type(e).__name__} - {e}", exc_info=True)

        # --- Post-Download Processing & Validation ---
        log.debug(f"yf.download completed. Exception caught: {yf_exception is not None}")
        log.debug(f"Raw result type: {type(df_downloaded)}")
        if df_downloaded is None or df_downloaded.empty:
            log.warning(f"No data downloaded via yfinance (result is None or empty) for: {', '.join(tickers_list)}")
             # Cache empty DataFrame to prevent re-download attempts for failed tickers in this run
            df_processed = pd.DataFrame()
            _memory_cache[cache_key_str] = df_processed
            # Optionally save empty file cache too (requires pyarrow)
            if cache_file:
                try: df_processed.to_parquet(cache_file, compression="snappy")
                except Exception as e_save: log.error(f"Error saving empty cache file {cache_file}: {e_save}", exc_info=False)
            # Return None if a yfinance exception occurred, empty DF otherwise
            return None if yf_exception else df_processed

        # --- Process successful download ---
        successful_tickers = []
        df_processed = pd.DataFrame()
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # CAPITALIZED

        if isinstance(df_downloaded.columns, pd.MultiIndex):
            valid_tickers_data = {}
            for ticker in df_downloaded.columns.levels[0]:
                if ticker in tickers_list: # Process only requested tickers
                    ticker_df = df_downloaded[ticker].copy() # Work on copy
                    # Standardize columns to Capitalized
                    ticker_df.columns = [str(c).strip().capitalize() for c in ticker_df.columns]
                    if all(col in ticker_df.columns for col in expected_cols):
                        # Drop rows where Close is NaN before adding
                        valid_tickers_data[ticker] = ticker_df.dropna(subset=['Close'])
                        if not valid_tickers_data[ticker].empty:
                             successful_tickers.append(ticker)
                        else: log.debug(f"{ticker} data became empty after dropping NaN Close.")
                    else: log.warning(f"{ticker} missing standard columns: {ticker_df.columns.tolist()}. Skipping.")
            failed_tickers = [t for t in tickers_list if t not in successful_tickers]
            if failed_tickers: log.warning(f"Failed/Invalid download for: {', '.join(failed_tickers)}")
            if valid_tickers_data: df_processed = pd.concat(valid_tickers_data, axis=1)
            else: log.error("No valid data obtained for requested tickers (MultiIndex case)."); df_processed = pd.DataFrame()
        else:
            # This case should be less common with group_by='ticker'
            log.warning(f"Unexpected non-MultiIndex df structure from yfinance for {tickers_list}. Cols: {df_downloaded.columns}. Attempting processing...")
            df_temp = df_downloaded.copy()
            df_temp.columns = [str(c).strip().capitalize() for c in df_temp.columns]
            if all(col in df_temp.columns for col in expected_cols) and len(tickers_list)==1:
                 df_temp = df_temp.dropna(subset=['Close'])
                 if not df_temp.empty:
                      # Create MultiIndex for consistency if it's a single ticker result
                      df_temp.columns = pd.MultiIndex.from_product([tickers_list, df_temp.columns])
                      df_processed = df_temp
                      successful_tickers = tickers_list
                 else: log.warning(f"Single ticker {tickers_list[0]} empty after dropping NaN Close.")
            else: log.error("Cannot process non-MultiIndex structure or missing columns.")


        if df_processed.empty:
            log.error(f"No valid data obtained for requested tickers after processing.")
            # Cache empty DataFrame
            _memory_cache[cache_key_str] = df_processed
            if cache_file:
                try: df_processed.to_parquet(cache_file, compression="snappy")
                except Exception as e_save: log.error(f"Error saving empty cache file {cache_file}: {e_save}", exc_info=False)
            return df_processed # Return empty DataFrame

        # --- Cache valid, processed data ---
        # Store in memory cache
        _memory_cache[cache_key_str] = df_processed
        # Save to file cache (if possible)
        if cache_file:
             try:
                 df_processed.to_parquet(cache_file, compression="snappy")
                 log.info(f"Saved data ({len(successful_tickers)} tickers) to file cache: {cache_file}")
             except ImportError:
                  log.warning("pyarrow not installed. Cannot save to file cache.")
             except Exception as e: log.error(f"Error saving file cache {cache_file}: {e}", exc_info=False)

        return df_processed.copy() # Return a copy

    # Fallback if logic path is unexpected
    log.warning("Returning None from get_price_data (unexpected execution path).")
    return None

# --- Fundamentals Function ---
def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Retrieves basic fundamental data using yfinance Ticker info."""
    if yf is None: log.error("yfinance not installed, cannot get fundamentals."); return {}
    fundamentals = {}; tkr = yf.Ticker(ticker)
    try: info = tkr.info
    except Exception as e:
         log.warning(f"Could not get Ticker info for {ticker}: {e}")
         info = {}
    # Map yfinance keys to desired fundamental keys
    key_map = {
        "market_cap": "marketCap", "eps_ttm": "trailingEps", "pe_ratio": "trailingPE",
        "ps_ratio": "priceToSalesTrailing12Months", "sector": "sector", "industry": "industry",
        "currency": "currency", "shares_outstanding": "sharesOutstanding", "beta": "beta",
        "dividend_yield": "dividendYield", "forward_pe": "forwardPE"
    }
    for f_key, yf_key in key_map.items():
        fundamentals[f_key] = info.get(yf_key) # Use .get for safety

    # Calculate Sales TTM if possible
    try:
         mcap = fundamentals.get("market_cap")
         ps = fundamentals.get("ps_ratio")
         if mcap is not None and ps is not None and ps != 0:
             fundamentals["sales_ttm"] = mcap / ps
         else: fundamentals["sales_ttm"] = None
    except Exception: fundamentals["sales_ttm"] = None # Catch potential type errors

    return fundamentals

# --- Universe Loading Function ---
def load_universe(name: str = "sp500") -> List[str]:
    """Loads ticker symbols from a CSV file in resources directory."""
    name_lower = name.lower().strip()
    if not name_lower: raise ValueError("Universe name cannot be empty.")
    csv_filename = f"{name_lower}.csv"
    # Assume resources dir is inside the package dir
    resources_dir = _PACKAGE_ROOT / "resources"
    csv_path = resources_dir / csv_filename
    log.info(f"Attempting to load universe '{name}' from: {csv_path}")
    if not csv_path.exists() or not csv_path.is_file():
        # Try looking in parent dir as fallback (if resources is at repo root)
        resources_dir_alt = _PACKAGE_ROOT.parent / "resources"
        csv_path_alt = resources_dir_alt / csv_filename
        if csv_path_alt.exists() and csv_path_alt.is_file():
            log.warning(f"Found universe file in parent directory: {csv_path_alt}")
            csv_path = csv_path_alt
        else:
            raise ValueError(f"Universe CSV file not found at {csv_path} or {csv_path_alt}")

    symbols_list = []
    df = pd.DataFrame()
    try:
        # Try reading with different encodings if default fails
        try: df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8', on_bad_lines='warn', skipinitialspace=True)
        except UnicodeDecodeError:
             log.warning(f"UTF-8 decoding failed for {csv_path}, trying latin1.")
             df = pd.read_csv(csv_path, delimiter=',', encoding='latin1', on_bad_lines='warn', skipinitialspace=True)
        except pd.errors.ParserError as pe: raise ValueError(f"Error parsing CSV {csv_path}: {pe}") from pe
        except Exception as read_err: raise ValueError(f"Error reading CSV {csv_path}: {read_err}") from read_err

        if df.empty: log.warning(f"Universe file {csv_path} loaded empty."); return []

        # Find the symbol column (case-insensitive check)
        symbol_col = None
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in ['symbol', 'ticker', 'code', 'isin']:
                symbol_col = col; break
        if symbol_col is None:
            if len(df.columns) > 0: symbol_col = df.columns[0]; log.warning(f"No standard symbol column found. Using first column '{symbol_col}' for symbols in {csv_path}.")
            else: raise ValueError(f"No columns found in {csv_path}.")

        # Extract, clean, and filter symbols
        symbols_raw = df[symbol_col].dropna().astype(str)
        symbols_list_raw = symbols_raw.str.strip().str.upper().tolist()
        symbols_list = []
        skip_prefixes = ("PERF.", "KGV", "MARKT-", "KAUFEN", "VERKAUFEN", "#", "//")
        skip_exact = {"SYMBOL", "TICKER", "ISIN", "NAME", "NAN", ""}
        problem_chars = ('/', '^', '+', '*')

        for s in symbols_list_raw:
            if s in skip_exact or s.startswith(skip_prefixes) or any(char in s for char in problem_chars):
                log.debug(f"Skipping invalid/header/problematic symbol: {s}")
                continue
            # Add specific cleaning if needed (e.g., removing prefixes)
            # s_cleaned = s.replace('DK:','') # Example - test carefully
            symbols_list.append(s) # Append the cleaned symbol

        if not symbols_list: log.warning(f"No valid symbols extracted from {csv_path} using column '{symbol_col}'."); return []
        log.info(f"Successfully loaded and cleaned {len(symbols_list)} symbols from {csv_path}.")
        return symbols_list
    except KeyError as e: raise ValueError(f"Column '{symbol_col}' not found in {csv_path}: {e}") from e
    except Exception as e: log.error(f"Error loading universe {csv_path}: {e}", exc_info=True); raise ValueError(f"Failed loading {csv_path}") from e

# --- Explicit Export List ---
__all__ = ["get_price_data", "get_fundamentals", "load_universe", "clear_memory_cache"]