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
_CACHE_DIR = _PACKAGE_ROOT.parent / ".cache" # Place .cache in parent of leader_scan
try:
    _CACHE_DIR.mkdir(exist_ok=True)
except OSError as e:
    log.warning(f"Could not create cache directory at {_CACHE_DIR}: {e}")
    _CACHE_DIR = _PACKAGE_ROOT / ".cache_fallback" # Fallback inside package
    try:
         _CACHE_DIR.mkdir(exist_ok=True)
         log.warning(f"Using fallback cache directory: {_CACHE_DIR}")
    except OSError as e_fallback:
         log.error(f"Could not create fallback cache directory: {e_fallback}. Caching disabled.")
         _CACHE_DIR = None

_DEFAULT_CACHE_DAYS = CONFIG.get("cache_days", 730) # Use config

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
    """Generates cache key. Uses universe_name/ticker if provided, else hashes."""
    if _CACHE_DIR is None: return None
    date_format = "%Y%m%d"
    ticker_list = list(tickers) if tickers else [] # Ensure list for checks
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

    # Check cache
    if not force_download and cache_file and cache_file.exists() and cache_file.is_file():
        try:
            log.debug(f"Loading cached data from: {cache_file}")
            df = pd.read_parquet(cache_file)
            if not df.empty:
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if isinstance(df.columns, pd.MultiIndex):
                     actual_cols = list(df.columns.levels[1]) if len(df.columns.levels) > 1 else []
                else: actual_cols = list(df.columns)
                has_ohlcv = all(col in actual_cols for col in expected_cols)
                if has_ohlcv: log.debug(f"Cache validated for {cache_file}."); return df
                else: log.warning(f"Cached file {cache_file} missing columns ({actual_cols}). Forcing download."); force_download = True
            else: log.warning(f"Cached file {cache_file} is empty. Forcing download."); force_download = True
        except Exception as e: log.error(f"Error reading cache {cache_file}: {e}. Forcing download.", exc_info=False); force_download = True

    # Download data
    if force_download or cache_file is None or not cache_file.exists():
        log.info(f"Downloading/fetching price data for: {', '.join(tickers_list)}")
        df_downloaded = None; yf_exception = None
        try:
            ticker_string_yf = " ".join(tickers_list)
            # Force group_by='ticker' for consistent MultiIndex output
            group_by_arg = "ticker"
            log.debug(f"Calling yf.download(tickers='{ticker_string_yf}', start='{final_start_date}', end='{final_end_date + dt.timedelta(days=1)}', interval='{interval}', group_by='{group_by_arg}')")
            df_downloaded = yf.download(tickers=ticker_string_yf, start=final_start_date,
                                        end=final_end_date + dt.timedelta(days=1), interval=interval,
                                        group_by=group_by_arg, auto_adjust=False, prepost=False,
                                        threads=True, progress=False, ignore_tz=True)
        except Exception as e: yf_exception = e; log.error(f"Exception DIRECTLY from yf.download for {tickers_list}: {type(e).__name__} - {e}", exc_info=True)

        log.debug(f"yf.download completed. Exception caught: {yf_exception is not None}"); log.debug(f"Raw result type: {type(df_downloaded)}")
        if df_downloaded is not None:
             log.debug(f"Raw result empty: {df_downloaded.empty}")
             if not df_downloaded.empty: log.debug(f"Raw result columns: {df_downloaded.columns}"); log.debug(f"Raw result head:\n{df_downloaded.head(2)}")
             else: log.warning("yf.download returned an empty DataFrame.")
        else: log.warning("yf.download returned None.")

        if df_downloaded is None or df_downloaded.empty:
            log.warning(f"No data downloaded (result is None or empty) for tickers: {', '.join(tickers_list)}")
            if cache_file: try: pd.DataFrame().to_parquet(cache_file, compression="snappy"); except Exception as e: log.error(f"Error saving empty cache {cache_file}: {e}")
            return None if yf_exception else pd.DataFrame()

        # Post-Download Processing & Validation
        successful_tickers = []; df_processed = pd.DataFrame(); expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if isinstance(df_downloaded.columns, pd.MultiIndex):
            valid_tickers_data = {}
            for ticker in df_downloaded.columns.levels[0]:
                if ticker in tickers_list:
                     ticker_df = df_downloaded[ticker]; ticker_df.columns = [str(c).strip().capitalize() for c in ticker_df.columns]
                     if all(col in ticker_df.columns for col in expected_cols):
                         valid_tickers_data[ticker] = ticker_df; successful_tickers.append(ticker)
                     else: log.warning(f"{ticker} missing standard cols: {ticker_df.columns.tolist()}")
            failed_tickers = [t for t in tickers_list if t not in successful_tickers]
            if failed_tickers: log.warning(f"Failed download for: {', '.join(failed_tickers)}")
            if valid_tickers_data: df_processed = pd.concat(valid_tickers_data, axis=1)
            else: log.error("No valid data obtained for requested tickers (MultiIndex case)."); df_processed = pd.DataFrame()
        else: log.error(f"Unexpected df structure from yfinance for {tickers_list}. Cols: {df_downloaded.columns}")

        if df_processed.empty:
            log.error(f"No valid data obtained for requested tickers after processing.")
            if cache_file: try: pd.DataFrame().to_parquet(cache_file, compression="snappy"); except Exception as e: log.error(f"Error saving empty cache {cache_file}: {e}")
            return pd.DataFrame()

        # Save valid, processed data to cache
        if cache_file:
             try:
                 df_processed.to_parquet(cache_file, compression="snappy")
                 log.info(f"Saved data ({len(successful_tickers)} tickers) to cache: {cache_file}")
             except Exception as e: log.error(f"Error saving cache file {cache_file}: {e}", exc_info=False)
        return df_processed

    log.debug("Returning None from get_price_data (unexpected path).")
    return None

# --- Fundamentals (unchanged) ---
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

# --- Universe Loading (Removed .replace('.', '-')) ---
def load_universe(name: str = "sp1500") -> List[str]:
    """Loads ticker symbols from a CSV file in resources. Uses symbols directly."""
    name_lower = name.lower().strip()
    if not name_lower: raise ValueError("Universe name cannot be empty.")
    csv_filename = f"{name_lower}.csv"
    resources_dir = _PACKAGE_ROOT / "resources" # Correct path relative to data.py
    csv_path = resources_dir / csv_filename
    log.info(f"Attempting to load universe '{name}' from: {csv_path}")
    if not csv_path.exists() or not csv_path.is_file():
        raise ValueError(f"Universe CSV file not found at: {csv_path}")

    symbols_list = []
    df = pd.DataFrame()
    try:
        try: df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8', on_bad_lines='skip', skipinitialspace=True)
        except TypeError: df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8', error_bad_lines=False, skipinitialspace=True)
        except pd.errors.ParserError as pe: raise ValueError(f"Error reading {csv_path}: {pe}") from pe
        except UnicodeDecodeError:
             log.warning(f"UTF-8 failed for {csv_path}, trying latin1.")
             try: df = pd.read_csv(csv_path, delimiter=',', encoding='latin1', on_bad_lines='skip', skipinitialspace=True)
             except TypeError: df = pd.read_csv(csv_path, delimiter=',', encoding='latin1', error_bad_lines=False, skipinitialspace=True)
        except Exception as read_err: raise ValueError(f"Error reading {csv_path}: {read_err}") from read_err
        if df.empty: log.warning(f"{csv_path} loaded empty."); return []

        symbol_col = None
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower == 'symbol': symbol_col = col; break
            if 'ticker' in col_lower: symbol_col = col
            if 'code' in col_lower and not symbol_col: symbol_col = col
            if 'isin' in col_lower and not symbol_col: symbol_col = col
        if symbol_col is None:
            if len(df.columns) > 0: symbol_col = df.columns[0]; log.warning(f"Using first column '{symbol_col}' for symbols in {csv_path}.")
            else: raise ValueError(f"No columns found in {csv_path}.")

        symbols_raw = df[symbol_col].dropna().astype(str)
        symbols_list_raw = symbols_raw.str.strip().str.upper().tolist()
        symbols_list = []
        for s in symbols_list_raw:
            if s in ["SYMBOL", "TICKER", "ISIN", "NAME", "NAN", ""] or s.startswith(("PERF.", "KGV", "MARKT-", "KAUFEN", "VERKAUFEN", "#", "//")): continue
            s_cleaned = s # Use symbol directly from CSV
            if '/' in s_cleaned or '^' in s_cleaned or '+' in s_cleaned or '*' in s_cleaned:
                 log.debug(f"Skipping potentially problematic symbol format: {s}")
                 continue
            symbols_list.append(s_cleaned)

        if not symbols_list: log.warning(f"No valid symbols extracted from {csv_path} using '{symbol_col}'."); return []
        log.info(f"Successfully loaded and cleaned {len(symbols_list)} symbols from {csv_path}.")
        return symbols_list
    except KeyError as e: raise ValueError(f"Column '{symbol_col}' not found in {csv_path}: {e}") from e
    except Exception as e: log.error(f"Error loading universe {csv_path}: {e}", exc_info=True); raise ValueError(f"Failed loading {csv_path}") from e

__all__ = ["get_price_data", "get_fundamentals", "load_universe"]