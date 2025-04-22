# leader_scan/find_symbol.py
# NOTE: This file seems to contain a partial/full definition of LeadershipScanner
# including the `load_universe` method. It might be better integrated into `main.py`
# or a dedicated `scanner_class.py`. Keeping the structure as found for now.

import sys
import pandas as pd
from pathlib import Path
import logging # Use logging instead of print for debug/warnings

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import yahoo_fin safely
try:
    import yahoo_fin.stock_info as si
except ImportError:
    logging.warning("yahoo_fin library not found. Dynamic universe loading (S&P 500, Nasdaq) will not work.")
    si = None # Set to None if not available

# Assume CONFIG is loaded elsewhere (e.g., in main.py or __init__)
# from .config import CONFIG

class LeadershipScanner:
    """
    (Partial class definition as found in find_symbol.py)
    Handles loading stock universes.
    """

    def __init__(self, universe_name: str):
        """
        Initializes the scanner by loading the specified universe.
        """
        self.universe_name = universe_name
        self.symbols_list = self.load_universe(universe_name)
        # Check if symbols were loaded successfully
        if not self.symbols_list:
            logging.warning(f"No symbols loaded for universe '{universe_name}'. Scanner may not function correctly.")
            # Consider raising an error or using a default list if appropriate
            # raise ValueError(f"Could not load symbols for universe: {universe_name}")
        self.data_cache = {} # Example: Initialize a cache if needed
        self.benchmark_data = None # Example: Placeholder for benchmark data

    def load_universe(self, universe_name: str) -> list[str]:
        """
        Loads the list of symbols for the specified universe.
        Supports predefined CSV files and dynamic fetching (if yahoo_fin is available).
        """
        clean_universe_name = universe_name.strip().lower()
        logging.info(f"Attempting to load universe: '{clean_universe_name}'")

        # --- Dynamic Universe Loading (S&P 500 + Nasdaq) ---
        if clean_universe_name == "dynamic_sp_nasdaq":
            if si is None:
                 logging.error("Cannot load 'dynamic_sp_nasdaq' universe because yahoo_fin is not installed.")
                 return []
            logging.info("Fetching dynamic universe (S&P 500 + Nasdaq)...")
            try:
                sp500_symbols = si.tickers_sp500(include_company_data=False) # Fetch only symbols
                nasdaq_symbols = si.tickers_nasdaq(include_company_data=False)
                # Combine, ensure uniqueness, sort
                combined_symbols = sorted(list(set(sp500_symbols + nasdaq_symbols)))
                logging.info(f"Fetched {len(combined_symbols)} unique symbols from S&P 500 & Nasdaq.")
                # Filter symbols: Remove those containing '.' or '/' often problematic for yfinance
                # Also remove common non-stock symbols or indices if necessary
                filtered_symbols = [
                    s for s in combined_symbols
                    if isinstance(s, str) and '.' not in s and '/' not in s and '^' not in s
                ]
                logging.info(f"Filtered to {len(filtered_symbols)} symbols.")
                return filtered_symbols
            except Exception as e:
                logging.error(f"Error fetching dynamic universe 'dynamic_sp_nasdaq': {e}")
                return [] # Return empty list on error

        # --- CSV Universe Loading ---
        # Use the load_universe function from data.py for consistency and robustness
        try:
            # Assuming data.py is in the same package
            from .data import load_universe as load_from_data_module
            symbols = load_from_data_module(clean_universe_name)
            return symbols
        except ImportError:
             logging.error("Could not import load_universe from data.py. Ensure it exists.")
             return []
        except ValueError as e:
             logging.error(f"Error loading universe '{clean_universe_name}' from data module: {e}")
             return []
        except Exception as e:
            logging.error(f"Unexpected error loading universe '{clean_universe_name}': {e}")
            return []

    # --- Other potential methods of LeadershipScanner ---
    # def fetch_data_for_symbol(self, symbol: str): ...
    # def score(self): ...
    # def run(self): ...

# --- Example Usage (if run directly) ---
if __name__ == "__main__":
    logging.info("Testing LeadershipScanner universe loading...")

    # Test loading from CSV (assuming sp500.csv exists in resources)
    print("\n--- Testing CSV Load (sp500) ---")
    scanner_sp500 = LeadershipScanner("sp500")
    print(f"Loaded {len(scanner_sp500.symbols_list)} symbols for sp500.")
    if scanner_sp500.symbols_list:
        print(f"First 10: {scanner_sp500.symbols_list[:10]}")

    # Test loading non-existent CSV
    print("\n--- Testing Non-Existent CSV Load (nonexistent) ---")
    scanner_none = LeadershipScanner("nonexistent")
    print(f"Loaded {len(scanner_none.symbols_list)} symbols for nonexistent.")

    # Test dynamic loading (if yahoo_fin installed)
    if si:
        print("\n--- Testing Dynamic Load (dynamic_sp_nasdaq) ---")
        scanner_dynamic = LeadershipScanner("dynamic_sp_nasdaq")
        print(f"Loaded {len(scanner_dynamic.symbols_list)} symbols for dynamic_sp_nasdaq.")
        if scanner_dynamic.symbols_list:
            print(f"First 10: {scanner_dynamic.symbols_list[:10]}")
    else:
        print("\n--- Skipping Dynamic Load Test (yahoo_fin not installed) ---")
