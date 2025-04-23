# leader_scan/__init__.py
"""
Leader Scan Package
-------------------
A stock screening tool to identify potential trading setups based on technical
and fundamental analysis principles.

Core Modules:
    * config: Manages configuration settings.
    * data: Handles data fetching, caching, and universe loading.
    * indicators: Provides technical indicator calculation functions.
    * filters: Contains boolean filters for detecting price action patterns.
    * scorer: Implements the main scoring engine and setup classification.
    * main: Orchestrates the scanning process and provides main entry points.
    * alert: Facilitates sending notifications (Slack, Email).
    * trade_logger: Enables logging of planned and executed trades.
    * screener: (Optional) Contains logic for specific setup scans like Stage 2.
    * advanced_screener: (Optional) Script for running combined scans.

Example Usage:
    >>> from leader_scan import run_daily_scan, CONFIG
    >>> # Run default scan
    >>> leaders = run_daily_scan()
    >>> print(leaders)
    >>>
    >>> # Access configuration
    >>> print(CONFIG.get("min_r_multiple"))

"""

__version__ = "0.2.0" # Updated version number

# --- Public API Exports ---
# Explicitly define what is available when importing `from leader_scan import *`
# or accessed directly like `leader_scan.CONFIG`.

# Core components:
from .config import CONFIG
from .main import run_daily_scan, LeadershipScanner # Main entry points
from .data import get_price_data, get_fundamentals, load_universe
from .scorer import score_dataframe # Core scoring function

# Optional components (users can import directly if needed):
# from .indicators import * # Or specific indicators
# from .filters import * # Or specific filters
# from .alert import dispatch, send_slack, send_email
# from .trade_logger import log_planned_trade, log_exit
# from .screener import find_new_stage2_stocks # If still used standalone
# from .advanced_screener import find_top_setups # If used as a library function

# Define __all__ for `import *` behavior
__all__ = [
    # Core
    "CONFIG",
    "run_daily_scan",
    "LeadershipScanner",
    "get_price_data",
    "get_fundamentals",
    "load_universe",
    "score_dataframe",
    # Add optional components if they should be included in `import *`
    # "dispatch",
    # "log_planned_trade",
    # "log_exit",
    # "find_new_stage2_stocks",
]

# --- Initialization Code (if any) ---
# (e.g., setting up logging, checking dependencies)
# print("Leader Scan package initialized.") # Optional init message