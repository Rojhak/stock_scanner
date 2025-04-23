# /Users/fehmikatar/Desktop/Stock/leader_scan/config.py
"""
Global configuration for the leader_scan package.

This module defines a central, mutable dictionary `CONFIG` that all sub-modules
import. Configuration values are loaded with the following priority (later
sources override earlier ones):

1. Hard-coded defaults defined in this file.
2. Values from a JSON file located at `~/.leader_scan.json`.
3. Environment variables prefixed with `LS_`.

Example Environment Variable Usage:
    export LS_SLACK_WEBHOOK="https://hooks.slack.com/services/XYZ"
    export LS_TO_EMAILS="user1@example.com,user2@example.com"
"""

import json
import os
import pathlib
from typing import Dict, Any, Optional, List, Union

# --------------------------------------------------------------------------- #
# Defaults – Can be overridden by JSON file or environment variables
# --------------------------------------------------------------------------- #
CONFIG: Dict[str, Any] = {
    # --- Alerts ------------------------------------------------------------ #
    "slack_webhook": None,  # Optional[str]: Slack Incoming Webhook URL
    "smtp_host": None,      # Optional[str]: SMTP server hostname
    "smtp_port": 587,       # int: SMTP server port (587 for TLS)
    "smtp_user": None,      # Optional[str]: SMTP username
    "smtp_password": None,  # Optional[str]: SMTP password or app key
    "from_email": None,     # Optional[str]: Email address to send alerts from
    "to_emails": None,      # Optional[Union[str, List[str]]]: Comma-separated string or list of recipient emails

    # --- Screening thresholds --------------------------------------------- #
    "drawdown_max": 0.25,           # float: Maximum allowed drawdown (e.g., 0.25 for 25%)
    "rs_new_high_lookback": 252,    # int: Lookback period (days) for RS new high
    "volume_thrust_multiple": 1.4,  # float: Volume must be this multiple of avg volume
    "eps_growth_min": 0.25,         # float: Minimum TTM EPS growth (e.g., 0.25 for 25%)
    "sales_growth_min": 0.20,       # float: Minimum TTM Sales growth (e.g., 0.20 for 20%)
    "risk_per_trade": 0.01,         # float: Default risk percentage per trade (e.g., 0.01 for 1%)
    "min_r_multiple": 2.5,          # float: Minimum required reward-to-risk ratio
    "journal_path": "trades.csv",   # str: Path to the trade journal CSV file

    # --- Data sources ------------------------------------------------------ #
    "universe": "sp1500",           # str: Default stock universe (e.g., "sp500", "sp1500")
    "price_interval": "1d",         # str: Data interval ("1d", "1wk", "1mo")
    "cache_days": 730,              # int: Number of days of price data to cache (approx 2 years)
    "min_data_rows": 50,            # int: Minimum rows needed for indicator calculation / scoring

    # --- Added for run_scheduled_scan.py --- #
    "schedule_benchmark": "SPY",    # str: Benchmark used in scheduled runs
    "schedule_top_n": 5,            # int: Default top N for scheduled runs (can be overridden)

    # --- Internal flags (usually set by CLI args) --- #
    "force_download_flag": False    # bool: Flag to force data download, ignoring cache
}


# --------------------------------------------------------------------------- #
# Helper Functions – Apply overrides from JSON and Environment Variables
# --------------------------------------------------------------------------- #
def _apply_json(filepath: pathlib.Path, config_dict: Dict[str, Any]) -> None:
    """Loads config from a JSON file if it exists and updates the config dict."""
    if filepath.exists() and filepath.is_file():
        try:
            with filepath.open('r', encoding='utf-8') as fp:
                json_config = json.load(fp)
                config_dict.update(json_config)
                print(f"[config] Loaded overrides from {filepath}")
        except json.JSONDecodeError as exc:
            print(f"[config] Error parsing JSON file {filepath}: {exc}")
        except Exception as exc:
            print(f"[config] Failed to load config file {filepath}: {exc}")


def _apply_env(config_dict: Dict[str, Any]) -> None:
    """Overrides config with environment variables prefixed with 'LS_'."""
    prefix = "LS_"
    updated = False
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            if config_key in config_dict:
                # Attempt type casting based on default value type
                default_value = config_dict[config_key]
                try:
                    if isinstance(default_value, bool):
                        # Handle boolean conversion (e.g., "true", "1", "false", "0")
                        config_dict[config_key] = value.lower() in ['true', '1', 'yes', 'y']
                    elif isinstance(default_value, int):
                        config_dict[config_key] = int(value)
                    elif isinstance(default_value, float):
                        config_dict[config_key] = float(value)
                    elif config_key == "to_emails": # Special handling for to_emails list
                         config_dict[config_key] = [email.strip() for email in value.split(',') if email.strip()]
                    else: # Default to string
                        config_dict[config_key] = value
                    updated = True
                    print(f"[config] Applied ENV override: {config_key} = {config_dict[config_key]}")
                except ValueError:
                    print(f"[config] Warning: Could not cast env var {key}={value} for {config_key}. Using raw string value.")
                    config_dict[config_key] = value # Keep as string if cast fails
            # else:
                 # Optional: Warn about unknown env vars
                 # print(f"[config] Warning: Environment variable {key} does not match any default config key.")

    # if updated: print("[config] Applied environment variable overrides.")


# --------------------------------------------------------------------------- #
# Initialization - Load overrides
# --------------------------------------------------------------------------- #
_CONFIG_TEMP = CONFIG.copy() # Work on a temporary copy

# 1. Apply JSON overrides (optional config file in user's home directory)
_user_config_file = pathlib.Path.home() / ".leader_scan.json"
_apply_json(_user_config_file, _CONFIG_TEMP)

# 2. Apply Environment variable overrides
_apply_env(_CONFIG_TEMP)

# 3. Update the main CONFIG dictionary
CONFIG.update(_CONFIG_TEMP)

# Ensure 'to_emails' is a list, even if set via JSON or default was None/str
if not isinstance(CONFIG.get("to_emails"), list):
    if isinstance(CONFIG.get("to_emails"), str):
         CONFIG["to_emails"] = [e.strip() for e in CONFIG["to_emails"].split(',') if e.strip()]
    else:
        CONFIG["to_emails"] = [] # Ensure it's at least an empty list

# Optional: Make the config read-only after initialization if desired
# from types import MappingProxyType
# CONFIG = MappingProxyType(CONFIG) # Makes the dictionary read-only

# --------------------------------------------------------------------------- #
# __all__ - Explicitly define public exports
# --------------------------------------------------------------------------- #
__all__ = ["CONFIG"]

# Example usage print after loading:
# print("\n[config] Final Configuration:")
# for k, v in CONFIG.items():
#     # Hide password in printout
#     print_val = '********' if 'password' in k and v else v
#     print(f"  {k}: {print_val} ({type(v).__name__})")