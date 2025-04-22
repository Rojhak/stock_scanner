# leader_scan/config.py
import os
import json
import sys
from pathlib import Path
from typing import Any

# --- Configuration Loading ---
_config_path = Path(__file__).parent / 'config.json' # Optional: Load base config from JSON
_config_data = {}
# Load base config from JSON if it exists
if _config_path.exists():
    try:
        with open(_config_path, 'r') as f:
            _config_data = json.load(f)
            # Use print as logging might not be set up yet when this module loads
            print(f"Config: Loaded base config from {_config_path}")
    except Exception as e:
        print(f"Warning: Could not load config.json: {e}", file=sys.stderr)
else:
    print(f"Config: No config.json found at {_config_path}, using ENV VARS/defaults.")

# --- Helper Function to Get Config ---
def _get_config_value(key: str, default: Any = None) -> Any:
    """Gets config value: Environment Variable > JSON file > Default."""
    # Environment variables take highest priority (use LS_ prefix)
    env_var_key = f"LS_{key.upper()}"
    env_var = os.environ.get(env_var_key)
    if env_var is not None:
        # Attempt to parse common types (int, float, bool)
        if env_var.lower() in ['true', 'false']:
             print(f"Config: Read '{key}' from ENV var '{env_var_key}' as bool.")
             return env_var.lower() == 'true'
        try:
             val = int(env_var)
             print(f"Config: Read '{key}' from ENV var '{env_var_key}' as int.")
             return val
        except ValueError: pass
        try:
             val = float(env_var)
             print(f"Config: Read '{key}' from ENV var '{env_var_key}' as float.")
             return val
        except ValueError: pass
        print(f"Config: Read '{key}' from ENV var '{env_var_key}' as string.")
        return env_var # Return as string otherwise

    # Fallback to JSON config data
    if key in _config_data:
        print(f"Config: Read '{key}' from JSON file.")
        return _config_data[key]

    # Fallback to provided default
    print(f"Config: Using default for '{key}'.")
    return default

# --- Configuration Dictionary ---
CONFIG = {
    # Alerting (Reads from Env Vars first, then JSON/defaults)
    "slack_webhook": _get_config_value("slack_webhook", None),
    "smtp_host": _get_config_value("smtp_host", None),
    "smtp_port": _get_config_value("smtp_port", 587),
    "smtp_user": _get_config_value("smtp_user", None),
    "smtp_password": _get_config_value("smtp_password", None),
    "from_email": _get_config_value("from_email", None),
    "to_emails": _get_config_value("to_emails", None), # Comma-separated string or list

    # Scanner settings (Reads from JSON or uses defaults)
    "drawdown_max": _get_config_value("drawdown_max", 0.25),
    "rs_new_high_lookback": _get_config_value("rs_new_high_lookback", 252),
    "volume_thrust_multiple": _get_config_value("volume_thrust_multiple", 1.4),
    "eps_growth_min": _get_config_value("eps_growth_min", 0.25),
    "sales_growth_min": _get_config_value("sales_growth_min", 0.2),
    "risk_per_trade": _get_config_value("risk_per_trade", 0.01),
    "min_r_multiple": _get_config_value("min_r_multiple", 2.5),
    "journal_path": _get_config_value("journal_path", "trades.csv"),
    "universe": _get_config_value("universe", "sp1500"),
    "price_interval": _get_config_value("price_interval", "1d"),
    "cache_days": _get_config_value("cache_days", 730),
    "min_data_rows": _get_config_value("min_data_rows", 50),
    "schedule_top_n": _get_config_value("schedule_top_n", 5), # For wrapper script
    "schedule_benchmark": _get_config_value("schedule_benchmark", "SPY") # For wrapper script
}

# Ensure to_emails is processed into a list if it's a string or None
if isinstance(CONFIG["to_emails"], str):
    CONFIG["to_emails"] = [e.strip() for e in CONFIG["to_emails"].split(',') if e.strip()]
elif not isinstance(CONFIG["to_emails"], list):
     CONFIG["to_emails"] = [] # Default to empty list if invalid type/None

# Optional: Print loaded config (excluding password) when module is loaded
# print("--- CONFIG Loaded ---")
# for key, value in CONFIG.items():
#     print_val = '********' if 'password' in key.lower() and value else value
#     print(f"  {key}: {print_val}")
# print("--------------------")