# leader_scan/trade_logger.py
"""
Trade journal for tracking planned and executed trades.
"""
import os
import csv
import datetime
import sys
from pathlib import Path
from typing import Dict, Optional, Union, List

# Assuming CONFIG is imported correctly and handles its loading
try:
    from .config import CONFIG
except ImportError:
    print("Error: Could not import CONFIG. Ensure config.py exists and is accessible.", file=sys.stderr)
    # Provide default config or exit if critical
    CONFIG = {"journal_path": "trades.csv"}


def _get_journal_path() -> Path:
    """Gets the journal path from config or uses a default."""
    default_path = Path("trades.csv").resolve() # Default in current working dir
    journal_path_str = CONFIG.get("journal_path", str(default_path))
    try:
        path = Path(journal_path_str).resolve()
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        print(f"Error resolving journal path '{journal_path_str}': {e}. Using default: {default_path}", file=sys.stderr)
        return default_path

def log_planned_trade(
    symbol: str,
    setup_type: str,
    entry: float,
    stop: float,
    target: float,
    position_size: Optional[Union[float, int]] = None,
    notes: str = ""
) -> None:
    """
    Log a planned trade to the journal CSV file.

    Args:
        symbol: Stock ticker symbol (cleaned).
        setup_type: Type of setup (e.g., 'LEADER', 'STAGE2', 'MA_CROSS').
        entry: Planned entry price.
        stop: Stop loss price.
        target: Price target.
        position_size: Optional position size in shares.
        notes: Optional notes about the trade setup.
    """
    if not all([symbol, setup_type, isinstance(entry, (int, float)), isinstance(stop, (int, float)), isinstance(target, (int, float))]):
        print("Error: Missing or invalid required arguments for logging planned trade.", file=sys.stderr)
        return

    path = _get_journal_path()

    # Calculate metrics safely
    try:
        risk_abs = entry - stop
        reward_abs = target - entry

        if entry <= 0: # Avoid division by zero/nonsensical percentages
            risk_pct = 0.0
            reward_pct = 0.0
        else:
            risk_pct = round((risk_abs / entry) * 100, 2) if risk_abs > 0 else 0.0
            reward_pct = round((reward_abs / entry) * 100, 2) if reward_abs > 0 else 0.0

        # R-multiple: reward / risk
        r_multiple = round(reward_abs / risk_abs, 2) if risk_abs > 0 else 0.0

    except (TypeError, ZeroDivisionError) as e:
        print(f"Error calculating trade metrics for {symbol}: {e}", file=sys.stderr)
        risk_pct = reward_pct = r_multiple = 0.0

    # Prepare new row
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # Define expected headers explicitly
    fieldnames = [
        "date_planned", "symbol", "setup_type", "entry", "stop", "target",
        "risk_pct", "reward_pct", "r_multiple", "position_size", "status",
        "date_entered", "date_exited", "exit_price", "pnl_pct", "notes" # Changed pnl to pnl_pct
    ]
    new_row = {
        "date_planned": today,
        "symbol": str(symbol).strip().upper(),
        "setup_type": str(setup_type).strip().upper(),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "r_multiple": r_multiple,
        "position_size": position_size if position_size is not None else "",
        "status": "PLANNED",
        "date_entered": "",
        "date_exited": "",
        "exit_price": "",
        "pnl_pct": "", # Initialize PnL percentage
        "notes": str(notes)
    }

    # Check if file exists to determine if header needs writing
    file_exists = path.is_file()

    try:
        with open(path, mode='a', newline='', encoding='utf-8') as f:
            # Use defined fieldnames to ensure consistency
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or path.stat().st_size == 0: # Write header if new or empty file
                writer.writeheader()
            writer.writerow(new_row)
        print(f"Trade logged: {symbol} {setup_type} setup | R: {r_multiple:.2f} | Risk: {risk_pct:.2f}% | Target: {target:.2f}")
    except IOError as e:
        print(f"Error writing planned trade to journal {path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during planned trade logging: {e}", file=sys.stderr)


def log_exit(
    symbol: str,
    exit_price: float,
    status: str = "EXITED",
    exit_notes: str = "" # Renamed notes to avoid confusion
) -> None:
    """
    Update the most recent 'PLANNED' trade for a symbol with exit information.

    Args:
        symbol: Stock ticker symbol to match.
        exit_price: Price at which position was exited.
        status: Trade status (EXITED, STOPPED, TARGETED, CANCELED).
        exit_notes: Optional notes specifically about the exit.
    """
    if not symbol or not isinstance(exit_price, (int, float)):
         print("Error: Missing or invalid required arguments for logging exit.", file=sys.stderr)
         return

    path = _get_journal_path()
    clean_symbol = str(symbol).strip().upper()
    valid_statuses = ["EXITED", "STOPPED", "TARGETED", "CANCELED"]
    clean_status = str(status).strip().upper()
    if clean_status not in valid_statuses:
        print(f"Warning: Invalid status '{status}'. Using 'EXITED'. Valid options: {valid_statuses}", file=sys.stderr)
        clean_status = "EXITED"

    if not path.is_file() or path.stat().st_size == 0:
        print(f"Journal file not found or empty at {path}", file=sys.stderr)
        return

    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    try:
        with open(path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Error: Could not read header from journal file {path}.", file=sys.stderr)
                return
            fieldnames = reader.fieldnames # Store original fieldnames
            # Ensure required fields exist
            required_fields = ["symbol", "status", "entry", "exit_price", "date_exited", "pnl_pct", "notes"]
            if not all(field in fieldnames for field in required_fields):
                 print(f"Error: Journal file {path} is missing required columns. Expected: {required_fields}", file=sys.stderr)
                 return
            rows = list(reader)
    except FileNotFoundError:
         print(f"Journal file not found at {path}", file=sys.stderr)
         return
    except Exception as e:
        print(f"Error reading journal file {path}: {e}", file=sys.stderr)
        return

    if not rows:
        print(f"No trades found in the journal file {path}.", file=sys.stderr)
        return

    # Find the most recent 'PLANNED' trade for this symbol
    updated = False
    # Iterate backwards to find the *last* matching planned trade
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if row.get('symbol') == clean_symbol and row.get('status') == 'PLANNED':
            try:
                entry_str = row.get('entry', '0.0')
                entry = float(entry_str) if entry_str else 0.0

                if entry <= 0:
                     pnl_pct = 0.0
                     print(f"Warning: Cannot calculate PnL for {symbol} due to invalid entry price ({entry_str}).", file=sys.stderr)
                else:
                    pnl_pct = round(((exit_price - entry) / entry) * 100, 2)

                # Update the row in the list
                rows[i]['exit_price'] = str(exit_price)
                rows[i]['date_exited'] = datetime.datetime.now().strftime("%Y-%m-%d")
                rows[i]['status'] = clean_status
                rows[i]['pnl_pct'] = str(pnl_pct) # Store PnL percentage
                # Append exit notes carefully
                original_notes = rows[i].get('notes', '')
                if exit_notes:
                    rows[i]['notes'] = f"{original_notes} | Exit: {exit_notes}" if original_notes else f"Exit: {exit_notes}"
                updated = True
                break # Exit loop once the latest planned trade is updated
            except (ValueError, TypeError) as e:
                 print(f"Error processing trade data for update (Symbol: {symbol}, Row: {i}): {e}", file=sys.stderr)
                 # Continue searching in case this row was corrupt
                 continue

    if updated:
        # Write updated rows back to file using original fieldnames
        try:
            with open(path, mode='w', newline='', encoding='utf-8') as f:
                # Use the fieldnames read from the file initially
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            pnl_value = rows[i]['pnl_pct'] # Get the calculated PnL to print
            print(f"Updated trade for {clean_symbol}: {clean_status} at {exit_price:.2f} (PnL: {pnl_value}%)")
        except IOError as e:
            print(f"Error writing updated journal to {path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred while writing updated journal: {e}", file=sys.stderr)
    else:
        print(f"No 'PLANNED' trade found for symbol '{clean_symbol}' to update.")

# Example Usage (optional, for testing)
if __name__ == "__main__":
    print(f"Using journal file: {_get_journal_path()}")
    # Example calls
    log_planned_trade("TEST", "STAGE2", 100.0, 95.0, 115.0, 10, "Test plan")
    log_planned_trade("XYZ", "LEADER", 50.0, 48.0, 56.0, notes="Another test")
    log_exit("TEST", 110.0, status="EXITED", exit_notes="Partial profit take")
    log_exit("XYZ", 47.5, status="STOPPED", exit_notes="Market reversed")
    log_exit("NOSYMBOL", 20.0) # Test non-existent symbol
