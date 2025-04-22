# run_scheduled_scan.py
"""
Script to run both leader and advanced scans for all universes
and send a combined email alert. Designed for scheduled execution.
"""
import sys
import logging
import datetime as dt
import pandas as pd

# Configure logging FIRST
log_level = logging.INFO # Or logging.DEBUG
logging.basicConfig(level=log_level, format='%(asctime)s - ScheduledScan - %(levelname)s - %(message)s', force=True)
log = logging.getLogger(__name__)

# Import leader_scan components AFTER logging is set up
try:
    from leader_scan.main import run_scan_for_all_universes
    from leader_scan.advanced_screener import run_advanced_scan_for_all
    from leader_scan.alert import dispatch
    from leader_scan.config import CONFIG
except ImportError as e:
    log.critical(f"Failed import leader_scan components: {e}. Check PYTHONPATH or if script is run from parent directory.", exc_info=True)
    sys.exit(1)
except Exception as e:
    log.critical(f"Unexpected error during imports: {e}", exc_info=True)
    sys.exit(1)

# --- Configuration ---
TOP_N = CONFIG.get("schedule_top_n", 5)
BENCHMARK = CONFIG.get("schedule_benchmark", "SPY")
DISPLAY_COLS_LEADER = ['symbol', 'date', 'setup_type', 'score', 'r_multiple', 'Close', 'stop', 'target']
DISPLAY_COLS_ADV = ['symbol', 'date', 'setup_type', 'score', 'r_multiple', 'entry', 'stop', 'target']

def format_results_for_email(results_dict: dict, title: str, display_cols: list) -> str:
    """Formats results from a dictionary {universe: DataFrame} into a string."""
    body = f"--- {title} (Top {TOP_N}) ---\n\n"
    found_any = False
    if not results_dict or not isinstance(results_dict, dict):
        body += "No results found or results format incorrect.\n"
        return body

    sorted_universes = sorted(results_dict.keys())
    for universe in sorted_universes:
        df = results_dict.get(universe)
        if df is not None and not df.empty and isinstance(df, pd.DataFrame): # Check df is DataFrame
            found_any = True
            body += f"=== {universe.upper()} ===\n"
            temp_df = df.copy()
            final_cols = []
            for col in display_cols:
                 if col in temp_df.columns: final_cols.append(col)
            if 'date' in temp_df.columns:
                 try: temp_df['date'] = pd.to_datetime(temp_df['date']).dt.strftime('%Y-%m-%d')
                 except: pass
            final_cols_existing = [c for c in final_cols if c in temp_df.columns]
            if not final_cols_existing: body+= "No display columns found in results DataFrame.\n\n"
            else: body += temp_df[final_cols_existing].to_string(index=False, float_format="%.2f") + "\n\n"
        # else: body += f"=== No results found for {universe.upper()} ===\n\n" # Optional: report empty universes

    if not found_any: body += f"No setups found matching criteria in any universe for {title}.\n"
    return body

def main():
    """Main execution function for scheduled scan."""
    log.info("Starting scheduled scan run...")
    if not all([CONFIG.get("smtp_host"), CONFIG.get("smtp_user"), CONFIG.get("smtp_password"), CONFIG.get("from_email"), CONFIG.get("to_emails")]):
        log.error("Email config incomplete via ENV VARS (LS_...). Cannot send alert.")
        sys.exit(1)

    log.info(f"Running standard leader scan (Top {TOP_N}, Benchmark {BENCHMARK})...")
    leader_results = run_scan_for_all_universes(top_per_universe=TOP_N, benchmark=BENCHMARK, return_results=True)

    log.info(f"Running advanced scan (Top {TOP_N}, Type ALL, Benchmark {BENCHMARK})...")
    advanced_results = run_advanced_scan_for_all(top_per_universe=TOP_N, setup_type_filter="ALL", benchmark=BENCHMARK, return_results=True)

    log.info("Formatting email body...")
    today_str = dt.date.today().strftime('%Y-%m-%d')
    subject = f"Leader/Advanced Scan Results - {today_str}"
    email_body = f"Scan Report for {today_str}\n"
    email_body += "="*40 + "\n\n"
    email_body += format_results_for_email(leader_results, "Standard Leader Scan", DISPLAY_COLS_LEADER)
    email_body += "="*40 + "\n\n"
    email_body += format_results_for_email(advanced_results, "Advanced Scan (All Types)", DISPLAY_COLS_ADV)
    email_body += "="*40 + "\n"

    log.info(f"Attempting to send combined email alert to: {CONFIG.get('to_emails')}")
    try:
        dispatch(subject, email_body)
        log.info("Combined email alert dispatched successfully.")
    except Exception as e:
        log.error(f"Failed to send combined email alert: {e}", exc_info=True)
        sys.exit(1)

    log.info("Scheduled scan run finished successfully.")

if __name__ == "__main__":
    main()