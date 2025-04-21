# leader_scan/alert.py
"""
Alerting utilities for sending notifications via Slack and Email.
Reads configuration from the central `CONFIG` object.
Handles potential errors during sending gracefully.
"""
import smtplib
import ssl
import json
import logging
import urllib.request
import urllib.error
from email.message import EmailMessage
from typing import Sequence, Dict, Any, Optional, List, Union

# Use absolute import
from .config import CONFIG

log = logging.getLogger(__name__)

# --- Constants ---
REQUEST_TIMEOUT = 15 # Seconds for HTTP requests
SMTP_TIMEOUT = 30 # Seconds for SMTP connection

# --- Slack Alerting ---
def _post_json_to_slack(url: str, payload: Dict[str, Any]) -> bool:
    """Sends a JSON payload to a Slack webhook URL using urllib."""
    if not url: log.error("Slack post failed: Webhook URL is empty."); return False
    try:
        data = json.dumps(payload).encode("utf-8"); headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        log.debug(f"Sending Slack POST to {url} with payload: {payload}")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            status_code = response.getcode(); log.debug(f"Slack response status: {status_code}")
            if 200 <= status_code < 300: log.info("Slack message sent."); return True
            else: log.error(f"Slack post failed ({status_code}): {response.read().decode('utf-8')}"); return False
    except urllib.error.HTTPError as e: log.error(f"Slack HTTPError: {e.code} - {e.reason}. Response: {e.read().decode('utf-8','ignore')}"); return False
    except urllib.error.URLError as e: log.error(f"Slack URLError: {e.reason}"); return False
    except Exception as e: log.error(f"Slack unexpected error: {e}", exc_info=True); return False

def send_slack(message: str) -> bool:
    """Sends a message to the configured Slack webhook."""
    webhook = CONFIG.get("slack_webhook") # Read from CONFIG
    if not webhook: log.warning("Slack alert skipped: No webhook configured."); return False
    payload = {"text": message}; return _post_json_to_slack(webhook, payload)

# --- Email Alerting ---
def _get_email_recipients() -> List[str]:
    """Parses the 'to_emails' config value (should be list or None)."""
    to_emails_config = CONFIG.get("to_emails") # Read from CONFIG
    if isinstance(to_emails_config, list): return [str(e).strip() for e in to_emails_config if str(e).strip()]
    log.warning("Email recipients invalid or missing in config."); return []

def _build_email_message(subject: str, body: str, sender: str, recipients: List[str]) -> Optional[EmailMessage]:
    """Creates an EmailMessage object."""
    if not sender or not recipients: log.error("Cannot build email: Sender or recipients missing."); return None
    msg = EmailMessage(); msg["Subject"] = subject; msg["From"] = sender
    msg["To"] = ", ".join(recipients); msg.set_content(body)
    return msg

def send_email(subject: str, body: str) -> bool:
    """Sends an email using SMTP configuration from CONFIG."""
    # Read all required settings from CONFIG
    host = CONFIG.get("smtp_host"); port = CONFIG.get("smtp_port", 587)
    user = CONFIG.get("smtp_user"); password = CONFIG.get("smtp_password")
    sender = CONFIG.get("from_email"); recipients = _get_email_recipients()

    if not all([host, user, password, sender]): log.warning("Email alert skipped: SMTP config incomplete."); return False
    if not recipients: log.warning("Email alert skipped: No valid recipients."); return False
    if not isinstance(port, int) or port <= 0: log.warning(f"Invalid SMTP port: {port}. Using 587."); port = 587

    msg = _build_email_message(subject, body, sender, recipients)
    if msg is None: return False

    log.debug(f"Attempting email via SMTP: {host}:{port}")
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT) as server:
            server.ehlo(); server.starttls(context=context); server.ehlo()
            server.login(user, password); server.send_message(msg)
            log.info(f"Email sent successfully to: {', '.join(recipients)}")
            return True
    except smtplib.SMTPAuthenticationError as e: log.error(f"Email failed: SMTP Auth Error - check user/pass. ({e.smtp_code}: {e.smtp_error})"); return False
    except smtplib.SMTPException as e: log.error(f"Email failed: SMTP Error - {e}"); return False
    except ConnectionRefusedError: log.error(f"Email failed: Connection refused by {host}:{port}."); return False
    except TimeoutError: log.error(f"Email failed: Connection to {host}:{port} timed out."); return False
    except Exception as e: log.error(f"Email failed unexpectedly: {e}", exc_info=True); return False

# --- Composite Dispatch Function ---
def dispatch(subject: str, body: str) -> None:
    """Attempts to send alerts via both Slack and Email if configured."""
    slack_success, email_success = False, False
    if CONFIG.get("slack_webhook"):
        log.info("Dispatching alert via Slack...")
        slack_message = f"*{subject}*\n\n{body}"
        if send_slack(slack_message): slack_success = True
        else: log.warning("Dispatch: Slack alert failed.")
    else: log.debug("Dispatch: Slack skipped (not configured).")

    if CONFIG.get("smtp_host") and CONFIG.get("to_emails"): # Check host and recipients list
        log.info("Dispatching alert via Email...")
        if send_email(subject, body): email_success = True
        else: log.warning("Dispatch: Email alert failed.")
    else: log.debug("Dispatch: Email skipped (not configured).")

    if not slack_success and not email_success: log.error("Dispatch: All channels failed/unconfigured.")
    elif slack_success and email_success: log.info("Dispatch: Alerts sent via Slack and Email.")
    elif slack_success: log.info("Dispatch: Alert sent via Slack only.")
    elif email_success: log.info("Dispatch: Alert sent via Email only.")

__all__ = ["send_slack", "send_email", "dispatch"]