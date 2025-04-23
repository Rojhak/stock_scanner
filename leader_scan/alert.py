# leader_scan/alert.py
"""
Alerting utilities for sending notifications via Slack and Email.

Reads configuration from the central `CONFIG` object.
Handles potential errors during sending gracefully.
"""

from __future__ import annotations

import json
import smtplib
import ssl
import urllib.request
import urllib.error
import logging
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
    """
    Sends a JSON payload to a Slack webhook URL using urllib.

    Args:
        url: The Slack Incoming Webhook URL.
        payload: The dictionary to send as JSON (e.g., {"text": "message"}).

    Returns:
        True if the request was successful (HTTP 2xx), False otherwise.
    """
    if not url:
        log.error("Slack post failed: Webhook URL is empty.")
        return False

    try:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")

        log.debug(f"Sending Slack POST request to {url} with payload: {payload}")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            status_code = response.getcode()
            log.debug(f"Slack response status code: {status_code}")
            # Slack typically returns 200 OK on success
            if 200 <= status_code < 300:
                log.info("Slack message sent successfully.")
                return True
            else:
                log.error(f"Slack post failed with status code: {status_code}. Response: {response.read().decode('utf-8')}")
                return False
    except urllib.error.HTTPError as e:
        # More specific error handling for HTTP errors
        log.error(f"Slack post failed (HTTPError): {e.code} - {e.reason}. Response: {e.read().decode('utf-8', errors='ignore')}")
        return False
    except urllib.error.URLError as e:
        # Handle URL errors (e.g., network issues, invalid URL)
        log.error(f"Slack post failed (URLError): {e.reason}")
        return False
    except Exception as e:
        # Catch any other unexpected exceptions
        log.error(f"Slack post failed unexpectedly: {e}", exc_info=True)
        return False

def send_slack(message: str) -> bool:
    """
    Sends a message to the configured Slack webhook.

    Args:
        message: The text message to send (supports Slack markdown).

    Returns:
        True if successful, False otherwise.
    """
    webhook = CONFIG.get("slack_webhook")
    if not webhook:
        log.warning("Slack alert skipped: Webhook URL not configured (CONFIG['slack_webhook']).")
        return False

    # Basic Slack payload structure
    payload = {"text": message}
    return _post_json_to_slack(webhook, payload)

# --- Email Alerting ---

def _get_email_recipients() -> List[str]:
    """Parses the 'to_emails' config value into a list of strings."""
    to_emails_config = CONFIG.get("to_emails")
    recipients = []
    if isinstance(to_emails_config, str):
        recipients = [email.strip() for email in to_emails_config.split(',') if email.strip()]
    elif isinstance(to_emails_config, list):
        recipients = [str(email).strip() for email in to_emails_config if str(email).strip()]

    if not recipients:
         log.warning("Email recipients list is empty or not configured properly.")
    return recipients

def _build_email_message(subject: str, body: str, sender: str, recipients: List[str]) -> Optional[EmailMessage]:
    """Creates an EmailMessage object."""
    if not sender or not recipients:
        log.error("Cannot build email: Sender or recipients missing.")
        return None

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    # Join recipients with comma for the 'To' header
    msg["To"] = ", ".join(recipients)
    # Set content - assuming plain text body for now
    msg.set_content(body)
    # Could add HTML content using msg.add_alternative(html_body, subtype='html')
    return msg

def send_email(subject: str, body: str) -> bool:
    """
    Sends an email using SMTP configuration from CONFIG.
    Uses STARTTLS for secure connection.

    Args:
        subject: The email subject line.
        body: The plain text body of the email.

    Returns:
        True if successful, False otherwise.
    """
    # Retrieve SMTP configuration safely
    host = CONFIG.get("smtp_host")
    port = CONFIG.get("smtp_port", 587) # Default port for TLS
    user = CONFIG.get("smtp_user")
    password = CONFIG.get("smtp_password")
    sender = CONFIG.get("from_email")
    recipients = _get_email_recipients()

    # Validate configuration
    if not all([host, user, password, sender]):
        log.warning("Email alert skipped: SMTP configuration incomplete (host, user, password, from_email).")
        return False
    if not recipients:
        log.warning("Email alert skipped: No valid recipients configured (to_emails).")
        return False
    if not isinstance(port, int) or port <= 0:
         log.warning(f"Invalid SMTP port configured: {port}. Using default 587.")
         port = 587

    # Build the email message
    msg = _build_email_message(subject, body, sender, recipients)
    if msg is None:
        return False # Error already logged

    # Send the email
    log.debug(f"Attempting to send email via SMTP: {host}:{port}")
    try:
        # Create a secure SSL context
        context = ssl.create_default_context()
        # Connect using SMTP_SSL for implicit SSL/TLS or SMTP for STARTTLS
        # Using STARTTLS is generally preferred on port 587
        with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT) as server:
            server.ehlo() # Identify server capabilities
            server.starttls(context=context) # Secure the connection
            server.ehlo() # Re-identify capabilities after TLS
            server.login(user, password)
            server.send_message(msg) # Sends to all recipients in 'To' header
            log.info(f"Email sent successfully to: {', '.join(recipients)}")
            return True
    except smtplib.SMTPAuthenticationError as e:
        log.error(f"Email failed: SMTP Authentication Error - Check username/password. ({e.smtp_code}: {e.smtp_error})")
        return False
    except smtplib.SMTPException as e:
        # Catch other SMTP related errors (connection, sending, etc.)
        log.error(f"Email failed: SMTP Error - {e}")
        return False
    except ConnectionRefusedError:
         log.error(f"Email failed: Connection refused by SMTP server {host}:{port}.")
         return False
    except TimeoutError:
        log.error(f"Email failed: Connection to SMTP server {host}:{port} timed out.")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        log.error(f"Email failed unexpectedly: {e}", exc_info=True)
        return False

# --- Composite Dispatch Function ---

def dispatch(subject: str, body: str) -> None:
    """
    Attempts to send alerts via both Slack and Email, if configured.
    Logs warnings or errors if a channel fails but continues.

    Args:
        subject: The subject line (used for email, prepended to Slack message).
        body: The main content of the alert.
    """
    slack_success = False
    email_success = False

    # --- Attempt Slack ---
    if CONFIG.get("slack_webhook"):
        log.info("Dispatching alert via Slack...")
        # Format Slack message with subject
        slack_message = f"*{subject}*\n\n{body}"
        if not send_slack(slack_message):
            log.warning("Dispatch: Slack alert failed.")
        else:
             slack_success = True
    else:
        log.debug("Dispatch: Slack skipped (not configured).")

    # --- Attempt Email ---
    if CONFIG.get("smtp_host") and CONFIG.get("to_emails"):
        log.info("Dispatching alert via Email...")
        if not send_email(subject, body):
            log.warning("Dispatch: Email alert failed.")
        else:
            email_success = True
    else:
        log.debug("Dispatch: Email skipped (not configured).")

    if not slack_success and not email_success:
         log.error("Dispatch: All alert channels failed or were not configured.")
    elif slack_success and email_success:
         log.info("Dispatch: Alerts sent successfully via Slack and Email.")
    elif slack_success:
         log.info("Dispatch: Alert sent successfully via Slack only.")
    elif email_success:
         log.info("Dispatch: Alert sent successfully via Email only.")


# --- Explicit Export List ---
__all__ = [
    "send_slack",
    "send_email",
    "dispatch",
]

# --- Example Usage (if run directly for testing) ---
if __name__ == '__main__':
    print("Testing alert functions (requires configuration in leader_scan/config.py or environment variables)...")

    # Example: Load config overrides from environment for testing
    # os.environ['LS_SLACK_WEBHOOK'] = 'YOUR_TEST_WEBHOOK_URL'
    # os.environ['LS_SMTP_HOST'] = 'smtp.example.com'
    # ... set other SMTP env vars ...

    test_subject = "Leader Scan Test Alert"
    test_body = "This is a test message from alert.py.\nSymbol | Score\n-------|------\nTEST   | 4.5"

    print("\n--- Testing Slack ---")
    try:
        if send_slack(f"*{test_subject}*\n{test_body}"):
            print("Test Slack message sent (check your channel).")
        else:
            print("Test Slack message failed (check logs and config).")
    except Exception as e:
        print(f"Error testing Slack: {e}")

    print("\n--- Testing Email ---")
    try:
        if send_email(test_subject, test_body):
            print("Test Email message sent (check recipient inbox).")
        else:
            print("Test Email message failed (check logs and config).")
    except Exception as e:
        print(f"Error testing Email: {e}")

    print("\n--- Testing Dispatch ---")
    dispatch(test_subject, "This is a test dispatch message.\nIt should attempt both Slack and Email.")
    print("Dispatch function called (check logs for success/failure).")