"""
Operational alerting for unattended runs.

Supports Telegram and SMTP email without requiring heavy external SDKs.
"""

from __future__ import annotations

import logging
import smtplib
import time
from datetime import datetime, timezone
from email.message import EmailMessage

import requests

logger = logging.getLogger(__name__)


class NotificationManager:
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.enabled = cfg.get("enabled", False)
        self.telegram = cfg.get("telegram", {})
        self.email = cfg.get("email", {})

    def notify(self, subject: str, message: str, severity: str = "INFO") -> None:
        if not self.enabled:
            return
        decorated_subject = f"[{severity.upper()}] {subject}"
        self._send_telegram(decorated_subject, message)
        self._send_email(decorated_subject, message)

    def _send_telegram(self, subject: str, message: str) -> None:
        bot_token = self.telegram.get("bot_token", "")
        chat_id = self.telegram.get("chat_id", "")
        if not bot_token or not chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": f"{subject}\n{message}"},
                timeout=10,
            ).raise_for_status()
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)

    def _send_email(self, subject: str, message: str) -> None:
        host = self.email.get("smtp_host", "")
        port = int(self.email.get("smtp_port", 587) or 587)
        username = self.email.get("username", "")
        password = self.email.get("password", "")
        recipients = self.email.get("to", [])
        sender = self.email.get("from", username)
        if not host or not sender or not recipients:
            return
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)
            msg.set_content(message)
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                if self.email.get("use_tls", True):
                    smtp.starttls()
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(msg)
        except Exception as exc:
            logger.warning("Email notification failed: %s", exc)

    def push_outgoing_webhook(self, signal: dict, target_url: str) -> bool:
        """
        POST a sanitized TradeVex signal snapshot to an external URL.
        Retries 3 times with 1s / 2s / 4s backoff. Never raises.
        """
        if not target_url or not str(target_url).strip():
            return False

        url = str(target_url).strip()
        payload = {
            "source": "tradevex",
            "ticker": signal.get("asset") or signal.get("ticker"),
            "action": signal.get("direction"),
            "alpha_score": signal.get("alpha_score", signal.get("alpha")),
            "regime": signal.get("regime"),
            "sl": signal.get("sl"),
            "tp": signal.get("tp"),
            "sqs": signal.get("sqs"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for attempt, wait in enumerate([1, 2, 4], start=1):
            try:
                resp = requests.post(url, json=payload, timeout=5)
                if resp.status_code < 300:
                    return True
                logger.warning(
                    "Outgoing webhook attempt %d: HTTP %d",
                    attempt,
                    resp.status_code,
                )
            except Exception as exc:
                logger.warning(
                    "Outgoing webhook attempt %d failed: %s",
                    attempt,
                    exc,
                )
            if attempt < 3:
                time.sleep(wait)

        logger.warning("Outgoing webhook failed after 3 attempts to %s", url)
        return False
