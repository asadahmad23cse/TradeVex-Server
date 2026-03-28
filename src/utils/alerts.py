"""
Layer 10 — Alerts.

  1. Desktop notification via plyer (Windows / macOS / Linux)
  2. Sound alert via winsound (Windows) or beep fallback
  3. Colored terminal output via colorama

  BUY  → green  + high beep
  SELL → red    + low beep
  HOLD → yellow (no sound)
"""

import logging
import platform
import sys

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)
logger = logging.getLogger(__name__)

# Try importing optional deps gracefully
try:
    from plyer import notification as _plyer_notif
    _PLYER_OK = True
except ImportError:
    _PLYER_OK = False
    logger.warning("plyer not installed — desktop notifications disabled")

try:
    import winsound
    _WINSOUND_OK = True
except ImportError:
    _WINSOUND_OK = False


# ------------------------------------------------------------------
# Colour helpers
# ------------------------------------------------------------------

def _colour(text: str, signal: str) -> str:
    if signal == "BUY":
        return Fore.GREEN + Style.BRIGHT + text + Style.RESET_ALL
    elif signal == "SELL":
        return Fore.RED + Style.BRIGHT + text + Style.RESET_ALL
    else:
        return Fore.YELLOW + text + Style.RESET_ALL


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def print_signal(signal, verbose: bool = True) -> None:
    """Print a TradingSignal to the terminal with colour."""
    from src.signals.engine import SignalEngine
    raw = SignalEngine().format_cli(signal)
    coloured = _colour(raw, signal.signal)
    print(coloured)


def print_status(message: str, level: str = "INFO") -> None:
    """Print a status line with colour coding."""
    colour = {
        "INFO": Fore.CYAN,
        "WARN": Fore.YELLOW,
        "ERROR": Fore.RED,
        "OK": Fore.GREEN,
    }.get(level.upper(), Fore.WHITE)
    print(colour + f"[{level}] " + message + Style.RESET_ALL)


def desktop_alert(signal) -> None:
    """Fire a desktop notification for STRONG signals."""
    if not _PLYER_OK:
        return
    sl_pct = (signal.stop_loss - signal.entry_price) / max(signal.entry_price, 1e-9) * 100
    tp_pct = (signal.take_profit - signal.entry_price) / max(signal.entry_price, 1e-9) * 100
    try:
        _plyer_notif.notify(
            title=f"🚨 {signal.signal} Signal — {signal.asset} [{signal.strength}]",
            message=(
                f"Confidence: {signal.confidence:.0f}/100\n"
                f"Entry: {signal.entry_price:.4f}\n"
                f"SL: {signal.stop_loss:.4f} ({sl_pct:+.2f}%)\n"
                f"TP: {signal.take_profit:.4f} ({tp_pct:+.2f}%)\n"
                f"Regime: {signal.regime}"
            ),
            app_name="QuantTrader",
            timeout=10,
        )
    except Exception as exc:
        logger.debug("Desktop notification error: %s", exc)


def sound_alert(signal_type: str) -> None:
    """
    Play a beep:
      BUY  → 1200 Hz, 200 ms
      SELL →  800 Hz, 200 ms
    Falls back silently on non-Windows or if winsound unavailable.
    """
    if not _WINSOUND_OK:
        return
    freq = 1200 if signal_type == "BUY" else 800
    try:
        winsound.Beep(freq, 200)
    except Exception as exc:
        logger.debug("Sound alert error: %s", exc)


def fire_all_alerts(signal) -> None:
    """
    Fire all alerts for a STRONG signal:
      desktop notification + sound + terminal print.
    For MODERATE/WEAK: terminal print only.
    """
    print_signal(signal)
    if signal.strength == "STRONG":
        desktop_alert(signal)
        sound_alert(signal.signal)
