"""
Real-time Zerodha WebSocket Pipeline.

Implements the tick -> feature -> signal pipeline using KiteTicker.
Connects to Zerodha, streams live ticks, and triggers the `process_realtime` callback
with minimal latency.
"""

import logging
import threading
import time
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)

class ZerodhaWebSocketClient:
    """
    Production-ready WebSocket client for Zerodha KiteConnect.
    Streams tick data and dispatches to a registered callback queue without delay.
    """

    def __init__(self, api_key: str, access_token: str, instrument_map: Dict[int, str] | None = None):
        """
        :param api_key: Zerodha API Key
        :param access_token: Zerodha Access Token
        :param instrument_map: Dictionary mapping Kite instrument_token to system symbol (e.g. 738561 -> 'RELIANCE')
        """
        self.api_key = api_key
        self.access_token = access_token
        self.instrument_map = instrument_map or {}
        
        self.kws = None
        self._callbacks: List[Callable[[str, float, float, float], None]] = []
        self._connected = False
        self._reconnect_attempts = 0
        self._thread: threading.Thread | None = None

    def register_callback(self, callback: Callable[[str, float, float, float], None]):
        """
        Register a callback function to run on every tick.
        Callback signature: func(symbol: str, last_price: float, volume: float, timestamp: float)
        """
        self._callbacks.append(callback)

    def _on_ticks(self, ws, ticks):
        """
        Triggered on every tick from Kite.
        Passes data to execution queue immediately.
        """
        for tick in ticks:
            instrument_token = tick.get("instrument_token")
            symbol = self.instrument_map.get(instrument_token, str(instrument_token))
            
            last_price = tick.get("last_price", 0.0)
            volume = tick.get("volume_traded", 0.0)
            # Kite tick might have exchange_timestamp
            timestamp = time.time()
            
            # Fire all registered callbacks (tick -> feature -> signal)
            for cb in self._callbacks:
                try:
                    cb(symbol, last_price, volume, timestamp)
                except Exception as e:
                    logger.error(f"Error in tick callback for {symbol}: {e}")

    def _on_connect(self, ws, response):
        """
        Triggered when WebSocket connection is successfully established.
        """
        self._connected = True
        self._reconnect_attempts = 0
        logger.info("Zerodha WebSocket Connected. Subscribing to instruments...")
        
        tokens = list(self.instrument_map.keys())
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            logger.info(f"Subscribed to {len(tokens)} instruments in FULL mode.")
        else:
            logger.warning("No instrument tokens mapping provided. Not subscribing to any ticks.")

    def _on_close(self, ws, code, reason):
        """Triggered when connection is closed."""
        self._connected = False
        logger.warning(f"Zerodha WebSocket Closed: {code} - {reason}")

    def _on_error(self, ws, code, reason):
        """Triggered on WS error."""
        logger.error(f"Zerodha WebSocket Error: {code} - {reason}")

    def _on_reconnect(self, ws, attempts_count):
        """Triggered when auto-reconnecting."""
        self._reconnect_attempts = attempts_count
        logger.info(f"Zerodha WebSocket Reconnecting... Attempt {attempts_count}")

    def _on_noreconnect(self, ws):
        """Triggered when max reconnect attempts reached."""
        logger.critical("Zerodha WebSocket failed to reconnect permanently.")

    def connect(self, threaded: bool = True):
        """
        Initialize and connect KiteTicker. 
        If threaded=True, runs in a background thread so the main program isn't blocked.
        """
        if not self.api_key or not self.access_token:
            logger.error("Cannot start WebSocket: Missing API key or Access token.")
            return

        try:
            from kiteconnect import KiteTicker  # type: ignore[import]
        except ImportError:
            logger.error("Please `pip install kiteconnect` to use Zerodha WebSockets.")
            return

        self.kws = KiteTicker(self.api_key, self.access_token)
        
        self.kws.on_ticks = self._on_ticks  # type: ignore
        self.kws.on_connect = self._on_connect  # type: ignore
        self.kws.on_close = self._on_close  # type: ignore
        self.kws.on_error = self._on_error  # type: ignore
        self.kws.on_reconnect = self._on_reconnect  # type: ignore
        self.kws.on_noreconnect = self._on_noreconnect  # type: ignore

        logger.info("Connecting to Zerodha KiteTicker WebSocket...")
        
        if threaded:
            self._thread = threading.Thread(target=self.kws.connect, kwargs={"threaded": True}, daemon=True)  # type: ignore[attr-stubs, assignment]
            self._thread.start()
        else:
            self.kws.connect(threaded=True)  # type: ignore
            
    def stop(self):
        """Close WebSocket connection cleanly."""
        if self.kws is not None and self._connected:
            self.kws.close()  # type: ignore[attr-stubs, union-attr]
            logger.info("Zerodha WebSocket stopped.")

    def is_connected(self) -> bool:
        return self._connected
