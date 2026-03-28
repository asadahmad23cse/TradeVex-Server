"""NSE F&O options analytics engine for QuantTrader."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests

from src.options.expiry_tracker import ExpiryTracker

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    ts: float
    payload: Any


class OptionsEngine:
    """
    NSE F&O signal engine.
    Covers: Nifty50, BankNifty, FinNifty + top F&O stocks.
    """

    BASE_URL = "https://www.nseindia.com"
    OC_INDICES_URL = BASE_URL + "/api/option-chain-indices"
    OC_EQUITIES_URL = BASE_URL + "/api/option-chain-equities"
    CACHE_TTL_SEC = 60

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.nseindia.com/option-chain",
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._cache: dict[str, _CacheEntry] = {}
        self._iv_history: dict[str, list[float]] = {}
        self._expiry = ExpiryTracker()
        self._warm_session()

    # -------------------------------------------------------------
    # Low-level helpers
    # -------------------------------------------------------------

    def _warm_session(self) -> None:
        try:
            self._session.get(self.BASE_URL, timeout=5)
        except Exception:
            pass

    def _cache_get(self, key: str, ttl: int | None = None) -> Any | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        age = time.time() - entry.ts
        if age <= (ttl if ttl is not None else self.CACHE_TTL_SEC):
            return entry.payload
        return None

    def _cache_set(self, key: str, payload: Any) -> None:
        self._cache[key] = _CacheEntry(ts=time.time(), payload=payload)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = (symbol or "").upper().strip()
        aliases = {
            "NIFTY50": "NIFTY",
            "BANK NIFTY": "BANKNIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "FIN NIFTY": "FINNIFTY",
            "NIFTY FIN SERVICE": "FINNIFTY",
        }
        return aliases.get(s, s)

    @staticmethod
    def _is_index_symbol(symbol: str) -> bool:
        s = symbol.upper()
        return s in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYIT", "NIFTY50"}

    @staticmethod
    def _strike_step(symbol: str, spot: float) -> int:
        sym = symbol.upper()
        if "BANK" in sym:
            return 100
        if "FIN" in sym:
            return 50
        if spot >= 20000:
            return 100
        return 50

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            if v in (None, ""):
                return default
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except Exception:
                continue
        return None

    def _request_chain_payload(self, symbol: str) -> dict:
        s = self._normalize_symbol(symbol)
        cache_key = f"raw_chain::{s}"
        cached = self._cache_get(cache_key, ttl=self.CACHE_TTL_SEC)
        if cached is not None:
            return cached

        params = {"symbol": s}
        urls: list[str] = [self.OC_INDICES_URL] if self._is_index_symbol(s) else [self.OC_EQUITIES_URL, self.OC_INDICES_URL]
        for url in urls:
            try:
                resp = self._session.get(url, params=params, timeout=10)
                if resp.status_code == 401:
                    self._warm_session()
                    resp = self._session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, dict) and payload.get("records"):
                    self._cache_set(cache_key, payload)
                    return payload
            except Exception as exc:
                logger.warning("Option chain fetch failed for %s via %s: %s", s, url, exc)
        return {}

    def _event_within_5_days(self, symbol: str) -> bool:
        # Best effort for stocks; indices usually have macro events not in ticker calendars.
        try:
            if self._is_index_symbol(symbol):
                return False
            import yfinance as yf  # type: ignore[import]

            t = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
            cal = yf.Ticker(t).calendar
            if cal is None:
                return False
            now = datetime.utcnow().date()
            if hasattr(cal, "index") and "Earnings Date" in list(cal.index):
                raw = cal.loc["Earnings Date"].iloc[0]
                if hasattr(raw, "to_pydatetime"):
                    d = raw.to_pydatetime().date()
                else:
                    d = self._parse_date(str(raw))
                if d is None:
                    return False
                return 0 <= (d - now).days <= 5
        except Exception:
            return False
        return False

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------

    def get_option_chain(self, symbol: str, expiry: str = "nearest") -> dict:
        """
        Fetch live option chain from NSE and normalize records.
        """
        s = self._normalize_symbol(symbol)
        key = f"chain::{s}::{expiry or 'nearest'}"
        cached = self._cache_get(key, ttl=self.CACHE_TTL_SEC)
        if cached is not None:
            return cached

        payload = self._request_chain_payload(s)
        if not payload:
            out = {
                "calls": [],
                "puts": [],
                "spot_price": 0.0,
                "expiry_dates": [],
                "pcr": 0.0,
                "max_pain": 0.0,
                "symbol": s,
                "selected_expiry": "",
            }
            self._cache_set(key, out)
            return out

        records = payload.get("records", {}) if isinstance(payload, dict) else {}
        rows = records.get("data", []) if isinstance(records, dict) else []
        expiry_dates = list(records.get("expiryDates", []) or [])
        selected_expiry = ""
        if expiry_dates:
            if expiry and expiry.lower() != "nearest" and expiry in expiry_dates:
                selected_expiry = expiry
            else:
                selected_expiry = expiry_dates[0]

        calls: list[dict] = []
        puts: list[dict] = []
        for item in rows:
            exp = item.get("expiryDate")
            if selected_expiry and exp != selected_expiry:
                continue
            strike = self._safe_float(item.get("strikePrice"))
            ce = item.get("CE") or {}
            pe = item.get("PE") or {}
            if ce:
                calls.append(
                    {
                        "strike": strike,
                        "oi": self._safe_float(ce.get("openInterest")),
                        "oi_change": self._safe_float(ce.get("changeinOpenInterest")),
                        "iv": self._safe_float(ce.get("impliedVolatility")),
                        "ltp": self._safe_float(ce.get("lastPrice")),
                        "volume": self._safe_float(ce.get("totalTradedVolume")),
                        "expiry": exp,
                    }
                )
            if pe:
                puts.append(
                    {
                        "strike": strike,
                        "oi": self._safe_float(pe.get("openInterest")),
                        "oi_change": self._safe_float(pe.get("changeinOpenInterest")),
                        "iv": self._safe_float(pe.get("impliedVolatility")),
                        "ltp": self._safe_float(pe.get("lastPrice")),
                        "volume": self._safe_float(pe.get("totalTradedVolume")),
                        "expiry": exp,
                    }
                )

        calls.sort(key=lambda x: x["strike"])
        puts.sort(key=lambda x: x["strike"])
        spot = self._safe_float(records.get("underlyingValue"))
        total_call_oi = sum(x.get("oi", 0.0) for x in calls)
        total_put_oi = sum(x.get("oi", 0.0) for x in puts)
        pcr = total_put_oi / max(total_call_oi, 1.0)
        out = {
            "calls": calls,
            "puts": puts,
            "spot_price": spot,
            "expiry_dates": expiry_dates,
            "pcr": round(pcr, 4),
            "max_pain": round(self.calculate_max_pain({"calls": calls, "puts": puts}), 2),
            "symbol": s,
            "selected_expiry": selected_expiry,
        }
        self._cache_set(key, out)
        return out

    def calculate_max_pain(self, option_chain: dict) -> float:
        """
        Max pain = strike where total option writers' payout is minimized.
        """
        calls = option_chain.get("calls", []) or []
        puts = option_chain.get("puts", []) or []
        strikes = sorted({self._safe_float(x.get("strike")) for x in calls + puts if self._safe_float(x.get("strike")) > 0})
        if not strikes:
            return 0.0

        min_pain = float("inf")
        max_pain_strike = strikes[0]
        for settle in strikes:
            call_pain = 0.0
            for c in calls:
                k = self._safe_float(c.get("strike"))
                oi = self._safe_float(c.get("oi"))
                call_pain += max(0.0, settle - k) * oi
            put_pain = 0.0
            for p in puts:
                k = self._safe_float(p.get("strike"))
                oi = self._safe_float(p.get("oi"))
                put_pain += max(0.0, k - settle) * oi
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                max_pain_strike = settle
        return float(max_pain_strike)

    def get_iv_surface(self, symbol: str, expiry: str = "nearest") -> dict:
        """
        Build a simplified IV surface and return IV percentile.
        """
        s = self._normalize_symbol(symbol)
        chain = self.get_option_chain(s, expiry=expiry)
        calls = chain.get("calls", []) or []
        puts = chain.get("puts", []) or []
        spot = self._safe_float(chain.get("spot_price"))
        expiries = chain.get("expiry_dates", []) or []

        points: list[dict] = []
        for row in calls:
            points.append({"strike": row.get("strike"), "expiry": row.get("expiry"), "type": "CE", "iv": row.get("iv", 0.0)})
        for row in puts:
            points.append({"strike": row.get("strike"), "expiry": row.get("expiry"), "type": "PE", "iv": row.get("iv", 0.0)})

        # Current ATM IV estimate.
        current_atm_iv = 0.0
        if spot > 0 and calls and puts:
            atm = min(sorted({x.get("strike", 0.0) for x in calls}), key=lambda k: abs(float(k) - spot))
            ce_iv = next((self._safe_float(x.get("iv")) for x in calls if self._safe_float(x.get("strike")) == float(atm)), 0.0)
            pe_iv = next((self._safe_float(x.get("iv")) for x in puts if self._safe_float(x.get("strike")) == float(atm)), 0.0)
            valid = [v for v in (ce_iv, pe_iv) if v > 0]
            current_atm_iv = float(sum(valid) / len(valid)) if valid else 0.0

        hist = self._iv_history.setdefault(s, [])
        if current_atm_iv > 0:
            hist.append(current_atm_iv)
        if len(hist) > 252:
            self._iv_history[s] = hist[-252:]
            hist = self._iv_history[s]

        iv_percentile = 50.0
        if hist:
            sorted_hist = sorted(hist)
            count_le = sum(1 for v in sorted_hist if v <= current_atm_iv)
            iv_percentile = (count_le / max(len(sorted_hist), 1)) * 100.0
        elif points:
            vals = [self._safe_float(p.get("iv")) for p in points if self._safe_float(p.get("iv")) > 0]
            if vals:
                lo, hi = min(vals), max(vals)
                iv_percentile = ((current_atm_iv - lo) / max(hi - lo, 1e-9)) * 100.0

        return {
            "symbol": s,
            "spot_price": spot,
            "expiries": expiries,
            "points": points,
            "atm_iv": round(current_atm_iv, 3),
            "iv_percentile": round(float(max(0.0, min(100.0, iv_percentile))), 2),
            "iv_bias": (
                "SELL_PREMIUM"
                if iv_percentile > 80
                else "BUY_OPTIONS"
                if iv_percentile < 20
                else "NEUTRAL"
            ),
        }

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        expiry_days: int,
        iv: float,
        risk_free_rate: float = 0.065,
        option_type: str = "CE",
    ) -> dict:
        """Black-Scholes Greeks (Delta, Gamma, Theta, Vega)."""
        s = max(float(spot), 1e-9)
        k = max(float(strike), 1e-9)
        t = max(float(expiry_days) / 365.0, 1e-6)
        sigma = max(float(iv) / 100.0, 1e-6)
        r = float(risk_free_rate)

        d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)

        # Standard normal CDF/PDF without scipy dependency.
        n_cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        n_pdf = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

        if option_type.upper() == "PE":
            delta = n_cdf(d1) - 1.0
            theta = (
                -(s * n_pdf(d1) * sigma) / (2 * math.sqrt(t))
                + r * k * math.exp(-r * t) * n_cdf(-d2)
            )
        else:
            delta = n_cdf(d1)
            theta = (
                -(s * n_pdf(d1) * sigma) / (2 * math.sqrt(t))
                - r * k * math.exp(-r * t) * n_cdf(d2)
            )

        gamma = n_pdf(d1) / (s * sigma * math.sqrt(t))
        vega = s * n_pdf(d1) * math.sqrt(t)

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            # Keep theta in daily decay terms for usability.
            "theta": round(theta / 365.0, 4),
            "vega": round(vega / 100.0, 4),
        }

    def get_span_margin(self, strategy: str, symbol: str, lots: int = 1) -> dict:
        """Approximate SPAN margin for common option strategies."""
        s = self._normalize_symbol(symbol)
        chain = self.get_option_chain(s)
        spot = self._safe_float(chain.get("spot_price"))
        lot_map = {"NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 25}
        lot_size = lot_map.get(s, 500)
        units = max(int(lots), 1) * lot_size

        strategy_u = (strategy or "HOLD").upper()
        margin = 0.0
        max_loss = 0.0

        if strategy_u in {"BUY_CE", "BUY_PE"}:
            # Premium-only strategy.
            chain_rows = chain.get("calls", []) if strategy_u == "BUY_CE" else chain.get("puts", [])
            premium = self._safe_float(chain_rows[0].get("ltp")) if chain_rows else 0.0
            margin = premium * units
            max_loss = margin
        elif strategy_u == "SELL_STRADDLE":
            lot_value = spot * units
            margin = 1.5 * lot_value
            max_loss = margin * 1.2
        elif strategy_u == "COVERED_CALL":
            stock_value = spot * units
            margin = stock_value
            max_loss = stock_value
        else:
            margin = spot * units * 0.5
            max_loss = margin

        return {"margin_required": round(margin, 2), "max_loss": round(max_loss, 2), "lot_size": lot_size}

    def get_options_signal(self, symbol: str) -> dict:
        """Generate actionable F&O strategy signal."""
        s = self._normalize_symbol(symbol)
        chain = self.get_option_chain(s, expiry="nearest")
        iv = self.get_iv_surface(s, expiry=chain.get("selected_expiry", "nearest"))
        spot = self._safe_float(chain.get("spot_price"))
        pcr = self._safe_float(chain.get("pcr"))
        max_pain = self._safe_float(chain.get("max_pain"))
        iv_pct = self._safe_float(iv.get("iv_percentile"), 50.0)
        expiry_days = self._expiry.get_days_to_expiry(s)
        selected_expiry = chain.get("selected_expiry") or ""

        step = self._strike_step(s, spot if spot > 0 else max_pain)
        ref_spot = spot if spot > 0 else max_pain
        rec_strike = round(ref_spot / step) * step if ref_spot > 0 else 0.0

        call_row = next((x for x in chain.get("calls", []) if self._safe_float(x.get("strike")) == rec_strike), None)
        put_row = next((x for x in chain.get("puts", []) if self._safe_float(x.get("strike")) == rec_strike), None)
        ce_ltp = self._safe_float(call_row.get("ltp")) if call_row else 0.0
        pe_ltp = self._safe_float(put_row.get("ltp")) if put_row else 0.0
        atm_iv = self._safe_float(iv.get("atm_iv"), 15.0)

        strategy = "HOLD"
        rationale = "No high-probability F&O setup"
        confidence = 50.0
        risk_reward = 1.0
        entry_premium = 0.0
        stop_loss_premium = 0.0
        target_premium = 0.0
        opt_type = "CE"

        if pcr > 1.3 and spot < max_pain and iv_pct < 50:
            strategy = "BUY_CE"
            rationale = "PCR high, spot below max pain, IV moderate: bullish reversal setup"
            confidence = 74.0
            entry_premium = ce_ltp
            stop_loss_premium = entry_premium * 0.7
            target_premium = entry_premium * 1.5
            risk_reward = 1.67
            opt_type = "CE"
        elif pcr < 0.7 and spot > max_pain and iv_pct < 50:
            strategy = "BUY_PE"
            rationale = "PCR low, spot above max pain, IV moderate: bearish reversal setup"
            confidence = 74.0
            entry_premium = pe_ltp
            stop_loss_premium = entry_premium * 0.7
            target_premium = entry_premium * 1.5
            risk_reward = 1.67
            opt_type = "PE"
        elif iv_pct > 80 and expiry_days <= 3:
            strategy = "SELL_STRADDLE"
            rationale = "IV extremely rich near expiry: theta decay premium-selling setup"
            confidence = 78.0
            entry_premium = ce_ltp + pe_ltp
            stop_loss_premium = entry_premium * 1.35
            target_premium = entry_premium * 0.6
            risk_reward = 1.14
            opt_type = "CE"
        elif iv_pct < 20 and self._event_within_5_days(s):
            strategy = "BUY_STRADDLE"
            rationale = "IV very cheap with event risk: volatility expansion setup"
            confidence = 72.0
            entry_premium = ce_ltp + pe_ltp
            stop_loss_premium = entry_premium * 0.6
            target_premium = entry_premium * 1.6
            risk_reward = 1.50
            opt_type = "CE"

        greeks = self.calculate_greeks(
            spot=spot if spot > 0 else rec_strike,
            strike=rec_strike if rec_strike > 0 else spot,
            expiry_days=max(expiry_days, 1),
            iv=atm_iv if atm_iv > 0 else 15.0,
            option_type=opt_type,
        )

        return {
            "symbol": s,
            "strategy": strategy,
            "rationale": rationale,
            "max_pain": round(max_pain, 2),
            "spot_price": round(spot, 2),
            "pcr": round(pcr, 4),
            "iv_percentile": round(iv_pct, 2),
            "recommended_strike": round(rec_strike, 2),
            "recommended_expiry": selected_expiry,
            "days_to_expiry": expiry_days,
            "entry_premium": round(entry_premium, 2),
            "stop_loss_premium": round(stop_loss_premium, 2),
            "target_premium": round(target_premium, 2),
            "greeks": greeks,
            "confidence": round(confidence, 2),
            "risk_reward": round(risk_reward, 2),
            "margin": self.get_span_margin(strategy, s, lots=1),
        }
