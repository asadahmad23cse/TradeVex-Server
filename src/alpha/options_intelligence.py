"""
OptionsIntelligence — computes F15/F16/F17 from NSE option chain.

F15: IV Skew    — put/call IV differential (z-scored)
F16: Max Pain   — strike minimising writer pain (z-scored)
F17: GEX        — dealer gamma exposure (z-scored)

Outputs in approximately [-1, +1]. Methods return 0.0 on failure — never raise.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger("alpha.options_intelligence")

_RISK_FREE_RATE = 0.065
_DEFAULT_IV = 0.20
_MAX_GEX_STRIKES = 50
_NEUTRAL_BAND = 0.15


def _z_normalize(
    value: float,
    history: deque,
    window: int = 21,
) -> float:
    try:
        history.append(float(value))
        if len(history) < 5:
            return 0.0
        arr = np.array(list(history)[-window:], dtype=float)
        mu, sigma = float(arr.mean()), float(arr.std())
        if sigma < 1e-9:
            return 0.0
        z = (float(value) - mu) / sigma
        z = float(np.clip(z, -1.0, 1.0))
        if abs(z) < _NEUTRAL_BAND:
            return 0.0
        return z
    except Exception:
        return 0.0


class OptionsIntelligence:
    """Per-ticker rolling z-scores for options-derived factors."""

    def __init__(self) -> None:
        self._iv_skew_hist = deque(maxlen=63)
        self._max_pain_hist = deque(maxlen=63)
        self._gex_hist = deque(maxlen=63)
        self.last_max_pain_level: Optional[float] = None

    def compute_iv_skew(self, chain: Optional[dict], underlying: float) -> float:
        try:
            if chain is None or underlying <= 0:
                return 0.0

            strikes = chain.get("strikes") or {}
            lo = underlying * 0.92
            hi = underlying * 0.98

            put_ivs = [
                float(d["PE"]["iv"])
                for strike, d in strikes.items()
                if lo <= float(strike) <= hi
                and "PE" in d
                and float(d["PE"].get("iv", 0) or 0) > 0.5
            ]

            lo2 = underlying * 1.02
            hi2 = underlying * 1.08
            call_ivs = [
                float(d["CE"]["iv"])
                for strike, d in strikes.items()
                if lo2 <= float(strike) <= hi2
                and "CE" in d
                and float(d["CE"].get("iv", 0) or 0) > 0.5
            ]

            if len(put_ivs) < 3 or len(call_ivs) < 3:
                return 0.0

            raw_skew = float(np.mean(put_ivs) - np.mean(call_ivs))
            return _z_normalize(raw_skew, self._iv_skew_hist)
        except Exception as exc:
            logger.warning("IV skew computation failed: %s", exc)
            return 0.0

    def compute_max_pain(self, chain: Optional[dict], underlying: float) -> float:
        try:
            if chain is None or underlying <= 0:
                return 0.0

            strikes = chain.get("strikes") or {}
            strike_list = sorted(strikes.keys())
            if len(strike_list) < 5:
                return 0.0

            ce_oi = {
                int(s): int((strikes[s].get("CE") or {}).get("oi") or 0)
                for s in strike_list
            }
            pe_oi = {
                int(s): int((strikes[s].get("PE") or {}).get("oi") or 0)
                for s in strike_list
            }

            min_pain = float("inf")
            pain_strike = strike_list[0]

            for candidate in strike_list:
                pain = 0.0
                c = float(candidate)
                for s in strike_list:
                    sf = float(s)
                    pain += ce_oi[int(s)] * max(c - sf, 0.0)
                    pain += pe_oi[int(s)] * max(sf - c, 0.0)
                if pain < min_pain:
                    min_pain = pain
                    pain_strike = candidate

            self.last_max_pain_level = float(pain_strike)

            raw_signal = (underlying - float(pain_strike)) / underlying
            clipped = float(np.clip(raw_signal * 10.0, -1.0, 1.0))
            return _z_normalize(clipped, self._max_pain_hist)
        except Exception as exc:
            logger.warning("Max pain computation failed: %s", exc)
            return 0.0

    def compute_gex(
        self,
        chain: Optional[dict],
        underlying: float,
        days_to_expiry: int = 7,
    ) -> float:
        try:
            if chain is None or underlying <= 0:
                return 0.0

            T = max(int(days_to_expiry), 1) / 365.0
            if T < 1e-6:
                return 0.0

            strikes = chain.get("strikes") or {}
            all_s = sorted(strikes.keys())
            if not all_s:
                return 0.0

            mid_idx = min(
                range(len(all_s)),
                key=lambda i: abs(float(all_s[i]) - underlying),
            )
            half = _MAX_GEX_STRIKES // 2
            lo_i = max(0, mid_idx - half)
            hi_i = min(len(all_s), mid_idx + half + 1)
            selected = all_s[lo_i:hi_i]

            total_gex = 0.0
            sqrt_T = math.sqrt(T)
            r = _RISK_FREE_RATE
            s = float(underlying)

            for K in selected:
                row = strikes[K]
                kf = float(K)

                for opt_type, oi_sign in (("CE", 1), ("PE", -1)):
                    opt = row.get(opt_type) or {}
                    oi = int(opt.get("oi") or 0)
                    if oi == 0:
                        continue

                    iv = float(opt.get("iv", 0.0) or 0.0) / 100.0
                    if iv < 0.01:
                        iv = _DEFAULT_IV

                    try:
                        d1 = (
                            math.log(s / kf)
                            + (r + 0.5 * iv * iv) * T
                        ) / (iv * sqrt_T)
                        gamma = float(norm.pdf(d1)) / (s * iv * sqrt_T)
                    except (ValueError, ZeroDivisionError, OverflowError):
                        continue

                    total_gex += oi_sign * oi * gamma

            gex_scaled = total_gex * (s ** 2) * 0.01
            return _z_normalize(gex_scaled, self._gex_hist)
        except Exception as exc:
            logger.warning("GEX computation failed: %s", exc)
            return 0.0

    def get_all_factors(
        self,
        chain: Optional[dict],
        underlying: float,
        days_to_expiry: int = 7,
    ) -> dict:
        ok = chain is not None and underlying > 0
        return {
            "F15_iv_skew": self.compute_iv_skew(chain, underlying),
            "F16_max_pain": self.compute_max_pain(chain, underlying),
            "F17_gex": self.compute_gex(chain, underlying, days_to_expiry),
            "max_pain_level": self.last_max_pain_level,
            "computation_ok": ok,
        }
