# Quant Trading Signal System — Design Spec
**Date:** 2026-03-25
**Status:** Approved (v2 — post spec-review fixes)
**Scope:** Full institutional-grade quant pipeline producing real-time actionable trading signals for demo account practice across Indian stocks, US stocks, and Forex — intraday and swing timeframes.

---

## 1. Goals

- Generate real-time BUY/SELL/HOLD signals with entry price, ATR-based stop-loss, and take-profit levels
- Apply the same mathematics and algorithms used at institutional quant funds (AQR, Two Sigma, Citadel level)
- Cover all three asset classes: Indian stocks (NSE/BSE), US stocks (NYSE/NASDAQ), Forex pairs
- Support both intraday (5m/15m/1h) and swing (daily) timeframes simultaneously
- Deliver signals via live web dashboard, colored CLI output, and desktop/sound alerts
- Respect Alpha Vantage free tier (5 req/min, 500 req/day) — used for Forex EOD only; yfinance handles all intraday
- Log all signals to SQLite with outcome tracking for continuous strategy improvement

---

## 2. Architecture — 10-Layer Quant Pipeline

```
Layer 1:  Data Ingestion         →  Multi-source fetcher with fallback chain
Layer 2:  Feature Engineering    →  30+ technical + microstructure indicators
Layer 3:  5-Factor Alpha Model   →  IC-weighted multi-factor scoring
Layer 4:  Regime Detection       →  Hidden Markov Model (3 states, 4 emissions)
Layer 5:  Signal Engine          →  Final BUY/SELL/HOLD + full metadata
Layer 6:  Risk Manager           →  Empirical Kelly + VaR + circuit breakers
Layer 7:  Portfolio Tracker      →  Live PnL + Sharpe/Sortino/Calmar + costs
Layer 8:  Live Scheduler         →  Intraday (5m) + EOD swing loops
Layer 9:  Signal Store           →  SQLite with indexes + outcome tracking
Layer 10: Output Layer           →  FastAPI dashboard + WebSocket + CLI + alerts
```

---

## 3. Layer 1 — Data Ingestion

### Sources and Roles

| Source | Assets | Role |
|--------|--------|------|
| **yfinance** | Indian stocks (.NS/.BO), US stocks intraday + historical | Primary for ALL intraday + Indian data |
| **Alpha Vantage** | Forex EOD (5 pairs) + supplemental US daily EOD | Secondary — reserved for Forex only |
| **NSE Scraper** (existing) | Indian live quotes | Live supplement; has fallback chain |

### Alpha Vantage Budget (corrected)
```
Each TIME_SERIES_INTRADAY call = 1 request per asset (returns ALL candles)
6 US stocks × 78 five-min intervals = 468 requests/day for US intraday alone
→ THIS BLOWS THE 500/day budget. DO NOT USE AV for intraday.

Corrected allocation:
  Forex EOD (5 pairs × 1/day):            5 requests/day
  US stock daily EOD (6 stocks × 1/day):  6 requests/day
  Buffer:                                489 requests/day
  Total:                                  11/day  ✓  (well under 500)

All intraday data (US + India): yfinance (free, no hard cap)
```

### NSE Scraper Fallback Chain
The NSE scraper uses undocumented internal endpoints subject to bot-blocking. The system must follow this fallback chain on any non-200 response or session expiry:
```
1. NSE Scraper (primary, session refresh on 401/403 with 1 retry)
2. yfinance (.NS ticker, last 1-day 5-min interval)
3. Stale cache (serve last known value; log WARNING if >10 min stale)
4. Suppress signal for this asset this cycle (log ERROR)
```

### Rate Limit Management
- **Token bucket** rate limiter for Alpha Vantage: max 5 tokens/min, 500/day hard stop
- **In-memory TTL cache**: intraday data 4 min, daily data 60 min, Forex 4 min
- **Priority queue**: high-volatility / near-threshold assets polled first

### Asset Universe (default `config.yaml`)
```yaml
indian_stocks:  [RELIANCE, INFY, TCS, HDFCBANK, ICICIBANK]
us_stocks:      [AAPL, TSLA, NVDA, MSFT, GOOGL, SPY]
forex:          [EURUSD, GBPUSD, USDINR, USDJPY, XAUUSD]
benchmarks:     [^NSEI, SPY, DXY]   # index benchmarks — NOT traded, used for beta/regime
```
Note: `^NSEI` (NIFTY 50 index) is a benchmark only, not a tradeable signal target.

---

## 4. Layer 2 — Feature Engineering (30+ indicators)

### Existing (kept from current codebase)
- SMA (20, 50), EMA (20), RSI (14), MACD + Signal Line
- Bollinger Bands (upper/mid/lower/std), ATR (14)
- Volatility (20-day annualized), Returns, Alpha factors (MeanReversion, Momentum, VolumeIntensity), Lag features

### New: Trend / Momentum
- **Stochastic Oscillator** (%K, %D) — overbought/oversold + trend confirmation
- **Williams %R** — momentum oscillator, divergence detection
- **CCI** (Commodity Channel Index) — cycle-based trend
- **ROC** (Rate of Change) — multi-period: 1d, 5d, 10d, 20d, 60d
- **Ichimoku Cloud** (Tenkan, Kijun, Senkou A/B, Chikou) — full trend structure

### New: Volume / Microstructure
- **OBV** (On-Balance Volume) + OBV slope (5d linear regression slope)
- **VWAP** — Volume-Weighted Average Price (intraday only; reset each session)
- **Volume Oscillator** — short (5d) vs long (20d) volume MA ratio
- **Chaikin Money Flow (CMF)** — buying/selling pressure over 20 periods

### New: Volatility / Risk
- **Keltner Channels** — ATR-based bands (Bollinger + Keltner squeeze detection)
- **Historical Volatility Cone** — percentile rank of current ATR vs 90-day ATR history
- **Ulcer Index** — downside risk measure over 14 periods

### New: Quant-Specific
- **Hurst Exponent** (H): R/S analysis on rolling 60-day daily data — computed in EOD loop, cached for full trading session (not recomputed every 5 min; too expensive)
  - H < 0.45: mean-reverting → weight mean-reversion factor higher
  - H > 0.55: trending → weight momentum factor higher
  - 0.45 ≤ H ≤ 0.55: random walk → halve position size
- **Z-score of Close vs SMA_20** (cross-sectional normalization for mean-reversion factor)
- **Rolling Beta** (60d) vs benchmark: `^NSEI` for Indian, SPY for US, DXY for Forex
- **Volume Ratio**: current volume / 20d avg volume (used as HMM emission feature)

---

## 5. Layer 3 — 5-Factor Alpha Model

### Sign Convention (important for IC-weighting)
Each factor enters the IC-weighted combination as a pre-signed Z-score, meaning the sign already encodes bullish (+) or bearish (−) direction. The IC for each factor is computed as `pearson_corr(factor_score, forward_return)` — when IC is negative, the factor has been inversely predictive and the formula correctly inverts its contribution. No double-inversion occurs because factors are pre-signed.

### The 5 Factors

**Factor 1 — Momentum** (pre-signed: + = bullish momentum)
```
Raw = 0.4×ROC_5d + 0.3×ROC_20d + 0.2×ROC_60d + 0.1×ROC_1d
F1 = Z_score(Raw, cross_sectional_within_asset_class)
```
Based on Jegadeesh & Titman (1993) momentum anomaly. Hurst > 0.55 multiplies this factor by 1.25.

**Factor 2 — Mean Reversion** (pre-signed: − = bullish when price below mean)
```
F2 = −Z_score(Close − SMA_20, rolling_std=BB_Std)
```
Price far above mean → F2 negative (expect reversion down). Pre-signed, so no double-inversion. Hurst < 0.45 multiplies by 1.25.

**Factor 3 — Volume Alpha** (pre-signed: + = bullish accumulation)
```
F3 = Z_score(0.5×OBV_slope_5d + 0.3×CMF_20 + 0.2×VolumeOscillator)
```
Rising OBV + rising price = institutional accumulation. Divergence = distribution.

**Factor 4 — ML Alpha** (pre-signed: + = bullish LSTM prediction)
```
# MC Dropout: run model(X, training=True) × 20 iterations
predictions = [model(X, training=True) for _ in range(20)]
mean_pred = mean(predictions)
uncertainty = std(predictions)
confidence_weight = 1 / (1 + uncertainty)   # higher uncertainty = lower weight

F4 = sign(mean_pred − current_price) × confidence_weight
```
⚠️ **Implementation note**: `trend_model.py` `predict()` must be replaced. Keras `model.predict()` disables dropout at inference. Use `model(X_tensor, training=True)` called 20 times in a loop.

**Factor 5 — Volatility / Squeeze** (pre-signed: direction-neutral; magnitude = breakout readiness)
```
# Keltner Squeeze: BB inside Keltner = compression = breakout imminent
squeeze_active = (BB_Upper < Keltner_Upper) and (BB_Lower > Keltner_Lower)
vol_percentile = percentile_rank(ATR_14, window=90)

# When squeeze fires, use momentum direction to sign it
F5 = (−1 × vol_percentile) if not squeeze_active
     else sign(F1) × (1 − vol_percentile)   # squeeze + momentum direction
```
Low volatility = higher confidence baseline. Keltner Squeeze fires breakout signal aligned with momentum direction.

### Factor Combination (IC-Weighted)

```python
# IC: rolling 60-day Pearson correlation of each factor score vs next-day return
IC_i = rolling_pearson_corr(factor_i_t, return_t+1, window=60)

# IC-weighted combination — with divide-by-zero protection
denominator = max(sum(abs(IC_i) for i in range(5)), 0.1)  # floor at 0.1
alpha_score = sum(IC_i * Z_score_i for i in range(5)) / denominator

# Map alpha_score to confidence (0–100)
confidence = 50 + 50 * tanh(alpha_score)   # maps (-inf, +inf) → (0, 100)

# Signal thresholds
if alpha_score > 0.30:   signal = BUY
elif alpha_score < -0.30: signal = SELL
else:                     signal = HOLD

# Strength from confidence
if confidence > 75:   strength = STRONG
elif confidence > 50: strength = MODERATE
else:                 strength = WEAK
```

### WFO Validation of Alpha Model Parameters
The following parameters are validated via Walk-Forward Optimization before going live:
- `alpha_score threshold` (default 0.30): tested range [0.20, 0.45] in 0.05 steps
- `IC_window` (default 60d): tested range [30, 90] in 15-day steps
- `kelly_fraction` (default 0.25): fixed (not tuned — set conservatively by design)

WFO configuration: expanding window, minimum 252-day train, 21-day out-of-sample test step.
Overfitting check: out-of-sample Information Ratio (IR = IC / std(IC)) must be > 0.3. Parameters that produce IR < 0.3 out-of-sample are rejected regardless of in-sample performance.

---

## 6. Layer 4 — Regime Detection (HMM)

### Model Specification
- **Type**: Gaussian HMM, 3 hidden states (Bull, Bear, Sideways)
- **Library**: `hmmlearn.hmm.GaussianHMM`
- **covariance_type**: `'full'` — returns and volatility are correlated by construction; diagonal covariance is incorrect here
- **Training**: rolling 252-day window of daily data, retrained weekly (Sundays)
- **Inference**: Viterbi decoding → most likely current state + state probabilities

### Emission Features (4 features, up from 2)
```
Feature 1: Daily log-return                    (captures drift direction)
Feature 2: Realized volatility (20d rolling)   (captures vol regime)
Feature 3: Volume ratio (volume / 20d avg)      (high vol = distribution or panic)
Feature 4: ATR percentile rank (90d window)     (cross-asset vol percentile)
```
Using 4 features significantly improves Bull/Sideways discrimination (both have low volatility but differ in volume and ATR percentile).

### State Definitions
| State | Return profile | Vol | Volume | Signal filter |
|-------|---------------|-----|--------|---------------|
| BULL | Positive drift | Low-medium | Normal | BUY signals pass; SELL suppressed |
| BEAR | Negative drift | High | High (panic) | SELL signals pass; BUY suppressed |
| SIDEWAYS | Near-zero drift | Low | Low-normal | Only STRONG signals pass; position size halved |

### State Label Assignment
HMM states are unlabeled after training. Assignment: rank states by mean return — highest = BULL, lowest = BEAR, middle = SIDEWAYS.

---

## 7. Layer 5 — Signal Engine

### Signal Object
```python
@dataclass
class TradingSignal:
    signal_id: str            # UUID4
    timestamp: datetime
    asset: str
    asset_class: str          # 'indian_stock' | 'us_stock' | 'forex'
    timeframe: str            # 'intraday' | 'swing'
    signal: str               # 'BUY' | 'SELL' | 'HOLD'
    strength: str             # 'STRONG' | 'MODERATE' | 'WEAK'
    confidence: float         # 0-100 via tanh mapping of alpha_score
    alpha_score: float        # raw IC-weighted factor score
    regime: str               # 'BULL' | 'BEAR' | 'SIDEWAYS'
    hurst_exponent: float
    entry_price: float
    stop_loss: float          # ATR-based (see formula below)
    take_profit: float        # R:R based (see formula below)
    risk_pct: float           # % portfolio at risk (Kelly-derived)
    kelly_fraction: float     # raw Kelly output before capping
    position_size_pct: float  # final capped position size
    factor_scores: dict       # {F1: score, F2: score, F3: score, F4: score, F5: score}
    slippage_cost_pct: float  # estimated transaction cost (see Layer 7)
```

### Stop Loss Calculation
```
For BUY:  SL = entry_price − (ATR_multiplier × ATR_14)
For SELL: SL = entry_price + (ATR_multiplier × ATR_14)

ATR_multiplier by strength:
  STRONG   → 1.5×  (tight stop; conviction is high, position size is larger)
  MODERATE → 2.0×
  WEAK     → 2.5×  (wider stop; lower conviction, smaller size)

Rationale: Tighter stop on STRONG signals is the institutional convention for conviction
trades. The larger position size from Kelly (driven by higher confidence/win rate) means
total dollar risk stays controlled: Risk$ = position_size × distance_to_stop.
A STRONG signal with 1.5× ATR stop and larger Kelly fraction risks the same dollar
amount as a WEAK signal with 2.5× ATR and a smaller fraction.
```

### Take Profit Calculation
```
R:R ratio: STRONG = 3.0, MODERATE = 2.5, WEAK = 2.0

For BUY:  TP = entry_price + (|entry_price − SL| × R:R)
For SELL: TP = entry_price − (|entry_price − SL| × R:R)
```

---

## 8. Layer 6 — Risk Management

### Kelly Criterion (Empirical Win Rate)
```
f* = (p × b − q) / b

where:
  p = empirical win probability from SQLite signal history
      = COUNT(outcome='WIN') / COUNT(*) for signals of same (strength, regime) bucket
      Falls back to p = 0.50 (coin-flip prior) if < 30 closed trades in bucket
  q = 1 − p
  b = realized avg_win_pct / avg_loss_pct from same bucket
      Falls back to b = R:R ratio if < 30 closed trades

Fractional Kelly: position_size = 0.25 × f*   (quarter-Kelly; standard practice per Ed Thorp)
Hard cap:         position_size = min(position_size, 0.05)  (never > 5% per trade)
Cold start rule:  Until 30 closed trades exist in a bucket, use hard cap directly (2%)
```

**Why empirical, not confidence-score**: The confidence score measures signal strength (how far alpha_score is from zero), not calibrated win probability. Equating them would overstate Kelly fractions on a system with no verified live track record. Win rate must be earned from real signal outcomes.

### Portfolio-Level Risk Controls
- **Daily drawdown circuit breaker**: portfolio PnL < −5% today → all new signals suppressed until next session
- **Max open positions**: 8 simultaneous across all asset classes
- **Correlation filter**: cross-asset-class applies — do not open any 2 positions where rolling 30d correlation magnitude > 0.7. This includes Forex-to-Indian-equity correlations (USD/INR is strongly correlated with Nifty during risk-off events)
- **Max per-asset-class exposure**: 40% Indian stocks, 40% US stocks, 20% Forex (by position_size_pct sum)
- **VaR limit**: Daily 95% VaR must not exceed 3% of portfolio

### Value at Risk
```
Historical VaR (95%) = 5th percentile of rolling 252-day daily returns × portfolio_value
Expected Shortfall (CVaR) = mean of all returns below VaR threshold
```

---

## 9. Layer 7 — Portfolio Tracker

### Transaction Cost Model (for realistic demo PnL)
```
Indian equities: 0.03% per side (≈ NSE + STT approximation)
US equities:     0.01% per side (tight spread liquid stocks)
Forex majors:    0.5 pip = 0.005% per side (EUR/USD, GBP/USD)
Forex minors:    1.0 pip = 0.010% per side (USD/INR, USD/JPY)

Applied to all PnL calculations: net_pnl = gross_pnl − (2 × slippage_cost_pct × position_value)
```

### Performance Metrics (computed continuously)
| Metric | Formula | Target |
|--------|---------|--------|
| Sharpe Ratio | (Rp − Rf) / σp × √252 | > 1.5 |
| Sortino Ratio | (Rp − Rf) / σ_downside × √252 | > 2.0 |
| Calmar Ratio | CAGR / Max_Drawdown | > 1.0 |
| Max Drawdown | max(1 − Vt / V_peak) | < 15% |
| Win Rate | wins / closed_trades | |
| Profit Factor | gross_profit / gross_loss | > 1.5 |
| Avg Realized R:R | avg_win / avg_loss | |
| Daily VaR 95% | 5th pct of 252d daily returns | < 3% |
| Information Ratio | IC_mean / IC_std (per factor) | > 0.3 |

### Open Position Tracking
- Mark-to-market PnL updated every 5-min poll cycle (net of transaction costs)
- Auto-generates CLOSE signal when price hits SL or TP level
- Tracks unrealized PnL, realized PnL, total PnL by asset class

---

## 10. Layer 8 — Live Scheduler

### Intraday Loop (market hours only)
```
Every 5 minutes:
  1. Fetch latest candles via yfinance for all watchlist assets (no AV budget used)
  2. Attempt NSE scraper for Indian assets → fallback chain if needed
  3. Run feature engineering on new candles (Hurst skipped — cached from EOD)
  4. Score all 5 factors
  5. Run HMM regime check (model already trained; just run Viterbi on new data)
  6. Generate signals for assets with |alpha_score| > threshold
  7. Apply risk filters (correlation, drawdown circuit, position limits)
  8. Persist to SQLite + push to dashboard via WebSocket
  9. Fire desktop alert + sound for STRONG signals

Market session hours:
  NSE:    09:15 – 15:30 IST
  NYSE:   09:30 – 16:00 EST  (19:00 – 01:30 IST next day)
  Forex:  24/5 (Sun 17:00 EST – Fri 17:00 EST)
```

### EOD Swing Loop
```
Triggered after each market close:
  1. Fetch daily OHLCV for all watchlist assets (yfinance + AV Forex EOD)
  2. Compute Hurst exponents for all assets (stored in cache for next day's intraday)
  3. Full feature engineering pass on daily data
  4. Retrain HMM if 7 days since last retrain
  5. Run WFO parameter validation if 30 days since last validation
  6. Generate swing signals (next-day / multi-day horizon)
  7. Persist + push to dashboard
  8. Update win/loss outcomes for signals where SL/TP was hit during the session
```

---

## 11. Layer 9 — Signal Store (SQLite)

### Schema
```sql
CREATE TABLE signals (
  signal_id         TEXT PRIMARY KEY,
  timestamp         TEXT NOT NULL,
  asset             TEXT NOT NULL,
  asset_class       TEXT NOT NULL,
  timeframe         TEXT NOT NULL,
  signal            TEXT NOT NULL,
  strength          TEXT NOT NULL,
  confidence        REAL NOT NULL,
  alpha_score       REAL NOT NULL,
  regime            TEXT NOT NULL,
  entry_price       REAL NOT NULL,
  stop_loss         REAL NOT NULL,
  take_profit       REAL NOT NULL,
  position_size_pct REAL NOT NULL,
  kelly_fraction    REAL NOT NULL,
  hurst_exponent    REAL,
  factor_scores     TEXT NOT NULL,  -- JSON: {F1, F2, F3, F4, F5}
  slippage_cost_pct REAL NOT NULL,
  outcome           TEXT,           -- NULL until closed: 'WIN' | 'LOSS' | 'PARTIAL'
  close_price       REAL,           -- price when SL/TP hit
  pnl_pct           REAL            -- net PnL including transaction costs
);

CREATE INDEX idx_signals_asset_time   ON signals (asset, timestamp);
CREATE INDEX idx_signals_outcome_bucket ON signals (asset_class, signal, strength, outcome);

CREATE TABLE portfolio_snapshots (
  snapshot_id     TEXT PRIMARY KEY,
  timestamp       TEXT NOT NULL,
  total_pnl_pct   REAL,
  sharpe          REAL,
  sortino         REAL,
  calmar          REAL,
  max_drawdown    REAL,
  open_positions  INTEGER,
  daily_var_95    REAL,
  win_rate        REAL,
  profit_factor   REAL
);

CREATE INDEX idx_portfolio_time ON portfolio_snapshots (timestamp);
```

---

## 12. Layer 10 — Output Layer

### Web Dashboard (FastAPI + WebSocket)
- Binds to **`127.0.0.1` (localhost only)** — never `0.0.0.0` to prevent network exposure
- Real-time updates via WebSocket; all pages auto-refresh without polling

**Pages:**
1. **Live Signals** — real-time signal cards with entry/SL/TP, factor breakdown, confidence bar; filter by asset class / timeframe / strength
2. **Portfolio** — open positions table with live mark-to-market PnL, drawdown chart, metrics panel (Sharpe, Sortino, Calmar, VaR, Win Rate)
3. **Signal History** — all closed signals, outcome tracking, win rate by asset/timeframe/strength
4. **Factor Analysis** — per-asset factor score table, rolling IC chart per factor (shows which factors are currently predictive), Hurst exponent trend
5. **Regime Monitor** — current HMM state + state probabilities per asset class, transition matrix visualization

### CLI Mode
```bash
python main.py --mode live          # full live system (scheduler + dashboard)
python main.py --mode dashboard     # dashboard only (no data fetching)
python main.py --mode backtest      # historical WFO validation
python main.py --mode signals       # terminal-only signals (no dashboard)
```

### Alerts
- **Desktop notification** (plyer): fires on STRONG signal — shows asset, direction, confidence, entry/SL/TP
- **Sound**: distinct beep for BUY vs SELL (different frequency)
- **Terminal**: colorama — green BUY, red SELL, yellow HOLD, bold for STRONG

---

## 13. Signal Output Format

```
╔══════════════════════════════════════════════════════════╗
║  SIGNAL FIRED — 2026-03-25 14:32:05 IST                ║
║  Asset     : AAPL (US Stock)                            ║
║  Timeframe : INTRADAY (5m candles)                     ║
║  Signal    : ██████████ BUY  [STRONG]                  ║
║  Confidence: 87/100  |  Alpha Score: +0.72             ║
║  Regime    : BULL  |  Hurst: 0.61 (trending)           ║
║  ─────────────────────────────────────────────────────  ║
║  Entry     : $172.45                                    ║
║  Stop Loss : $170.20  (−1.3%)   [1.5× ATR]            ║
║  Target    : $179.15  (+3.9%)   [3:1 R:R]             ║
║  ─────────────────────────────────────────────────────  ║
║  Position  : 2.1% of portfolio  [Kelly raw: 9.4%→cap] ║
║  Est. Cost : 0.02% slippage                            ║
║  ─────────────────────────────────────────────────────  ║
║  Factors   : F1:Momentum +0.8  F2:MeanRev +0.3        ║
║              F3:Volume   +0.6  F4:LSTM    +0.9        ║
║              F5:VolSqz   +0.4                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## 14. New Dependencies

```
hmmlearn          # Hidden Markov Model (GaussianHMM, full covariance)
plyer             # Desktop notifications
colorama          # Colored terminal output
scipy             # Hurst exponent R/S analysis, statistical tests
sqlalchemy        # SQLite ORM
apscheduler       # Intraday + EOD job scheduling (APScheduler 3.x)
pyyaml            # config.yaml watchlist + thresholds
jinja2            # Dashboard HTML templating
```

---

## 15. File Structure

```
Trading/
├── src/
│   ├── api/
│   │   ├── connectors.py          (extended: yfinance intraday + AV Forex EOD)
│   │   └── rate_limiter.py        (NEW: token bucket + daily AV budget tracker)
│   ├── features/
│   │   ├── engineer.py            (extended: 30+ indicators, VWAP, OBV, CMF, Ichimoku)
│   │   └── hurst.py               (NEW: R/S Hurst exponent — EOD only, cached)
│   ├── alpha/
│   │   ├── factor_model.py        (NEW: 5-factor IC-weighted alpha model, tanh mapping)
│   │   └── regime.py              (NEW: HMM, 4 emissions, full covariance, Viterbi)
│   ├── signals/
│   │   ├── engine.py              (NEW: signal generation + SL/TP + slippage)
│   │   └── store.py               (NEW: SQLite persistence with indexes)
│   ├── risk/
│   │   ├── kelly.py               (NEW: empirical Kelly + cold-start rule + cross-class correlation)
│   │   └── portfolio.py           (NEW: portfolio tracker + VaR + all metrics)
│   ├── scheduler/
│   │   └── live_runner.py         (NEW: APScheduler intraday + EOD loops + market hours)
│   ├── dashboard/
│   │   ├── api.py                 (NEW: FastAPI app, localhost only, WebSocket)
│   │   └── static/
│   │       ├── index.html         (live signals page)
│   │       ├── portfolio.html     (portfolio + metrics page)
│   │       ├── history.html       (signal history + outcomes page)
│   │       ├── factors.html       (IC chart + factor scores page)
│   │       └── regime.html        (HMM state + probabilities page)
│   ├── models/
│   │   └── trend_model.py         (modified: MC Dropout via model(X, training=True) ×20)
│   ├── utils/
│   │   ├── math_utils.py          (extended: Sortino, Calmar, VaR, CVaR, IR)
│   │   └── alerts.py              (NEW: plyer desktop + sound + colorama CLI)
│   └── validator.py               (extended: WFO now covers 5-factor parameter validation)
├── config.yaml                    (NEW: watchlist, thresholds, market hours)
├── data/
│   └── signals.db                 (auto-created SQLite)
├── docs/superpowers/specs/
│   └── 2026-03-25-quant-trading-system-design.md
├── main.py                        (extended: --mode flag)
└── requirements.txt               (updated)
```

---

## 16. Success Criteria

- System generates at least 1 actionable signal per trading session per asset class
- Confidence score (tanh-mapped) calibration: higher confidence buckets show higher empirical win rate
- Alpha Vantage daily budget stays under 20 requests/day (well within 500 limit)
- Dashboard binds to localhost only; loads in < 2 seconds; WebSocket latency < 500ms
- All signals persisted with entry/SL/TP/slippage for outcome tracking
- WFO out-of-sample Information Ratio > 0.3 for chosen alpha model parameters
- Portfolio Sharpe > 1.0 after first 30 closed trades

---

## 17. What's NOT in Scope (Phase 2)

- Automated order execution (broker API integration)
- Options / derivatives signals
- Sentiment analysis (news/social media NLP)
- Transformer model replacing LSTM
- Multi-machine deployment / cloud hosting
- Full backtesting engine with realistic market impact / transaction cost modeling
