"""
Layer 9 — Signal Store (SQLite via SQLAlchemy).

Persists every TradingSignal and portfolio snapshot.
Provides queries for:
  - Retrieving win-rate / avg R:R by (strength, regime) bucket  → used by Kelly
  - Signal history for dashboard
  - Portfolio performance snapshots
"""

import json
import logging
from pathlib import Path

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/signals.db")

CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
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
    factor_scores     TEXT NOT NULL,
    ic_weights        TEXT,
    slippage_cost_pct REAL NOT NULL DEFAULT 0,
    cost_pct          REAL DEFAULT 0,
    net_alpha_score   REAL DEFAULT 0,
    cs_alpha_score    REAL DEFAULT 0,
    execution_price   REAL DEFAULT 0,
    implementation_shortfall_pct REAL DEFAULT 0,
    outcome           TEXT,
    close_price       REAL,
    pnl_pct           REAL
);
"""

CREATE_IDX_ASSET = """
CREATE INDEX IF NOT EXISTS idx_signals_asset_time
    ON signals (asset, timestamp);
"""

CREATE_IDX_BUCKET = """
CREATE INDEX IF NOT EXISTS idx_signals_outcome_bucket
    ON signals (asset_class, signal, strength, outcome);
"""

CREATE_IDX_REGIME_BUCKET = """
CREATE INDEX IF NOT EXISTS idx_signals_regime_bucket
    ON signals (asset_class, strength, regime, outcome);
"""

CREATE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id    TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    total_pnl_pct  REAL,
    sharpe         REAL,
    sortino        REAL,
    calmar         REAL,
    max_drawdown   REAL,
    open_positions INTEGER,
    daily_var_95   REAL,
    win_rate       REAL,
    profit_factor  REAL
);
"""

CREATE_IDX_SNAP = """
CREATE INDEX IF NOT EXISTS idx_portfolio_time
    ON portfolio_snapshots (timestamp);
"""

CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    order_id          TEXT PRIMARY KEY,
    signal_id         TEXT,
    asset             TEXT NOT NULL,
    side              TEXT NOT NULL,
    broker            TEXT NOT NULL,
    state             TEXT NOT NULL,
    requested_price   REAL,
    fill_price        REAL,
    requested_qty     REAL,
    filled_qty        REAL,
    slippage_pct      REAL,
    expected_slippage_pct REAL,
    error             TEXT,
    broker_payload    TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
"""

CREATE_ORDER_EVENTS = """
CREATE TABLE IF NOT EXISTS order_events (
    event_id          TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL,
    old_state         TEXT,
    new_state         TEXT NOT NULL,
    event_type        TEXT NOT NULL,
    details           TEXT,
    timestamp         TEXT NOT NULL
);
"""

CREATE_RECONCILIATION_EVENTS = """
CREATE TABLE IF NOT EXISTS reconciliation_events (
    event_id          TEXT PRIMARY KEY,
    scope             TEXT NOT NULL,
    asset             TEXT,
    status            TEXT NOT NULL,
    severity          TEXT NOT NULL,
    message           TEXT NOT NULL,
    details           TEXT,
    timestamp         TEXT NOT NULL
);
"""

CREATE_DATA_QUALITY_EVENTS = """
CREATE TABLE IF NOT EXISTS data_quality_events (
    event_id          TEXT PRIMARY KEY,
    asset             TEXT NOT NULL,
    asset_class       TEXT NOT NULL,
    timeframe         TEXT NOT NULL,
    severity          TEXT NOT NULL,
    issue_types       TEXT NOT NULL,
    details           TEXT NOT NULL,
    timestamp         TEXT NOT NULL
);
"""

CREATE_MODEL_VALIDATION = """
CREATE TABLE IF NOT EXISTS model_validation (
    validation_id     TEXT PRIMARY KEY,
    model_name        TEXT NOT NULL,
    asset_class       TEXT NOT NULL,
    metrics           TEXT NOT NULL,
    top_features      TEXT,
    timestamp         TEXT NOT NULL
);
"""

CREATE_SYSTEM_HEALTH = """
CREATE TABLE IF NOT EXISTS system_health (
    health_id         TEXT PRIMARY KEY,
    component         TEXT NOT NULL,
    status            TEXT NOT NULL,
    message           TEXT NOT NULL,
    details           TEXT,
    timestamp         TEXT NOT NULL
);
"""

CREATE_WEBHOOK_EVENTS = """
CREATE TABLE IF NOT EXISTS webhook_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    source      TEXT,
    ticker      TEXT,
    action      TEXT,
    status      TEXT,
    reason      TEXT,
    order_id    TEXT,
    sized_qty   REAL,
    sl_used     REAL,
    tp_used     REAL,
    auto_sized  INTEGER,
    auto_sltp   INTEGER
);
"""


class SignalStore:

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        db_path_str = str(db_path)
        if "://" in db_path_str:
            self.engine = create_engine(db_path_str, echo=False)
        else:
            local_path = Path(db_path_str)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(f"sqlite:///{local_path}", echo=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.engine.connect() as conn:
            for stmt in [
                CREATE_SIGNALS,
                CREATE_IDX_ASSET,
                CREATE_IDX_BUCKET,
                CREATE_IDX_REGIME_BUCKET,
                CREATE_SNAPSHOTS,
                CREATE_IDX_SNAP,
                CREATE_ORDERS,
                CREATE_ORDER_EVENTS,
                CREATE_RECONCILIATION_EVENTS,
                CREATE_DATA_QUALITY_EVENTS,
                CREATE_MODEL_VALIDATION,
                CREATE_SYSTEM_HEALTH,
                CREATE_WEBHOOK_EVENTS,
            ]:
                conn.execute(text(stmt))
            # Migration: add new columns if they don't exist (SQLite safe ALTER TABLE)
            new_cols = [
                ("cost_pct",                          "REAL DEFAULT 0"),
                ("net_alpha_score",                   "REAL DEFAULT 0"),
                ("cs_alpha_score",                    "REAL DEFAULT 0"),
                ("execution_price",                   "REAL DEFAULT 0"),
                ("implementation_shortfall_pct",      "REAL DEFAULT 0"),
            ]
            for col_name, col_def in new_cols:
                try:
                    conn.execute(text(f"ALTER TABLE signals ADD COLUMN {col_name} {col_def}"))
                except Exception:
                    pass  # column already exists
            _explain_cols = [
                ("confluence_grade", "TEXT"),
                ("confluence_pct", "REAL"),
                ("sqs", "INTEGER"),
                ("sqs_grade", "TEXT"),
                ("size_multiplier_used", "REAL"),
                ("factor_breakdown", "TEXT"),
                ("top_aligned_factors", "TEXT"),
                ("top_drag_factors", "TEXT"),
            ]
            for col_name, col_type in _explain_cols:
                try:
                    conn.execute(text(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}"))
                except Exception:
                    pass
            conn.commit()
        logger.info("SignalStore schema initialised (with execution tracking columns).")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_signal(self, sig) -> None:
        """Persist a TradingSignal object."""
        try:
            signal_data = sig.to_dict()
            logger.info(f"Saving signal to DB: {signal_data}")

            sql = text("""
                INSERT OR REPLACE INTO signals
                (signal_id, timestamp, asset, asset_class, timeframe,
                 signal, strength, confidence, alpha_score, regime,
                 entry_price, stop_loss, take_profit, position_size_pct,
                 kelly_fraction, hurst_exponent, factor_scores, ic_weights,
                 slippage_cost_pct, cost_pct, net_alpha_score, cs_alpha_score,
                 execution_price, implementation_shortfall_pct,
                 confluence_grade, confluence_pct, sqs, sqs_grade,
                 size_multiplier_used, factor_breakdown, top_aligned_factors,
                 top_drag_factors)
                VALUES
                (:signal_id, :timestamp, :asset, :asset_class, :timeframe,
                 :signal, :strength, :confidence, :alpha_score, :regime,
                 :entry_price, :stop_loss, :take_profit, :position_size_pct,
                 :kelly_fraction, :hurst_exponent, :factor_scores, :ic_weights,
                 :slippage_cost_pct, :cost_pct, :net_alpha_score, :cs_alpha_score,
                 :execution_price, :implementation_shortfall_pct,
                 :confluence_grade, :confluence_pct, :sqs, :sqs_grade,
                 :size_multiplier_used, :factor_breakdown, :top_aligned_factors,
                 :top_drag_factors)
            """)
            with self.engine.connect() as conn:
                conn.execute(sql, signal_data)
                conn.commit()
                row_count = conn.execute(text("SELECT COUNT(*) FROM signals")).scalar_one()
            logger.info(f"Signal saved successfully, DB row count: {row_count}")
        except Exception as e:
            logger.error(f"Failed to save signal: {e}", exc_info=True)
            raise

    def close_signal(
        self,
        signal_id: str,
        outcome: str,     # 'WIN' | 'LOSS' | 'PARTIAL'
        close_price: float,
        pnl_pct: float,
    ) -> None:
        """Mark a signal as closed with its outcome."""
        sql = text("""
            UPDATE signals
            SET outcome = :outcome, close_price = :close_price, pnl_pct = :pnl_pct
            WHERE signal_id = :signal_id
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, {
                "signal_id": signal_id,
                "outcome": outcome,
                "close_price": close_price,
                "pnl_pct": pnl_pct,
            })
            conn.commit()

    def save_snapshot(self, snapshot: dict) -> None:
        """Persist a portfolio snapshot dict."""
        sql = text("""
            INSERT OR REPLACE INTO portfolio_snapshots
            (snapshot_id, timestamp, total_pnl_pct, sharpe, sortino, calmar,
             max_drawdown, open_positions, daily_var_95, win_rate, profit_factor)
            VALUES
            (:snapshot_id, :timestamp, :total_pnl_pct, :sharpe, :sortino, :calmar,
             :max_drawdown, :open_positions, :daily_var_95, :win_rate, :profit_factor)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, snapshot)
            conn.commit()

    def save_order(self, order: dict) -> None:
        sql = text("""
            INSERT OR REPLACE INTO orders
            (order_id, signal_id, asset, side, broker, state, requested_price,
             fill_price, requested_qty, filled_qty, slippage_pct,
             expected_slippage_pct, error, broker_payload, created_at, updated_at)
            VALUES
            (:order_id, :signal_id, :asset, :side, :broker, :state, :requested_price,
             :fill_price, :requested_qty, :filled_qty, :slippage_pct,
             :expected_slippage_pct, :error, :broker_payload, :created_at, :updated_at)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, order)
            conn.commit()

    def save_order_event(self, event: dict) -> None:
        sql = text("""
            INSERT OR REPLACE INTO order_events
            (event_id, order_id, old_state, new_state, event_type, details, timestamp)
            VALUES
            (:event_id, :order_id, :old_state, :new_state, :event_type, :details, :timestamp)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, event)
            conn.commit()

    def save_reconciliation_event(self, event: dict) -> None:
        sql = text("""
            INSERT OR REPLACE INTO reconciliation_events
            (event_id, scope, asset, status, severity, message, details, timestamp)
            VALUES
            (:event_id, :scope, :asset, :status, :severity, :message, :details, :timestamp)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, event)
            conn.commit()

    def save_data_quality_event(self, event: dict) -> None:
        sql = text("""
            INSERT OR REPLACE INTO data_quality_events
            (event_id, asset, asset_class, timeframe, severity, issue_types, details, timestamp)
            VALUES
            (:event_id, :asset, :asset_class, :timeframe, :severity, :issue_types, :details, :timestamp)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, event)
            conn.commit()

    def save_model_validation(self, record: dict) -> None:
        sql = text("""
            INSERT OR REPLACE INTO model_validation
            (validation_id, model_name, asset_class, metrics, top_features, timestamp)
            VALUES
            (:validation_id, :model_name, :asset_class, :metrics, :top_features, :timestamp)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, record)
            conn.commit()

    def save_system_health(self, event: dict) -> None:
        sql = text("""
            INSERT OR REPLACE INTO system_health
            (health_id, component, status, message, details, timestamp)
            VALUES
            (:health_id, :component, :status, :message, :details, :timestamp)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, event)
            conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_bucket_stats(
        self,
        asset_class: str,
        strength: str,
        regime: str,
        signal: str | None = None,
    ) -> dict:
        """
        Returns empirical win rate and avg R:R for the Kelly Criterion.
        bucket = (asset_class, strength, regime) with closed trades.
        """
        sql = """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS wins,
                AVG(CASE WHEN pnl_pct > 0 THEN pnl_pct ELSE NULL END) AS avg_win,
                AVG(CASE WHEN pnl_pct < 0 THEN ABS(pnl_pct) ELSE NULL END) AS avg_loss
            FROM signals
            WHERE asset_class = :ac
              AND strength = :str
              AND regime = :reg
              AND outcome IS NOT NULL
        """
        params = {"ac": asset_class, "str": strength, "reg": regime}
        if signal is not None:
            sql += "\n              AND signal = :sig"
            params["sig"] = signal
        query = text(sql)
        with self.engine.connect() as conn:
            row = conn.execute(query, params).fetchone()

        if row is None or row.total == 0:
            return {"total": 0, "win_rate": None, "avg_win": None, "avg_loss": None}

        win_rate = row.wins / row.total if row.total > 0 else None
        return {
            "total": row.total,
            "win_rate": win_rate,
            "avg_win": row.avg_win,
            "avg_loss": row.avg_loss,
        }

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        """Fetch the most recent N signals for the dashboard."""
        sql = text("""
            SELECT * FROM signals ORDER BY timestamp DESC LIMIT :lim
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"lim": limit}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_open_signals(self) -> list[dict]:
        """Fetch all signals without a closed outcome."""
        sql = text("""
            SELECT * FROM signals WHERE outcome IS NULL ORDER BY timestamp DESC
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(sql).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_closed_returns(self) -> list[float]:
        """Return list of net pnl_pct for all closed signals."""
        sql = text("SELECT pnl_pct FROM signals WHERE pnl_pct IS NOT NULL")
        with self.engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [r[0] for r in rows if r[0] is not None]

    def get_latest_snapshot(self) -> dict | None:
        """Fetch the most recent portfolio snapshot."""
        sql = text("""
            SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1
        """)
        with self.engine.connect() as conn:
            row = conn.execute(sql).mappings().fetchone()
        return dict(row) if row else None

    def get_recent_orders(self, limit: int = 50) -> list[dict]:
        sql = text("SELECT * FROM orders ORDER BY updated_at DESC LIMIT :lim")
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"lim": limit}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_recent_reconciliation_events(self, limit: int = 50) -> list[dict]:
        sql = text("SELECT * FROM reconciliation_events ORDER BY timestamp DESC LIMIT :lim")
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"lim": limit}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_recent_data_quality_events(self, limit: int = 50) -> list[dict]:
        sql = text("SELECT * FROM data_quality_events ORDER BY timestamp DESC LIMIT :lim")
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"lim": limit}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_recent_model_validation(self, limit: int = 20) -> list[dict]:
        sql = text("SELECT * FROM model_validation ORDER BY timestamp DESC LIMIT :lim")
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"lim": limit}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_recent_system_health(self, limit: int = 50) -> list[dict]:
        sql = text("SELECT * FROM system_health ORDER BY timestamp DESC LIMIT :lim")
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"lim": limit}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_signal_by_id(self, signal_id) -> dict:
        """Returns single signal dict by id. {} if not found."""
        try:
            sid = str(signal_id).strip()
            if not sid:
                return {}
            sql = text("SELECT * FROM signals WHERE signal_id = :sid LIMIT 1")
            with self.engine.connect() as conn:
                row = conn.execute(sql, {"sid": sid}).mappings().fetchone()
            if row is None:
                return {}
            d = dict(row)
            for k in ("factor_breakdown",):
                raw = d.get(k)
                if isinstance(raw, str) and raw.strip():
                    try:
                        d[k] = json.loads(raw)
                    except Exception:
                        d[k] = {}
            for k in ("top_aligned_factors", "top_drag_factors"):
                raw = d.get(k)
                if isinstance(raw, str) and raw.strip():
                    try:
                        d[k] = json.loads(raw)
                    except Exception:
                        d[k] = []
            return d
        except Exception:
            return {}

    def get_signal_quality_stats(self, days: int = 7) -> dict:
        """
        Aggregated SQS stats for recent window.
        suppressed_count from system_health messages when available.
        """
        try:
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
            recent = self.get_recent_signals(limit=3000)
            filtered: list[dict] = []
            for r in recent:
                ts = r.get("timestamp")
                try:
                    if not isinstance(ts, str):
                        continue
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    if t >= cutoff:
                        filtered.append(r)
                except Exception:
                    continue

            premium_count = sum(1 for r in filtered if r.get("sqs_grade") == "PREMIUM")
            standard_count = sum(1 for r in filtered if r.get("sqs_grade") == "STANDARD")
            weak_count = sum(1 for r in filtered if r.get("sqs_grade") == "WEAK")
            sqs_vals = [float(r["sqs"]) for r in filtered if r.get("sqs") is not None]
            conf_vals = [
                float(r["confluence_pct"])
                for r in filtered
                if r.get("confluence_pct") is not None
            ]
            avg_sqs = float(sum(sqs_vals) / len(sqs_vals)) if sqs_vals else 0.0
            avg_conf = float(sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0

            suppressed_count = 0
            try:
                csql = text("""
                    SELECT COUNT(*) AS n FROM system_health
                    WHERE timestamp >= :ts
                      AND (
                        message LIKE '%Signal suppressed by SQS%'
                        OR message LIKE '%confluence grade D%'
                      )
                """)
                with self.engine.connect() as conn:
                    crow = conn.execute(csql, {"ts": cutoff.isoformat()}).fetchone()
                if crow is not None and crow[0] is not None:
                    suppressed_count = int(crow[0])
            except Exception:
                suppressed_count = 0

            return {
                "last_7d": {
                    "premium_count": premium_count,
                    "standard_count": standard_count,
                    "weak_count": weak_count,
                    "suppressed_count": suppressed_count,
                    "avg_sqs": round(avg_sqs, 2),
                    "avg_confluence_pct": round(avg_conf, 2),
                }
            }
        except Exception:
            return {"last_7d": {}}

    def save_regime_analysis(self, analysis: dict) -> None:
        """Save regime analysis as JSON in system_health table.
        Uses existing system_health table — no schema change needed.
        Key: 'regime_analysis', value: json.dumps(analysis)
        SAFETY: fails silently on any exception."""
        try:
            from datetime import datetime, timezone
            import uuid

            payload = {
                "health_id": str(uuid.uuid4()),
                "component": "regime_analysis",
                "status": "info",
                "message": "regime_analysis_snapshot",
                "details": json.dumps({"regime_analysis": analysis}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.save_system_health(payload)
        except Exception:
            pass

    def get_latest_regime_analysis(self) -> dict:
        """Retrieve latest regime_analysis from system_health table.
        Returns {} if not found."""
        try:
            sql = text("""
                SELECT details FROM system_health
                WHERE component = :comp
                ORDER BY timestamp DESC LIMIT 1
            """)
            with self.engine.connect() as conn:
                row = conn.execute(sql, {"comp": "regime_analysis"}).fetchone()
            if row is None or row[0] is None:
                return {}
            raw = row[0]
            if isinstance(raw, str):
                d = json.loads(raw)
            else:
                d = json.loads(str(raw))
            if not isinstance(d, dict):
                return {}
            return d.get("regime_analysis") or {}
        except Exception:
            return {}

    def save_webhook_event(self, event: dict) -> None:
        """Persist a webhook audit row (no raw payload — may contain secrets)."""
        try:
            sql = text("""
                INSERT INTO webhook_events
                (timestamp, source, ticker, action, status, reason, order_id,
                 sized_qty, sl_used, tp_used, auto_sized, auto_sltp)
                VALUES
                (:timestamp, :source, :ticker, :action, :status, :reason, :order_id,
                 :sized_qty, :sl_used, :tp_used, :auto_sized, :auto_sltp)
            """)
            row = {
                "timestamp": event.get("timestamp", ""),
                "source": event.get("source"),
                "ticker": event.get("ticker"),
                "action": event.get("action"),
                "status": event.get("status"),
                "reason": event.get("reason"),
                "order_id": event.get("order_id"),
                "sized_qty": event.get("sized_qty"),
                "sl_used": event.get("sl_used"),
                "tp_used": event.get("tp_used"),
                "auto_sized": int(event.get("auto_sized") or 0),
                "auto_sltp": int(event.get("auto_sltp") or 0),
            }
            with self.engine.connect() as conn:
                conn.execute(sql, row)
                conn.commit()
        except Exception as exc:
            logger.warning("save_webhook_event failed (non-critical): %s", exc)

    def get_webhook_log(self, limit: int = 50) -> list:
        """Last N webhook events, newest first; each row as a dict."""
        try:
            lim = max(1, min(int(limit), 500))
            sql = text("""
                SELECT id, timestamp, source, ticker, action, status, reason, order_id,
                       sized_qty, sl_used, tp_used, auto_sized, auto_sltp
                FROM webhook_events
                ORDER BY id DESC
                LIMIT :lim
            """)
            with self.engine.connect() as conn:
                rows = conn.execute(sql, {"lim": lim}).fetchall()
            out: list = []
            for r in rows:
                out.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "source": r[2],
                    "ticker": r[3],
                    "action": r[4],
                    "status": r[5],
                    "reason": r[6],
                    "order_id": r[7],
                    "sized_qty": r[8],
                    "sl_used": r[9],
                    "tp_used": r[10],
                    "auto_sized": bool(r[11]) if r[11] is not None else False,
                    "auto_sltp": bool(r[12]) if r[12] is not None else False,
                })
            return out
        except Exception:
            return []
