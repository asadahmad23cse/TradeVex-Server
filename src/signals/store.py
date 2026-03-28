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
            conn.commit()
        logger.info("SignalStore schema initialised (with execution tracking columns).")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_signal(self, sig) -> None:
        """Persist a TradingSignal object."""
        d = sig.to_dict()
        sql = text("""
            INSERT OR REPLACE INTO signals
            (signal_id, timestamp, asset, asset_class, timeframe,
             signal, strength, confidence, alpha_score, regime,
             entry_price, stop_loss, take_profit, position_size_pct,
             kelly_fraction, hurst_exponent, factor_scores, ic_weights,
             slippage_cost_pct, cost_pct, net_alpha_score, cs_alpha_score,
             execution_price, implementation_shortfall_pct)
            VALUES
            (:signal_id, :timestamp, :asset, :asset_class, :timeframe,
             :signal, :strength, :confidence, :alpha_score, :regime,
             :entry_price, :stop_loss, :take_profit, :position_size_pct,
             :kelly_fraction, :hurst_exponent, :factor_scores, :ic_weights,
             :slippage_cost_pct, :cost_pct, :net_alpha_score, :cs_alpha_score,
             :execution_price, :implementation_shortfall_pct)
        """)
        with self.engine.connect() as conn:
            conn.execute(sql, d)
            conn.commit()

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
