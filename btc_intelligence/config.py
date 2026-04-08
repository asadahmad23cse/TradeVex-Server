from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=('btc_intelligence/.env', '.env'), env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'Advanced Bitcoin Trade Intelligence System v2.0'
    host: str = '0.0.0.0'
    port: int = 9000
    log_level: str = 'INFO'

    # Exchange/data APIs.
    binance_ws_url: str = (
        'wss://fstream.binance.com/stream?streams='
        'btcusdt@kline_1m/'
        'btcusdt@kline_5m/'
        'btcusdt@kline_15m/'
        'btcusdt@kline_1h/'
        'btcusdt@kline_4h/'
        'btcusdt@aggTrade/'
        'btcusdt@depth20@100ms/'
        'btcusdt@forceOrder'
    )
    binance_rest_base: str = 'https://fapi.binance.com'

    bybit_ws_url: str = 'wss://stream.bybit.com/v5/public/linear'
    okx_ws_url: str = 'wss://ws.okx.com:8443/ws/v5/public'

    coinglass_base: str = 'https://open-api-v3.coinglass.com'
    glassnode_base: str = 'https://api.glassnode.com/v1/metrics'
    deribit_base: str = 'https://www.deribit.com/api/v2'
    whale_alert_base: str = 'https://api.whale-alert.io/v1/transactions'
    cryptopanic_base: str = 'https://cryptopanic.com/api/free/v1/posts/'

    fng_api: str = 'https://api.alternative.me/fng/'
    fred_api: str = 'https://api.stlouisfed.org/fred/series/observations'

    # Credentials.
    binance_api_key: str = ''
    binance_api_secret: str = ''
    bybit_api_key: str = ''
    bybit_api_secret: str = ''
    okx_api_key: str = ''
    okx_api_secret: str = ''
    okx_passphrase: str = ''

    coinglass_api_key: str = ''
    glassnode_api_key: str = ''
    deribit_client_id: str = ''
    deribit_client_secret: str = ''
    arkham_api_key: str = ''
    cryptopanic_api_key: str = ''
    fred_api_key: str = ''

    telegram_bot_token: str = ''
    telegram_chat_id: str = ''

    # Portfolio/risk.
    portfolio_usdt: float = 10000.0
    risk_per_trade_pct: float = 1.0
    max_portfolio_heat_pct: float = 3.0
    drawdown_pause_pct: float = 15.0
    signal_cooldown_hours: int = 4
    signal_stale_minutes: int = 30

    min_confidence: float = 60.0
    min_net_alpha: float = 0.0

    auto_retrain_days: int = 30
    auto_pause_loss_streak: int = 5
    auto_pause_winrate_floor: float = 0.48

    paper_trade: bool = True
    redis_url: str = 'redis://localhost:6379'
    redis_state_enabled: bool = True
    redis_key_prefix: str = 'btc'

    # Buffers.
    candles_1m_max: int = 500
    candles_5m_max: int = 300
    candles_15m_max: int = 200
    candles_1h_max: int = 100
    candles_4h_max: int = 50
    trades_max: int = 500
    depth_max_levels: int = 20
    signal_history_size: int = 100

    # Runtime intervals.
    feature_eval_interval_sec: int = 15
    binance_rest_poll_sec: int = 30
    multi_exchange_poll_sec: int = 15
    coinglass_poll_sec: int = 60
    glassnode_poll_sec: int = 300
    deribit_poll_sec: int = 60
    whale_poll_sec: int = 300
    macro_poll_sec: int = 900
    cryptopanic_poll_sec: int = 300

    # Files.
    data_path: str = 'btc_intelligence/data'
    model_dir: str = 'btc_intelligence/models/artifacts'
    signal_log_path: str = 'btc_intelligence/logs/signals.jsonl'
    app_log_path: str = 'btc_intelligence/logs/app.jsonl'
    monitoring_log_path: str = 'btc_intelligence/logs/monitoring.jsonl'
    edge_store_path: str = 'btc_intelligence/logs/edge_stats.json'

    # Guard rails.
    max_ws_retries: int = 5
    max_price_spike_pct: float = 5.0
    spread_reject_pct: float = 0.05
    slippage_reject_pct: float = 0.03

    # Adverse selection / fill-quality (tape + book); see signals/execution_adverse_selection.py
    adverse_selection_window_ms: int = 8000
    adverse_mid_drift_threshold_pct: float = 0.02
    adverse_spread_widen_ratio: float = 1.4
    adverse_min_trades: int = 12
    adverse_noise_mad_multiplier: float = 1.25
    adverse_spread_baseline_floor_pct: float = 0.008

    # 3-state Gaussian HMM regime (see regime/hmm_regime.py)
    hmm_n_states: int = 3
    hmm_min_bars: int = 80
    hmm_training_bars: int = 220
    hmm_max_iter: int = 120
    hmm_random_state: int = 42
    hmm_cascade_prob_threshold: float = 0.45

    # Macro BTC–SPX correlation gate (Kelly cap)
    macro_corr_threshold: float = 0.85
    macro_corr_kelly_multiplier: float = 0.50
    macro_corr_lookback_days: int = 60
    macro_corr_min_samples: int = 30

    # Dynamic IC hibernation (Spearman vs forward daily log return)
    ic_hibernation_threshold: float = 0.02
    ic_rolling_days: int = 60
    ic_min_samples: int = 20

    # Background SHAP / cluster snapshot (not on hot path)
    shap_cluster_interval_sec: int = 3600
    shap_cluster_output_path: str = 'btc_intelligence/logs/shap_clusters.json'

    # Probability calibration (Platt + isotonic; optional ElasticNet meta via JSON)
    calibration_min_trades: int = 25
    calibration_train_frac: float = 0.65
    calibration_use_elasticnet_meta: bool = True
    calibration_meta_fit_enabled: bool = True
    elasticnet_calibrator_path: str = 'btc_intelligence/logs/elasticnet_calibrator.json'

    # Brier watchdog → observation / auto-pause (raw_prob vs outcomes, per regime)
    brier_watchdog_threshold: float = 0.25
    brier_watchdog_rolling_trades: int = 80  # last N trades per regime; 0 = full history in that regime

    # Anti-crowding (HHI on aggressive flow): defer SignalEngine emission after crowded tape
    crowd_gate_enabled: bool = True
    crowd_delay_ms: int = 35000
    crowd_hhi_threshold: float = 0.38
    crowd_flow_imbalance_min: float = 0.68
    crowd_score_trigger: float = 72.0  # 0–100 composite (HHI + imbalance)
    crowd_max_trades: int = 500
    crowd_min_trades: int = 25
    crowd_price_tick_bps: float = 2.0

    # Event keywords.
    blocking_news_keywords: list[str] = Field(
        default_factory=lambda: ['fomc', 'cpi', 'pce', 'etf', 'sec', 'fed', 'rate decision']
    )


settings = Settings()
