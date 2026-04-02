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
    redis_key_prefix: str = 'btcintel'

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

    # Event keywords.
    blocking_news_keywords: list[str] = Field(
        default_factory=lambda: ['fomc', 'cpi', 'pce', 'etf', 'sec', 'fed', 'rate decision']
    )


settings = Settings()
