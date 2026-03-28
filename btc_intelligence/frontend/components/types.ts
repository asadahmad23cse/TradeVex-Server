export type SignalPayload = {
  signal: "LONG" | "SHORT" | "HOLD";
  validated: "LONG" | "SHORT" | "HOLD" | "RULE_BASED";
  confidence: number;
  stacked_probability: number;
  alpha_score: number;
  net_alpha: number;
  entry_zone: [number, number];
  stop_loss: number;
  take_profit: { TP1: number; TP2: number; TP3: number };
  risk_reward: number;
  position_size_pct: number;
  position_size_btc: number;
  market_regime: string;
  session: string;
  execution_quality: string;
  mtf: {
    "15m": string;
    "1h": string;
    "4h": string;
    alignment_score: number;
  };
  smc: {
    ob_level: number;
    fvg_present: boolean;
    bos_type: string;
    liq_sweep_recent: boolean;
    choch: boolean;
  };
  order_flow: {
    cvd_slope: string;
    obi: number;
    whale_pressure: string;
    absorption: string;
    stacked_imbalance: string;
    single_bar_divergence: boolean;
    cross_exchange_cvd: string;
    speed_of_tape: string;
  };
  derivatives: {
    funding_rate: number;
    funding_momentum: string;
    oi_change_1h: string;
    liq_magnet: string;
    ls_ratio: string;
    max_pain: number;
    iv_skew: number;
    put_call_ratio: number;
  };
  on_chain: {
    exchange_netflow: string;
    sopr: number;
    lth_behavior: string;
    whale_deposits_1h: number;
    whale_withdrawals_1h: number;
  };
  macro: {
    fear_greed: number;
    dxy_bias: string;
    vix: number;
    vix_regime: string;
    spx_direction: string;
    spx_btc_correlation: number;
    us_10y_trend: string;
    gold_direction: string;
    eth_btc_trend: string;
    btc_dominance_trend: string;
  };
  algo: string;
  strategy: string;
  reason: string;
  factors_present: string[];
  independent_edges: Record<string, number>;
  as_of_utc: string;
  cooldown_until: string;
  next_check_utc: string;
  stale_after_utc: string;
  model_version: string;
  system_status: {
    recent_win_rate: number;
    current_drawdown_pct: number;
    portfolio_heat_pct: number;
    auto_pause: boolean;
  };
};

export const EMPTY_SIGNAL: SignalPayload = {
  signal: "HOLD",
  validated: "HOLD",
  confidence: 0,
  stacked_probability: 0,
  alpha_score: 0,
  net_alpha: 0,
  entry_zone: [0, 0],
  stop_loss: 0,
  take_profit: { TP1: 0, TP2: 0, TP3: 0 },
  risk_reward: 0,
  position_size_pct: 1,
  position_size_btc: 0,
  market_regime: "unknown",
  session: "unknown",
  execution_quality: "poor",
  mtf: { "15m": "mixed", "1h": "mixed", "4h": "mixed", alignment_score: 0 },
  smc: { ob_level: 0, fvg_present: false, bos_type: "none", liq_sweep_recent: false, choch: false },
  order_flow: {
    cvd_slope: "flat",
    obi: 0.5,
    whale_pressure: "neutral",
    absorption: "none",
    stacked_imbalance: "none",
    single_bar_divergence: false,
    cross_exchange_cvd: "neutral",
    speed_of_tape: "normal"
  },
  derivatives: {
    funding_rate: 0,
    funding_momentum: "stable",
    oi_change_1h: "+0.00%",
    liq_magnet: "neutral",
    ls_ratio: "50/50",
    max_pain: 0,
    iv_skew: 0,
    put_call_ratio: 1
  },
  on_chain: {
    exchange_netflow: "neutral",
    sopr: 1,
    lth_behavior: "flat",
    whale_deposits_1h: 0,
    whale_withdrawals_1h: 0
  },
  macro: {
    fear_greed: 50,
    dxy_bias: "neutral",
    vix: 20,
    vix_regime: "NORMAL",
    spx_direction: "flat",
    spx_btc_correlation: 0.5,
    us_10y_trend: "flat",
    gold_direction: "flat",
    eth_btc_trend: "flat",
    btc_dominance_trend: "flat"
  },
  algo: "NO_TRADE",
  strategy: "NONE",
  reason: "Waiting for first qualified signal",
  factors_present: [],
  independent_edges: {},
  as_of_utc: "",
  cooldown_until: "",
  next_check_utc: "",
  stale_after_utc: "",
  model_version: "",
  system_status: {
    recent_win_rate: 0,
    current_drawdown_pct: 0,
    portfolio_heat_pct: 0,
    auto_pause: false
  }
};
