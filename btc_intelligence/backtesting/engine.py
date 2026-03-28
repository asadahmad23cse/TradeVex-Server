from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from btc_intelligence.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from btc_intelligence.features.feature_vector import FEATURE_COLUMNS


def run_walk_forward_backtest(data_csv: str, model_path: str) -> dict:
    df = pd.read_csv(data_csv).dropna(subset=FEATURE_COLUMNS + ['was_profitable'])
    x = df[FEATURE_COLUMNS].astype(float).values
    y = df['was_profitable'].astype(int).values

    model = joblib.load(model_path)
    tscv = TimeSeriesSplit(n_splits=6)

    returns: list[float] = []
    equity = [1.0]
    maes: list[float] = []
    mfes: list[float] = []
    for tr, te in tscv.split(x):
        model.fit(x[tr], y[tr])
        p = model.predict_proba(x[te])[:, 1]
        for prob, label in zip(p, y[te]):
            # Conservative policy: only trade on high conviction.
            if prob < 0.6:
                r = 0.0
                mae = 0.0
                mfe = 0.0
            else:
                r = 0.01 if int(label) == 1 else -0.01
                mae = 0.003 if int(label) == 1 else 0.01
                mfe = 0.01 if int(label) == 1 else 0.002
            returns.append(r)
            equity.append(equity[-1] * (1 + r))
            if r != 0:
                maes.append(mae)
                mfes.append(mfe)

    return {
        'trades': len([r for r in returns if r != 0]),
        'sharpe_ratio': round(sharpe_ratio(returns), 4),
        'max_drawdown_pct': round(max_drawdown(equity), 2),
        'win_rate_pct': round(win_rate([r for r in returns if r != 0]), 2),
        'avg_mae_pct': round(float(np.mean(maes) * 100.0), 3) if maes else 0.0,
        'avg_mfe_pct': round(float(np.mean(mfes) * 100.0), 3) if mfes else 0.0,
        'final_equity': round(equity[-1], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Walk-forward backtester (6m train / 1m test rolling approximation)')
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', default='btc_intelligence/backtesting/report.json')
    args = parser.parse_args()

    metrics = run_walk_forward_backtest(args.data, args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
