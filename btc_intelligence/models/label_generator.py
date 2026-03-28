from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def generate_labels(df: pd.DataFrame, horizon: int = 48) -> pd.DataFrame:
    work = df.copy()
    if 'direction' not in work.columns:
        work['direction'] = np.where(work.get('ema9_dist', 0) >= 0, 'LONG', 'SHORT')

    was_profitable: list[int] = []
    for i in range(len(work)):
        if i + 1 >= len(work):
            was_profitable.append(0)
            continue

        row = work.iloc[i]
        entry = float(row.get('close', row.get('Close', 0.0)))
        atr = float(row.get('atr14', max(entry * 0.005, 1.0)))
        if entry <= 0 or atr <= 0:
            was_profitable.append(0)
            continue

        direction = str(row.get('direction', 'LONG')).upper()
        sl = entry - 1.5 * atr if direction == 'LONG' else entry + 1.5 * atr
        tp1 = entry + 1.5 * atr if direction == 'LONG' else entry - 1.5 * atr

        future = work.iloc[i + 1 : i + 1 + horizon]
        hit = 0
        for _, f in future.iterrows():
            hi = float(f.get('high', f.get('High', entry)))
            lo = float(f.get('low', f.get('Low', entry)))
            if direction == 'LONG':
                if lo <= sl:
                    hit = 0
                    break
                if hi >= tp1:
                    hit = 1
                    break
            else:
                if hi >= sl:
                    hit = 0
                    break
                if lo <= tp1:
                    hit = 1
                    break
        was_profitable.append(hit)

    work['was_profitable'] = was_profitable
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate supervised labels for BTC models')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--horizon', type=int, default=48)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = generate_labels(df, horizon=args.horizon)
    out.to_csv(args.output, index=False)
    print(f'Wrote labeled dataset: {args.output} rows={len(out)}')


if __name__ == '__main__':
    main()
