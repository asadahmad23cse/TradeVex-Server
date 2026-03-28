# Architecture (living map)

Use this file so Cursor and Obsidian share the same mental model. Update when layers or entrypoints change.

- **ML / NN diagram + how to use Graph in Obsidian:** [[neural-network-map]]

## Entry

- `main.py` — modes: `live`, `dashboard`, `signals`, `backtest`, `capacity`
- Canonical stack overview: repo `README.md` (15-layer diagram)

## Layers I own / touch most

| Layer | Code area | Notes |
|-------|-----------|-------|
| Data | `src/api/` | Connectors, universe, quality |
| Features | `src/features/` | Engineer, Hurst, etc. |
| Alpha / ML | `src/alpha/`, `src/models/` | Factors, ensemble, LSTM, transformer |
| Risk / portfolio | `src/risk/`, `src/portfolio/` | HRP, Kelly, scenarios |
| Execution | `src/execution/` | Paper, Zerodha, state machine |
| Backtest / research | `src/backtest/`, `src/research/` | WFO, stress, hyperparam |

## Invariants (project-specific)

- Point-in-time data rules:
- Regime / HMM persistence paths:
- Cost model assumptions:

## Open questions

-
