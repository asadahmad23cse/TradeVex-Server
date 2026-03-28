# Focus Quant Research Notes
Date: 2026-03-27

This note records the recent quant research references used to guide the Gold/Silver/Bitcoin focus upgrade.

## Recent papers reviewed

1. WaveLSFormer (arXiv:2601.13435, latest revision March 12, 2026)  
   Link: https://arxiv.org/abs/2601.13435  
   Key point: multi-scale decomposition with risk-aware objective improves risk-adjusted long/short decisions.

2. IVE: Probabilistic Intraday Volume Ratio Forecasting with Transformers (arXiv:2411.10956, revised March 9, 2025)  
   Link: https://arxiv.org/abs/2411.10956  
   Key point: probabilistic intraday modeling and volume-aware execution timing are useful for live trade quality.

3. HRFT: Intraday Risk Factor Transformer (arXiv:2408.01271, latest revision September 20, 2025)  
   Link: https://arxiv.org/abs/2408.01271  
   Key point: explicit interpretable factor construction should remain visible in production outputs.

4. A limit order book model for high frequency trading with rough volatility (Numerical Algebra, Control and Optimization, published online April 21, 2025)  
   Link: https://www.aimsciences.org/article/doi/10.3934/naco.2025010  
   Key point: rough-volatility and order-flow microstructure effects justify strict cost and data-quality gates.

## How this was applied in the codebase

1. Multi-horizon signal agreement gate was added in `FocusQuantEngine` to require cross-interval confirmation before a trade is marked validated.
2. Validation score and explicit pass/fail checks were added to the API payload to expose model reliability, not only directional signal output.
3. Net-alpha cost gating was added to the focus trade path so trade viability is checked after estimated friction.
4. Factor scores and IC weights are returned with every focus trade to keep the signal interpretable and auditable.

## Practical scope

These changes are implementation-grade adaptations inspired by the above research and are not a direct reproduction of any single paper.
