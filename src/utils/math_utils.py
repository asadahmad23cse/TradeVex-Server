import numpy as np
import pandas as pd


def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    """Annualised Sharpe Ratio."""
    mean_return = np.mean(returns)
    periodic_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    return (mean_return - periodic_rf) / std_return * np.sqrt(252)


def calculate_sortino_ratio(returns, target_return=0, risk_free_rate=0.05):
    """Annualised Sortino Ratio — penalises only downside volatility."""
    mean_return = np.mean(returns)
    periodic_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    downside = returns[returns < target_return]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-9
    if downside_std == 0:
        return 0.0
    return (mean_return - periodic_rf) / downside_std * np.sqrt(252)


def calculate_max_drawdown(prices):
    """Maximum drawdown from a price series (returns negative fraction)."""
    prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min())


def calculate_calmar_ratio(returns, risk_free_rate=0.05):
    """
    Calmar Ratio = CAGR / |Max Drawdown|.
    Uses equity curve built from returns.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    equity = (1 + returns).cumprod()
    n_years = len(returns) / 252
    cagr = float(equity.iloc[-1]) ** (1 / max(n_years, 1e-9)) - 1
    mdd = abs(calculate_max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return cagr / mdd


def calculate_var(returns, confidence=0.95):
    """
    Historical Value at Risk at given confidence level.
    Returns the loss (positive number) exceeded only (1-confidence) of the time.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns, confidence=0.95):
    """
    Conditional Value at Risk (Expected Shortfall).
    Mean loss in the worst (1-confidence) fraction of days.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    var = -np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return float(var)
    return float(-tail.mean())


def calculate_information_ratio(factor_scores, forward_returns, window=60):
    """
    Information Ratio = mean(IC) / std(IC) over rolling window.
    IC = Pearson correlation(factor_score_t, forward_return_t+1).

    Parameters
    ----------
    factor_scores : pd.Series
    forward_returns : pd.Series  (shifted -1 returns aligned with factor_scores)
    window : int

    Returns
    -------
    float  — Information Ratio (annualised by * sqrt(252))
    """
    factor_scores = pd.Series(factor_scores)
    forward_returns = pd.Series(forward_returns)
    combined = pd.concat([factor_scores, forward_returns], axis=1).dropna()
    if len(combined) < window:
        return 0.0
    combined.columns = ['factor', 'fwd_ret']

    rolling_ic = combined['factor'].rolling(window).corr(combined['fwd_ret'])
    ic_series = rolling_ic.dropna()
    if len(ic_series) < 2 or ic_series.std() == 0:
        return 0.0
    return float(ic_series.mean() / ic_series.std() * np.sqrt(252))


def bootstrap_confidence_interval(values, n_bootstrap=1000, confidence=0.95, statistic=np.mean):
    """Bootstrap confidence interval for a statistic."""
    values = pd.Series(values).dropna().values
    if len(values) < 2:
        center = float(values[0]) if len(values) == 1 else 0.0
        return center, center, center
    rng = np.random.default_rng(42)
    samples = np.array(
        [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1 - alpha))
    center = float(statistic(values))
    return center, lower, upper


def calculate_profit_factor(returns):
    """Gross profit / gross loss. > 1.5 is good."""
    returns = pd.Series(returns).dropna()
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def calculate_annualized_return(returns):
    """CAGR from a daily returns series."""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    equity = (1 + returns).cumprod()
    n_years = len(returns) / 252
    return float(equity.iloc[-1]) ** (1 / max(n_years, 1e-9)) - 1


def calculate_jensens_alpha(portfolio_returns, market_returns, risk_free_rate=0.05):
    """Jensen's Alpha = Rp - [Rf + Beta * (Rm - Rf)], annualised."""
    periodic_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_pf = np.asarray(portfolio_returns) - periodic_rf
    excess_mkt = np.asarray(market_returns) - periodic_rf
    covariance = np.cov(excess_pf, excess_mkt)[0, 1]
    variance_mkt = np.var(excess_mkt)
    beta = covariance / variance_mkt if variance_mkt != 0 else 1.0
    alpha = np.mean(excess_pf) - beta * np.mean(excess_mkt)
    return float(alpha * 252)
