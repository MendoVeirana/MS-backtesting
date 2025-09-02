from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import math

def compute_drawdowns(series: pd.Series) -> Tuple[float, int]:
    """
    Max drawdown (in %) and longest drawdown duration (days) for an equity series in USD.
    """
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    max_dd = dd.min() * 100.0
    # Duration: longest consecutive days below prior peak
    below = series < cummax
    longest = 0
    cur = 0
    for b in below:
        if b:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return max_dd, longest


def sortino_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    downside = r[r < 0]
    denom = downside.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return (r.mean() / denom) * math.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    r = returns.dropna()
    if r.std(ddof=1) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=1)) * math.sqrt(periods_per_year)


def lo_variance_inflation(returns: pd.Series, max_lag: Optional[int] = None) -> float:
    """
    Lo (2002) variance inflation factor for Sharpe when returns are autocorrelated.
    VIF = 1 + 2 * sum_{k=1}^K rho_k, choose K ~ T^(1/3) (capped).
    """
    r = returns.dropna()
    T = len(r)
    if T < 3:
        return 1.0
    if max_lag is None:
        max_lag = min(int(round(T ** (1/3))), 10)
    vif = 1.0
    r_mean = r.mean()
    denom = ((r - r_mean)**2).sum()
    if denom == 0:
        return 1.0
    for k in range(1, max_lag + 1):
        num = ((r[k:] - r_mean) * (r.shift(k)[k:] - r_mean)).sum()
        rho_k = num / denom
        vif += 2.0 * rho_k
    return max(vif, 1.0)


def adjusted_sharpe_lo(returns: pd.Series, periods_per_year: int = 365) -> float:
    """
    Lo-adjusted Sharpe ratio to account for autocorrelation.
    """
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)
    if np.isnan(sr):
        return sr
    vif = lo_variance_inflation(returns)
    if vif <= 0:
        return sr
    return sr / math.sqrt(vif)


def var_es(returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Empirical one-day VaR/ES at alpha (returns as decimals). Negative numbers indicate loss.
    Returns (VaR_pct, ES_pct) in percentage terms.
    """
    r = returns.dropna().sort_values()
    if r.empty:
        return np.nan, np.nan
    cutoff = 1 - alpha
    var = r.quantile(cutoff)
    es = r[r <= var].mean() if not r[r <= var].empty else var
    return var * 100.0, es * 100.0


def summarize_performance(perf: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Accepts either:
      - a dict of metrics (as returned by run_backtest['perf']), or
      - a one-row DataFrame of metrics.
    Returns a plain dict of metrics.
    """
    if isinstance(perf, dict):
        return perf

    if isinstance(perf, pd.DataFrame):
        if perf.empty:
            return {}
        row = perf.iloc[0].to_dict()
        # sanitize NaN/Inf → None for clean JSON/printing
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        return clean

    raise TypeError("perf must be a dict or a pandas DataFrame")