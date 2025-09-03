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
    - a dict of metrics, or
    - a daily performance DataFrame returned by run_backtest (multi-row, with Equity_USD),
      or a one-row DataFrame of precomputed metrics.

    Returns a plain dict of metrics.
    """
    # Pass-through if already a metrics dict
    if isinstance(perf, dict):
        return perf

    if not isinstance(perf, pd.DataFrame):
        raise TypeError("perf must be a dict or a pandas DataFrame")

    if perf.empty:
        return {}

    # Case A: daily perf DataFrame (from run_backtest). Compute metrics here.
    if 'Equity_USD' in perf.columns:
        df = perf.copy()
        if 'Date' in df.columns:
            # ensure proper temporal ordering
            df = df.sort_values('Date')

        eq = pd.to_numeric(df['Equity_USD'], errors='coerce').dropna()
        if eq.empty:
            return {}

        # Determine elapsed days using Date if available, else row count
        if 'Date' in df.columns:
            start_date = pd.to_datetime(df['Date'].iloc[0])
            end_date = pd.to_datetime(df['Date'].iloc[-1])
            days = max((end_date - start_date).days, len(eq) - 1)
        else:
            days = max(len(eq) - 1, 1)

        start = float(eq.iloc[0])
        end = float(eq.iloc[-1])

        # Returns stream
        r = eq.pct_change().dropna()

        # Total return and CAGR
        total_ret = (end / start - 1.0) if start > 0 else np.nan
        years = max(days / 365.0, 1e-9)
        cagr = (end / start) ** (1.0 / years) - 1.0 if (start > 0 and end > 0) else np.nan

        # Risk metrics
        sharpe_lo = adjusted_sharpe_lo(r, periods_per_year=365)
        sharpe = sharpe_ratio(r, periods_per_year=365)
        sortino = sortino_ratio(r, periods_per_year=365)
        mdd, dd_days = compute_drawdowns(eq)
        var95, es95 = var_es(r, alpha=0.95)

        out = {
            'Final Equity (USD)': end,
            'Total Return (%)': total_ret * 100.0 if not np.isnan(total_ret) else np.nan,
            'CAGR (%)': cagr * 100.0 if not np.isnan(cagr) else np.nan,
            'Sharpe (Lo-Adj)': sharpe_lo if not np.isnan(sharpe_lo) else np.nan,
            'Sharpe': sharpe if not np.isnan(sharpe) else np.nan,
            'Sortino': sortino if not np.isnan(sortino) else np.nan,
            'Max Drawdown (%)': mdd if not np.isnan(mdd) else np.nan,
            'Longest DD (days)': int(dd_days),
            'VaR 95% (%)': var95 if not np.isnan(var95) else np.nan,
            'ES 95% (%)': es95 if not np.isnan(es95) else np.nan,
        }
        return out

    # Case B: one-row DataFrame of precomputed metrics -> sanitize and return
    if len(perf) == 1:
        row = perf.iloc[0].to_dict()
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        return clean

    # If it's a multi-row DataFrame without Equity_USD, we don't know how to summarize it
    raise TypeError("DataFrame must be daily perf (with Equity_USD) or a one-row metrics table")