import pandas as pd
from typing import Optional

def compute_previous_cycle_ath(price_df: pd.DataFrame, start_date: pd.Timestamp) -> float:
    prior = price_df[price_df['Date'] < start_date]
    if prior.empty:
        raise ValueError("No prior history before START_DATE to compute previous cycle ATH.")
    return prior['Close'].max()

def compute_objective_end_date(price_df: pd.DataFrame,
                               start_date: pd.Timestamp,
                               weekly_sma_window: int = 150,
                               weekly_rule: str = 'W-SUN') -> pd.Timestamp:
    df = price_df.copy().sort_values('Date').reset_index(drop=True)
    prev_ath = compute_previous_cycle_ath(df, start_date)

    wk = df.set_index('Date')['Close'].resample(weekly_rule).last().dropna()
    wk_sma = wk.rolling(weekly_sma_window, min_periods=weekly_sma_window).mean()

    crossing = wk_sma[wk_sma >= prev_ath]
    if crossing.empty:
        end_date = df['Date'].iloc[-1]
    else:
        crossing_week = crossing.index[0]
        daily_after = df[df['Date'] >= crossing_week]
        end_date = daily_after['Date'].iloc[0] if not daily_after.empty else df['Date'].iloc[-1]

    print(f"Objective end-date rule: prev cycle ATH={prev_ath:,.2f} USD; end_date={end_date.date()}")
    return end_date

# Halving dates (UTC close-of-day approximations). Adjust if you prefer exchange timestamps.
HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),  # 2024 halving (approx)
]

def next_halving_date(start_date: pd.Timestamp, halving_dates: list[pd.Timestamp]) -> Optional[pd.Timestamp]:
    for d in sorted(halving_dates):
        if d > start_date:
            return d
    return None