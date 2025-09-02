from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from config import StrategyConfig
from mechanics import (
    usd_notional_from_btc_margin,
    commission_btc_from_usd,
    funding_btc_from_usd,
    avg_entry_inverse_harmonic,
    liquidation_price_inverse,
    apply_slippage,
)
from buying_strategy import ladder

def run_backtest(price_df: pd.DataFrame,
                 cfg: StrategyConfig,
                 funding_daily_series: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    price_df columns: Date, Open, High, Low, Close (daily)
    Supports two contract models:
      - 'inverse' (default): coin-margined BTCUSD-like perp with inverse PnL in BTC

    funding_daily_series: optional pd.Series aligned to price_df with daily funding rate
                          (e.g., 0.0008 = 8 bps/day)

    Returns dict with:
      - perf: DataFrame of daily performance (includes diagnostics)
      - cycles: list of cycle dicts (includes start/last liq)
      - status: dict summary
    """
    model = getattr(cfg, 'contract_model', 'inverse')  # default to inverse
    df = price_df.copy().reset_index(drop=True)

    if funding_daily_series is None:
        df['FundingRateDaily'] = cfg.constant_funding_daily
    else:
        df['FundingRateDaily'] = funding_daily_series.values

    # Wallet state (BTC)
    free_margin = 0.0
    spot_btc = 0.0
    pos_margin_btc = 0.0

    # Position state
    avg_entry = None
    in_cycle = False
    liquidated = False
    cycle_id = 0

    # Inverse path (USD contracts)
    pos_q_usd = 0.0

    # Records
    daily: List[Dict[str, Any]] = []
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    cycles: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        date = row['Date']
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        fund_rate = row['FundingRateDaily']

        # 1) Start a new cycle (seed wallet and open)
        if not in_cycle and not liquidated:
            cycle_id += 1

            # Seed this cycle with the initial margin
            free_margin = cfg.initial_margin_btc
            pos_margin_btc = 0.0
            avg_entry = None
            pos_q_usd = 0.0

            # Deterministic open at Close with slippage/commission
            fill_price = apply_slippage(c, cfg.slippage_bps, buy=True)
            open_margin = min(cfg.open_margin_pct * cfg.initial_margin_btc, free_margin)

            q_add = usd_notional_from_btc_margin(open_margin, fill_price, cfg.leverage_limit)
            commission_btc = commission_btc_from_usd(q_add, fill_price, cfg.commission_rate)

            pos_q_usd = q_add
            pos_margin_btc += open_margin
            avg_entry = fill_price

            free_margin -= (open_margin + commission_btc)
            free_margin = max(0.0, free_margin)

            # Fix: For isolated margin, use ONLY pos_margin_btc as W (exclude free/spot)
            isolated_wallet_btc = pos_margin_btc  # Simulates isolated: only allocated margin backs liq
            start_liq_price = liquidation_price_inverse(
                entry_price=avg_entry,
                q_usd=pos_q_usd,
                wallet_btc_excl_unreal=isolated_wallet_btc,  # Key change: isolated W
                maintenance_margin_rate=cfg.maintenance_margin_rate
            )
            print('start_liq_price', start_liq_price)  # Now should be higher, closer to entry - entry/lev

            in_cycle = True
            starts.append(date)
            cycles.append({
                'cycle': cycle_id,
                'start_date': date,
                'start_price': fill_price,
                'start_free_margin_btc': free_margin,
                'start_spot_btc': spot_btc,
                'start_pos_margin_btc': pos_margin_btc,
                'start_pos_q_usd': pos_q_usd,
                'start_equity_btc_excl_unreal': free_margin + spot_btc + pos_margin_btc,  # Full for equity, but liq uses isolated
                'start_liq_price': start_liq_price,
                'last_liq_price': start_liq_price
            })

        # 2) Funding (debited from pos_margin for isolated)
        if in_cycle:
            if model == 'inverse' and pos_q_usd > 0:
                fee_btc = funding_btc_from_usd(pos_q_usd, c, fund_rate)
                pos_margin_btc -= fee_btc  # Deduct directly from pos (isolated)
                pos_margin_btc = max(0.0, pos_margin_btc)  # Prevent negative
                
                if free_margin >= fee_btc:
                    free_margin -= fee_btc
                else:
                    # Fix: Deduct remainder from pos_margin to avoid negative free/W reduction
                    remainder = fee_btc - free_margin
                    free_margin = 0.0
                    pos_margin_btc -= remainder
                    # Optional: Check if pos_margin_btc <0, trigger liq -- but for now, allow

        # 3) Pre-trade liquidation (after funding, before TP/ladder)
        liq_price_pre = 0.0
        liq_denom_pre_btc = np.nan
        mm_btc_mark = 0.0
        if in_cycle and avg_entry is not None and model == 'inverse' and pos_q_usd > 0:
            # Fix: Isolated - use only pos_margin_btc
            isolated_wallet_btc = pos_margin_btc
            liq_denom_pre_btc = isolated_wallet_btc + (pos_q_usd / avg_entry)
            liq_price_pre = liquidation_price_inverse(
                entry_price=avg_entry,
                q_usd=pos_q_usd,
                wallet_btc_excl_unreal=isolated_wallet_btc,
                maintenance_margin_rate=cfg.maintenance_margin_rate
            )
            # Diagnostic maintenance margin at mark
            mm_btc_mark = cfg.maintenance_margin_rate * pos_q_usd / c

        # 4) Liquidation check (intraday) using PRE-trade liq
        if (in_cycle and isinstance(liq_price_pre, (int, float))
                and not np.isnan(liq_price_pre) and liq_price_pre > 0 and l <= liq_price_pre):
            ends.append(date)
            cycles[-1].update({
                'end_date': date,
                'end_price': c,
                'end_free_margin_btc': 0.0,
                'end_spot_btc': 0.0,
                'end_pos_margin_btc': 0.0,
                'end_pos_q_usd': 0.0,
                'end_equity_btc_excl_unreal': 0.0,
                'reason': 'LIQUIDATION',
                'duration_days': (date - cycles[-1]['start_date']).days
            })
            # Reset
            free_margin = 0.0
            spot_btc = 0.0
            pos_margin_btc = 0.0
            pos_q_usd = 0.0
            avg_entry = None
            in_cycle = False
            liquidated = True

            if cfg.stop_on_liquidation:
                break
            else:
                # No position left today; record a flat day
                daily.append({
                    'Date': date,
                    'Close': c,
                    'Equity_USD': 0.0,
                    'Free_Margin_BTC': 0.0,
                    'Spot_BTC': 0.0,
                    'Pos_Margin_BTC': 0.0,
                    'Pos_Q_USD': 0.0,
                    'Base_Equity_BTC': 0.0,
                    'Unreal_PnL_BTC': 0.0,
                    'Unreal_PnL_USD': 0.0,
                    'Avg_Entry': 0.0,
                    'Liq_Price': 0.0,
                    'Liq_Price_Pre': 0.0,
                    'Liq_Price_Post': 0.0,
                    # Diagnostics:
                    'Wallet_BTC_Excl_Unreal': 0.0,
                    'Q_over_Entry_BTC': 0.0,
                    'Liq_Denom_BTC_Pre': np.nan,
                    'Liq_Denom_BTC_Post': np.nan,
                    'MM_BTC_at_Mark': 0.0,
                    'Cycle': cycle_id
                })
                continue

        if not in_cycle:
            # Nothing to do
            continue

        # 5) Take profit (pre-ladder)
        if c >= (avg_entry * (1.0 + cfg.take_profit_pct)):
            exit_price = apply_slippage(c, cfg.slippage_bps, buy=False)
            if model == 'inverse' and pos_q_usd > 0:
                realized_pnl_btc = pos_q_usd * (1.0 / avg_entry - 1.0 / exit_price)
                close_commission_btc = commission_btc_from_usd(pos_q_usd, exit_price, cfg.commission_rate)
            else:
                realized_pnl_btc = 0.0
                close_commission_btc = 0.0

            free_margin += pos_margin_btc + realized_pnl_btc
            free_margin -= close_commission_btc

            if free_margin > cfg.initial_margin_btc:
                sweep = free_margin - cfg.initial_margin_btc
                spot_btc += sweep
                free_margin = cfg.initial_margin_btc

            ends.append(date)
            cycles[-1].update({
                'end_date': date,
                'end_price': exit_price,
                'end_free_margin_btc': free_margin,
                'end_spot_btc': spot_btc,
                'end_pos_margin_btc': 0.0,
                'end_pos_q_usd': 0.0,
                'end_equity_btc_excl_unreal': free_margin + spot_btc,
                'reason': 'TAKE-PROFIT',
                'duration_days': (date - cycles[-1]['start_date']).days
            })

            # Reset position
            pos_margin_btc = 0.0
            pos_q_usd = 0.0
            avg_entry = None
            in_cycle = False

            # Record flat day after TP
            base_equity_btc = free_margin + spot_btc  # no position margin
            equity_usd = base_equity_btc * c
            daily.append({
                'Date': date,
                'Close': c,
                'Equity_USD': equity_usd,
                'Free_Margin_BTC': free_margin,
                'Spot_BTC': spot_btc,
                'Pos_Margin_BTC': 0.0,
                'Pos_Q_USD': 0.0,
                'Base_Equity_BTC': base_equity_btc,
                'Unreal_PnL_BTC': 0.0,
                'Unreal_PnL_USD': 0.0,
                'Avg_Entry': 0.0,
                'Liq_Price': 0.0,
                'Liq_Price_Pre': liq_price_pre,
                'Liq_Price_Post': 0.0,
                # Diagnostics:
                'Wallet_BTC_Excl_Unreal': free_margin + spot_btc,
                'Q_over_Entry_BTC': 0.0,
                'Liq_Denom_BTC_Pre': liq_denom_pre_btc,
                'Liq_Denom_BTC_Post': np.nan,
                'MM_BTC_at_Mark': 0.0,
                'Cycle': cycle_id
            })
            continue

        # 7) Ladder (scale-in + optional extra injection)
        delta = (c / avg_entry) - 1.0 if (avg_entry and avg_entry > 0) else 0.0
        add_margin_btc, extra_injection_btc = ladder(delta, cfg.initial_margin_btc)
        # Extra injection is new capital (external): add directly to pos margin
        if extra_injection_btc > 0:
            pos_margin_btc += extra_injection_btc
            # Do NOT subtract from free_margin here -- leave it for potential adds/fees
            # (Fix: Previously, this left free=0, blocking adds)
        # Now handle add (internal transfer + buy more Q)
        if model == 'inverse':
            # No q_cap to allow DCA even if over initial lev limit (fix: prevents blocking adds on drops)
            # Exposure cap in USD: Q_cap = (wallet_btc * leverage) * c  # Commented out for unlimited DCA
            if add_margin_btc > 0:
                fill = apply_slippage(c, cfg.slippage_bps, buy=True)
                # Planned add (no cap)
                q_usd_planned = usd_notional_from_btc_margin(add_margin_btc, fill, cfg.leverage_limit)
                q_usd_allowable = q_usd_planned  # No min with cap -- allow full planned
                if q_usd_allowable > 0:
                    # Margin required for the allowable notional
                    margin_needed = q_usd_allowable / (cfg.leverage_limit * fill)
                    # Commission for that notional
                    fee_btc = commission_btc_from_usd(q_usd_allowable, fill, cfg.commission_rate)
                    # Total needed (but treat as injection if free insufficient)
                    total_btc_needed = margin_needed + fee_btc
                    # Fix: If free insufficient, treat add_margin as new injection (don't scale to 0)
                    if total_btc_needed > free_margin:
                        # Inject the shortfall as new capital (simulate external add)
                        shortfall = total_btc_needed - max(0, free_margin)
                        free_margin += shortfall  # Add new capital to free
                    # Now execute (we have enough)
                    free_margin -= total_btc_needed
                    pos_margin_btc += margin_needed
                    # Update position notional and average entry (inverse harmonic)
                    avg_entry = avg_entry_inverse_harmonic(pos_q_usd, avg_entry or fill,
                                                        q_usd_allowable, fill)
                    pos_q_usd += q_usd_allowable

        # 7) Post-trade liquidation (after ladder/avg_entry updates)
        liq_price_post = 0.0
        liq_denom_post_btc = np.nan
        if in_cycle and avg_entry is not None and model == 'inverse' and pos_q_usd > 0:
            wallet_btc_excl_unreal = free_margin + spot_btc + pos_margin_btc
            liq_denom_post_btc = wallet_btc_excl_unreal + (pos_q_usd / avg_entry)
            liq_price_post = liquidation_price_inverse(
                entry_price=avg_entry,
                q_usd=pos_q_usd,
                wallet_btc_excl_unreal=wallet_btc_excl_unreal,
                maintenance_margin_rate=cfg.maintenance_margin_rate
            )
            cycles[-1]['last_liq_price'] = liq_price_post

        # 8) Equity and PnL (end-of-day, post-trade state)
        base_equity_btc = free_margin + spot_btc + pos_margin_btc
        if in_cycle and avg_entry is not None and model == 'inverse' and pos_q_usd > 0:
            unreal_pnl_btc = pos_q_usd * (1.0 / avg_entry - 1.0 / c)
        else:
            unreal_pnl_btc = 0.0
        unreal_pnl_usd = unreal_pnl_btc * c
        equity_usd = base_equity_btc * c + unreal_pnl_usd

        # Diagnostics
        wallet_btc_excl_unreal = free_margin + spot_btc + pos_margin_btc
        q_over_entry = (pos_q_usd / avg_entry) if (model == 'inverse' and pos_q_usd > 0 and avg_entry) else 0.0

        # Record end-of-day row (Liq_Price == post-trade)
        daily.append({
            'Date': date,
            'Close': c,
            'Equity_USD': equity_usd,
            'Free_Margin_BTC': free_margin,
            'Spot_BTC': spot_btc,
            'Pos_Margin_BTC': pos_margin_btc,
            'Pos_Q_USD': pos_q_usd if model == 'inverse' else 0.0,
            'Base_Equity_BTC': base_equity_btc,
            'Unreal_PnL_BTC': unreal_pnl_btc,
            'Unreal_PnL_USD': unreal_pnl_usd,
            'Avg_Entry': avg_entry or 0.0,
            'Liq_Price': liq_price_post if in_cycle else 0.0,
            'Liq_Price_Pre': liq_price_pre if in_cycle else 0.0,
            'Liq_Price_Post': liq_price_post if in_cycle else 0.0,
            # Diagnostics:
            'Wallet_BTC_Excl_Unreal': wallet_btc_excl_unreal,
            'Q_over_Entry_BTC': q_over_entry,  # equals Q/E (BTC units)
            'Liq_Denom_BTC_Pre': liq_denom_pre_btc,
            'Liq_Denom_BTC_Post': liq_denom_post_btc,
            'MM_BTC_at_Mark': mm_btc_mark,
            'Cycle': cycle_id if in_cycle else 0
        })

        # 9) Optional forced close on last day (post-trade)
        is_last_day = (i == len(df) - 1)
        if cfg.forced_close_on_last_day and is_last_day:
            exit_price = apply_slippage(c, cfg.slippage_bps, buy=False)
            if model == 'inverse' and pos_q_usd > 0:
                realized_pnl_btc = pos_q_usd * (1.0 / avg_entry - 1.0 / exit_price)
                close_commission_btc = commission_btc_from_usd(pos_q_usd, exit_price, cfg.commission_rate)
            else:
                realized_pnl_btc = 0.0
                close_commission_btc = 0.0

            free_margin += pos_margin_btc + realized_pnl_btc
            free_margin -= close_commission_btc

            if free_margin > cfg.initial_margin_btc:
                sweep = free_margin - cfg.initial_margin_btc
                spot_btc += sweep
                free_margin = cfg.initial_margin_btc

            ends.append(date)
            cycles[-1].update({
                'end_date': date,
                'end_price': exit_price,
                'end_free_margin_btc': free_margin,
                'end_spot_btc': spot_btc,
                'end_pos_margin_btc': 0.0,
                'end_pos_q_usd': 0.0,
                'end_equity_btc_excl_unreal': free_margin + spot_btc,
                'reason': 'FORCED-CLOSE (END DATE)',
                'duration_days': (date - cycles[-1]['start_date']).days
            })

            # Reset position
            pos_margin_btc = 0.0
            pos_q_usd = 0.0
            avg_entry = None
            in_cycle = False

    perf = pd.DataFrame(daily)
    status = {
        'liquidated': liquidated,
        'cycles_completed': sum(1 for c in cycles if 'end_date' in c),
        'total_cycles': len(cycles),
        'starts': starts,
        'ends': ends
    }
    return {'perf': perf, 'cycles': cycles, 'status': status}