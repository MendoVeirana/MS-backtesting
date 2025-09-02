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

    funding_daily_series: optional pd.Series aligned to price_df with daily funding rate (e.g., 0.0008 = 8 bps/day)

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

    # Linear path (BTC size)
    pos_size_btc = 0.0
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
            pos_size_btc = 0.0
            pos_q_usd = 0.0

            # Deterministic open at Close with slippage/commission
            fill_price = apply_slippage(c, cfg.slippage_bps, buy=True)
            open_margin = min(cfg.open_margin_pct * cfg.initial_margin_btc, free_margin)

            if model == 'inverse':
                q_add = usd_notional_from_btc_margin(open_margin, fill_price, cfg.leverage_limit)
                commission_btc = commission_btc_from_usd(q_add, fill_price, cfg.commission_rate)

                pos_q_usd = q_add
                pos_margin_btc += open_margin
                avg_entry = fill_price

                free_margin -= (open_margin + commission_btc)
                free_margin = max(0.0, free_margin)

                wallet_btc_excl_unreal = free_margin + spot_btc + pos_margin_btc
                start_liq_price = liquidation_price_inverse(
                    entry_price=avg_entry,
                    q_usd=pos_q_usd,
                    wallet_btc_excl_unreal=wallet_btc_excl_unreal,
                    maintenance_margin_rate=cfg.maintenance_margin_rate
                )

            in_cycle = True
            starts.append(date)
            cycles.append({
                'cycle': cycle_id,
                'start_date': date,
                'start_price': fill_price,
                'start_free_margin_btc': free_margin,
                'start_spot_btc': spot_btc,
                'start_pos_margin_btc': pos_margin_btc,
                'start_pos_size_btc': pos_size_btc,
                'start_pos_q_usd': pos_q_usd,
                'start_equity_btc_excl_unreal': free_margin + spot_btc + pos_margin_btc,
                'start_liq_price': start_liq_price,
                'last_liq_price': start_liq_price
            })

        # 2) Funding (debited from free margin)
        if in_cycle:
            if model == 'inverse' and pos_q_usd > 0:
                fee_btc = funding_btc_from_usd(pos_q_usd, c, fund_rate)
                free_margin -= fee_btc

        # 3) Update liquidation price (+ diagnostics)
        liq_price = 0.0
        liq_denom_btc = np.nan  # diagnostic denominator used in liq formula
        mm_btc_mark = 0.0       # maintenance margin at mark (diagnostic)

        if in_cycle and avg_entry is not None:
            wallet_btc_excl_unreal = free_margin + spot_btc + pos_margin_btc
            if model == 'inverse' and pos_q_usd > 0:
                # Denominator used in p_liq: Wallet + Q/E
                liq_denom_btc = wallet_btc_excl_unreal + (pos_q_usd / avg_entry)
                liq_price = liquidation_price_inverse(
                    entry_price=avg_entry,
                    q_usd=pos_q_usd,
                    wallet_btc_excl_unreal=wallet_btc_excl_unreal,
                    maintenance_margin_rate=cfg.maintenance_margin_rate
                )
                # Maintenance margin at mark (not used in liq calc, just diagnostic)
                mm_btc_mark = cfg.maintenance_margin_rate * pos_q_usd / c

            cycles[-1]['last_liq_price'] = liq_price

        # 4) Equity and PnL
        base_equity_btc = free_margin + spot_btc + pos_margin_btc
        if in_cycle and avg_entry is not None:
            if model == 'inverse' and pos_q_usd > 0:
                unreal_pnl_btc = pos_q_usd * (1.0 / avg_entry - 1.0 / c)
            else:
                unreal_pnl_btc = 0.0
        else:
            unreal_pnl_btc = 0.0

        unreal_pnl_usd = unreal_pnl_btc * c
        equity_usd = base_equity_btc * c + unreal_pnl_usd

        # Diagnostics to help verify liq dynamics
        wallet_btc_excl_unreal = free_margin + spot_btc + pos_margin_btc
        q_over_entry = (pos_q_usd / avg_entry) if (model == 'inverse' and pos_q_usd > 0 and avg_entry) else 0.0

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
            'Liq_Price': liq_price if in_cycle else 0.0,
            # Diagnostics:
            'Wallet_BTC_Excl_Unreal': wallet_btc_excl_unreal,
            'Q_over_Entry_BTC': q_over_entry,     # equals Q/E (BTC units)
            'Liq_Denom_BTC': liq_denom_btc,       # denominator used in p_liq (inverse) or similar (linear)
            'MM_BTC_at_Mark': mm_btc_mark,        # maintenance margin at current price (diagnostic)
            'Cycle': cycle_id if in_cycle else 0
        })

        # 5) Liquidation check (intraday)
        if in_cycle and isinstance(liq_price, (int, float)) and not np.isnan(liq_price) and liq_price > 0 and l <= liq_price:
            ends.append(date)
            cycles[-1].update({
                'end_date': date,
                'end_price': c,
                'end_free_margin_btc': 0.0,
                'end_spot_btc': 0.0,
                'end_pos_margin_btc': 0.0,
                'end_pos_size_btc': 0.0,
                'end_pos_q_usd': 0.0,
                'end_equity_btc_excl_unreal': 0.0,
                'reason': 'LIQUIDATION',
                'duration_days': (date - cycles[-1]['start_date']).days
            })

            # Reset
            free_margin = 0.0
            spot_btc = 0.0
            pos_margin_btc = 0.0
            pos_size_btc = 0.0
            pos_q_usd = 0.0
            avg_entry = None
            in_cycle = False
            liquidated = True

            if cfg.stop_on_liquidation:
                break
            else:
                continue

        if not in_cycle:
            continue

        # 6) Take profit
        if c >= (avg_entry * (1.0 + cfg.take_profit_pct)):
            exit_price = apply_slippage(c, cfg.slippage_bps, buy=False)
            if model == 'inverse' and pos_q_usd > 0:
                realized_pnl_btc = pos_q_usd * (1.0 / avg_entry - 1.0 / exit_price)
                close_commission_btc = commission_btc_from_usd(pos_q_usd, exit_price, cfg.commission_rate)

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
                'end_pos_size_btc': 0.0,
                'end_pos_q_usd': 0.0,
                'end_equity_btc_excl_unreal': free_margin + spot_btc,
                'reason': 'TAKE-PROFIT',
                'duration_days': (date - cycles[-1]['start_date']).days
            })

            pos_margin_btc = 0.0
            pos_size_btc = 0.0
            pos_q_usd = 0.0
            avg_entry = None
            in_cycle = False
            continue

        # 7) Ladder (scale-in + optional extra injection)
        delta = (c / avg_entry) - 1.0 if (avg_entry and avg_entry > 0) else 0.0
        add_margin_btc, extra_injection_btc = ladder(delta, cfg.initial_margin_btc)

        # Extra injection is new capital (external): first add to free, then move to pos margin
        if extra_injection_btc > 0:
            free_margin += extra_injection_btc
            pos_margin_btc += extra_injection_btc
            free_margin = max(0.0, free_margin - extra_injection_btc) + 0.0  # net: move to margin

        if model == 'inverse':
            # Exposure cap in USD: Q_cap = (wallet_btc * leverage) * price
            wallet_btc = free_margin + spot_btc + pos_margin_btc
            q_cap = wallet_btc * cfg.leverage_limit * c

            if add_margin_btc > 0:
                fill = apply_slippage(c, cfg.slippage_bps, buy=True)

                # Planned add and cap
                q_usd_planned = usd_notional_from_btc_margin(add_margin_btc, fill, cfg.leverage_limit)
                q_usd_allowable = max(0.0, min(q_usd_planned, q_cap - pos_q_usd))

                if q_usd_allowable > 0:
                    # Margin required for the allowable notional
                    margin_needed = q_usd_allowable / (cfg.leverage_limit * fill)

                    # Commission for that notional
                    fee_btc = commission_btc_from_usd(q_usd_allowable, fill, cfg.commission_rate)

                    # If not enough free to cover margin + fee, scale down proportionally
                    total_btc_needed = margin_needed + fee_btc
                    if total_btc_needed > free_margin and total_btc_needed > 0:
                        scale = free_margin / total_btc_needed
                        q_usd_allowable *= scale
                        margin_needed *= scale
                        fee_btc *= scale
                        total_btc_needed = margin_needed + fee_btc

                    if total_btc_needed > 0 and total_btc_needed <= free_margin + 1e-12:
                        # Transfer margin (internal) and pay fee (reduces wallet)
                        free_margin -= total_btc_needed
                        pos_margin_btc += margin_needed

                        # Update position notional and average entry (inverse harmonic)
                        avg_entry = avg_entry_inverse_harmonic(pos_q_usd, avg_entry or fill,
                                                               q_usd_allowable, fill)
                        pos_q_usd += q_usd_allowable

        # 8) Forced close on last day
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
                'end_pos_size_btc': 0.0,
                'end_pos_q_usd': 0.0,
                'end_equity_btc_excl_unreal': free_margin + spot_btc,
                'reason': 'FORCED-CLOSE (END DATE)',
                'duration_days': (date - cycles[-1]['start_date']).days
            })

            # Reset position
            pos_margin_btc = 0.0
            pos_size_btc = 0.0
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