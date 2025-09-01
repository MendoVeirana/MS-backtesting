import numpy as np

def usd_notional_from_btc_margin(margin_btc: float, price: float, leverage: float) -> float:
    """
    For inverse coin-M: with BTC margin m, leverage L, and price p, USD contracts Q = m * L * p.
    """
    return max(0.0, margin_btc) * leverage * price


def commission_btc_from_usd(notional_usd: float, price: float, commission_rate: float) -> float:
    """
    Commission (BTC) for inverse: fee_usd = rate * notional_usd; fee_btc = fee_usd / price.
    """
    return (commission_rate * abs(notional_usd)) / price


def funding_btc_from_usd(notional_usd: float, price: float, funding_daily_rate: float) -> float:
    """
    Funding (BTC) for inverse: funding_usd = rate * notional_usd; funding_btc = funding_usd / price.
    """
    return (funding_daily_rate * abs(notional_usd)) / price


def avg_entry_inverse_harmonic(q_usd_old: float, entry_old: float,
                               q_usd_add: float, fill_price: float) -> float:
    """
    For inverse contracts, the correct averaging uses harmonic weights:
        1/E_new = (Q_old/E_old + Q_add/fill_price) / (Q_old + Q_add)
    """
    if q_usd_add <= 0:
        return entry_old if q_usd_old > 0 else fill_price
    if q_usd_old <= 0 or entry_old <= 0:
        return fill_price
    denom = q_usd_old + q_usd_add
    inv_new = (q_usd_old / entry_old + q_usd_add / fill_price) / denom
    return 1.0 / inv_new


def liquidation_price_inverse(entry_price: float,
                              q_usd: float,
                              wallet_btc_excl_unreal: float,
                              maintenance_margin_rate: float) -> float:
    """
    Inverse (coin-margined) liquidation price.

    Solve Equity_btc(p) = mm_btc(p):
        wallet + Q*(1/E - 1/p) = (mm_rate * Q) / p
    =>  p_liq = Q * (1 + mm_rate) / (wallet + Q/E)

    Notes:
    - q_usd: USD notional (contracts); must be > 0 for a long.
    - entry_price E in USD/BTC; must be > 0.
    - wallet_btc_excl_unreal: total wallet for the position excluding unrealized PnL
      (i.e., free + position margin for isolated; account equity in cross).
    - maintenance_margin_rate as a decimal (e.g., 0.0045 for ~0.45%).
    """
    if q_usd <= 0 or entry_price <= 0:
        return 0.0
    denom_btc = wallet_btc_excl_unreal + (q_usd / entry_price)
    if denom_btc <= 0:
        # Effectively insolvent -> treat as immediate liquidation
        return float('inf')
    return q_usd * (1.0 + maintenance_margin_rate) / denom_btc


# Optional thin wrapper to make intent clear in isolated mode (use the position's wallet)
def liquidation_price_inverse_isolated(entry_price: float,
                                       q_usd: float,
                                       position_wallet_btc_excl_unreal: float,
                                       maintenance_margin_rate: float) -> float:
    return liquidation_price_inverse(entry_price, q_usd, position_wallet_btc_excl_unreal, maintenance_margin_rate)


def liquidation_price_coin_m(entry_price: float,
                             pos_size_btc: float,
                             wallet_btc_excl_unreal: float,
                             maintenance_margin_rate: float) -> float:
    """
    Simplified coin-margined liquidation price approximation.

    liq = (pos_size * entry_price) / (pos_size + wallet_btc - mm_btc)
    where mm_btc = pos_size * maintenance_margin_rate

    wallet_btc_excl_unreal: free_margin + spot + pos_margin (i.e., wallet balance in coin, excluding unrealized PnL).
    """
    if pos_size_btc <= 0:
        return 0.0
    mm_btc = pos_size_btc * maintenance_margin_rate
    denom = pos_size_btc + wallet_btc_excl_unreal - mm_btc
    if denom <= 0:
        return float('inf')
    return (pos_size_btc * entry_price) / denom


def apply_commission_btc(trade_size_btc: float, commission_rate: float) -> float:
    """
    Commission for coin-M expressed directly in BTC (rate * trade_size_btc).
    """
    return commission_rate * abs(trade_size_btc)


def apply_slippage(price: float, slippage_bps: float, buy: bool) -> float:
    """
    Apply slippage in bps to the price.
    - For buys, price goes up; for sells, price goes down.
    """
    adj = price * (slippage_bps * 1e-4)
    return price + (adj if buy else -adj)


def max_position_btc(wallet_btc: float, leverage_limit: float) -> float:
    """
    Position cap in BTC for coin-M, proportional to wallet BTC.
    """
    return max(0.0, wallet_btc * leverage_limit)