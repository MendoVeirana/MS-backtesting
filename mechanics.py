import numpy as np

def usd_notional_from_btc_margin(margin_btc: float, price: float, leverage: float) -> float:
    """
    Inverse coin-M: with BTC margin m, leverage L, and price p, USD contracts Q = m * L * p.
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
    For inverse contracts, average entry is harmonic-weighted by USD notional.
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
    Exact inverse (coin-margined) liquidation price for a long:
      wallet + Q*(1/E - 1/p) = (mm_rate * Q)/p
      => p_liq = Q * (1 + mm_rate) / (wallet + Q/E)

    Inputs:
    - entry_price: average entry (E)
    - q_usd: total USD contracts (Q), fixed between trades
    - wallet_btc_excl_unreal: isolated wallet for this position EXCLUDING unrealized PnL
    - maintenance_margin_rate: e.g. 0.004 to 0.006 typical
    """
    if q_usd <= 0 or entry_price <= 0:
        return 0.0
    denom_btc = wallet_btc_excl_unreal + (q_usd / entry_price)
    if denom_btc <= 0:
        return float('inf')
    return q_usd * (1.0 + maintenance_margin_rate) / denom_btc


def apply_slippage(price: float, slippage_bps: float, buy: bool) -> float:
    """
    Apply slippage in bps to the price.
    - For buys, price goes up; for sells, price goes down.
    """
    adj = price * (slippage_bps * 1e-4)
    return price + (adj if buy else -adj)