from dataclasses import dataclass

@dataclass
class StrategyConfig:
    # Collateral and leverage
    initial_margin_btc: float = 1.0
    leverage_limit: float = 1.2

    # Opening and scaling
    open_margin_pct: float = 0.02
    take_profit_pct: float = 0.10

    # Costs (commission is per side; for inverse, it’s applied on USD notional and converted to BTC)
    commission_rate: float = 0.0004
    slippage_bps: float = 0.5

    # Funding
    constant_funding_daily: float = 0.1/365.0
    maintenance_margin_rate: float = 0.005

    # Risk/logic
    stop_on_liquidation: bool = True
    forced_close_on_last_day: bool = True

    # Contract mechanics: 'inverse' (BTC-margined BTCUSD-like) or 'linear' (legacy simplified)
    contract_model: str = 'inverse'

    # Backward compatibility (not used in logic but kept to avoid breaking callers)
    deterministic_fills: bool = True