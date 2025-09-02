from typing import Optional, Dict, Any, List, Tuple

def ladder(delta_pct: float, initial_margin_btc: float) -> Tuple[float, float]:
    """
    Returns (add_margin_btc, extra_injection_btc)
    delta_pct = (close/avg_price - 1.0)
    """
    if delta_pct > 0.0:
        return 0.00 * initial_margin_btc, 0.0
    elif delta_pct > -0.05:
        return 0.01 * initial_margin_btc, 0.0
    elif delta_pct > -0.10:
        return 0.02 * initial_margin_btc, 0.0
    elif delta_pct > -0.15:
        return 0.03 * initial_margin_btc, 0.0
    elif delta_pct > -0.20:
        return 0.04 * initial_margin_btc, 0.0
    elif delta_pct > -0.40:
        return 0.05 * initial_margin_btc, 0.05 * initial_margin_btc
    elif delta_pct > -0.60:
        return 0.10 * initial_margin_btc, 0.10 * initial_margin_btc
    else:
        return 0.0, 0.0


def ladder2(delta_pct: float, initial_margin_btc: float) -> Tuple[float, float]:
    """
    Returns (add_margin_btc, extra_injection_btc)
    Increased amounts for more aggressive DCA to pull liq down.
    """
    if delta_pct > 0.0:
        return 0.00 * initial_margin_btc, 0.0
    elif delta_pct > -0.02:  # Small drop: add more
        return 0.05 * initial_margin_btc, 0.0
    elif delta_pct > -0.05:
        return 0.10 * initial_margin_btc, 0.0
    elif delta_pct > -0.10:
        return 0.15 * initial_margin_btc, 0.0
    elif delta_pct > -0.20:
        return 0.20 * initial_margin_btc, 0.05 * initial_margin_btc
    elif delta_pct > -0.40:
        return 0.30 * initial_margin_btc, 0.10 * initial_margin_btc
    elif delta_pct > -0.60:
        return 0.40 * initial_margin_btc, 0.20 * initial_margin_btc
    else:
        return 0.0, 0.0