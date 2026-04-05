from __future__ import annotations

from .config import RiskLimits, StrategyParameters
from .models import FeatureSnapshot, ModeName, StrategyState


class RegimeManager:
    def __init__(self, strategy: StrategyParameters, risk: RiskLimits) -> None:
        self.strategy = strategy
        self.risk = risk

    def determine_mode(self, state: StrategyState, snapshot: FeatureSnapshot) -> ModeName:
        inventory = abs(state.inventory)
        if inventory >= self.risk.position_limit:
            return "RISK_OFF"
        if inventory >= self.risk.shock_soft_position_limit:
            return "INVENTORY_UNWIND"

        if state.last_earnings_time_ms is not None:
            elapsed = snapshot.time_ms - state.last_earnings_time_ms
            if elapsed <= self.strategy.shock_window_ms:
                return "EARNINGS_SHOCK"
            if (
                elapsed <= self.strategy.overshoot_window_ms
                and snapshot.quote_to_fair_px is not None
                and self._is_overshoot_active(state, snapshot)
            ):
                return "OVERSHOOT_FADE"

        if inventory >= self.risk.normal_soft_position_limit:
            return "INVENTORY_UNWIND"
        return "NORMAL_MM"

    def _is_overshoot_active(self, state: StrategyState, snapshot: FeatureSnapshot) -> bool:
        if snapshot.quote_to_fair_px is None:
            return False
        deviation = snapshot.quote_to_fair_px
        if state.last_earnings_direction > 0:
            return deviation >= self.strategy.overshoot_trigger_px and snapshot.trade_pressure_1s <= 0.15
        if state.last_earnings_direction < 0:
            return deviation <= -self.strategy.overshoot_trigger_px and snapshot.trade_pressure_1s >= -0.15
        return False
