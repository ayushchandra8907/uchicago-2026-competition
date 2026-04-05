from __future__ import annotations

from .config import RiskLimits
from .models import ModeName, Side


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def soft_limit(self, mode: ModeName) -> int:
        if mode == "EARNINGS_SHOCK":
            return self.limits.shock_soft_position_limit
        return self.limits.normal_soft_position_limit

    def side_capacity(self, side: Side, inventory: int, *, mode: ModeName) -> int:
        soft_limit = self.soft_limit(mode)
        if mode in {"INVENTORY_UNWIND", "RISK_OFF"}:
            if side == "BUY" and inventory >= 0:
                return 0
            if side == "SELL" and inventory <= 0:
                return 0
        if side == "BUY":
            return max(0, soft_limit - inventory)
        return max(0, soft_limit + inventory)

    def clamp_order_qty(self, qty: int, *, side: Side, inventory: int, resting_open_qty: int, mode: ModeName) -> int:
        capped = min(int(qty), self.limits.order_size_limit)
        capped = min(capped, max(0, self.limits.open_volume_limit - resting_open_qty))
        capped = min(capped, self.side_capacity(side, inventory, mode=mode))
        if side == "BUY":
            capped = min(capped, self.limits.position_limit - inventory)
        else:
            capped = min(capped, self.limits.position_limit + inventory)
        return max(0, capped)
