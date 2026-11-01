"""Example guard plugin: a slow zone (speed and separation monitoring, simplified).

Inside a disc around `center` (e.g. where a person works next to the robot) every action is scaled down to
`factor` of its size. Runs between the shield's mandatory finite and bounds checks like every guard plugin.

    safety: {guards: [{type: examples.slow_zone, params: {state_key: pos, center: [0.5, 0.5], radius: 0.2, factor: 0.3}}]}
"""

from __future__ import annotations

import numpy as np

from hylo.core.safety import GuardVerdict, SafetyContext


class SlowZoneGuard:
    name = "slow_zone"

    def __init__(self, state_key: str, center: list[float], radius: float, factor: float = 0.3) -> None:
        if not 0.0 <= factor <= 1.0:
            raise ValueError("factor must be in [0, 1]: a guard may only slow down")
        self.key, self.center, self.radius, self.factor = state_key, np.asarray(center, float), float(radius), float(factor)

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        if not ctx.state or self.key not in ctx.state:
            return GuardVerdict(data, False)
        pos = np.asarray(ctx.state[self.key], float).reshape(-1)[: len(self.center)]
        if np.linalg.norm(pos - self.center) > self.radius or not np.any(data):
            return GuardVerdict(data, False)
        return GuardVerdict(data * self.factor, True, f"slow zone: scaled by {self.factor}")
