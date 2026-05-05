"""NavigationCommand dataclass — the output of any policy (heuristic or RL).

This is the *only* type published over ZMQ to RasPi.
Keeping it in a dedicated module avoids circular imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum


class NavigationMode(IntEnum):
    CRUISE = 0
    CAUTIOUS = 1
    AVOID = 2
    RESERVED = 3
    STOP = 4


MODE_NAMES = {m: m.name for m in NavigationMode}


@dataclass
class NavigationCommand:
    mode: NavigationMode = NavigationMode.STOP
    velocity_scale: float = 0.0  # [-1.0, 1.0] — âm = lùi, dương = tiến (Linear X)
    velocity_y: float = 0.0  # [-1.0, 1.0] — âm = phải, dương = trái (Linear Y - Mecanum)
    heading_offset: float = 0.0  # radians [-π/4, π/4] (Angular Z)
    confidence: float = 1.0  # policy confidence [0, 1]
    safety_override: bool = False  # True if safety monitor changed cmd
    timestamp: float = field(default_factory=time.time)

    def is_safe_to_move(self) -> bool:
        return self.mode != NavigationMode.STOP and (
            self.velocity_scale != 0.0 or self.velocity_y != 0.0
        )

    def __repr__(self) -> str:
        return (
            f"NavCmd({MODE_NAMES[self.mode]} "
            f"vx={self.velocity_scale:.2f} vy={self.velocity_y:.2f} "
            f"h={self.heading_offset:.3f}rad "
            f"{'[SAFE] ' if self.safety_override else ''}"
            f"conf={self.confidence:.2f})"
        )

    def clip(self) -> NavigationCommand:
        import math

        self.velocity_scale = float(max(-1.0, min(1.0, self.velocity_scale)))
        self.velocity_y = float(max(-1.0, min(1.0, self.velocity_y)))
        self.heading_offset = float(max(-math.pi / 4, min(math.pi / 4, self.heading_offset)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        return self
