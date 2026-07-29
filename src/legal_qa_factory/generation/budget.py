from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenBudget:
    max_input_tokens: int
    max_output_tokens: int
    max_cost_usd: float
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    used_cost_usd: float = 0.0

    def assert_available(self) -> None:
        if (
            self.used_input_tokens > self.max_input_tokens
            or self.used_output_tokens > self.max_output_tokens
            or self.used_cost_usd > self.max_cost_usd
        ):
            raise RuntimeError("generation budget exceeded")
