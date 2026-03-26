"""
API Cost Tracker — tracks token usage and estimated USD cost for Azure OpenAI calls.

Pricing (as of 2024):
  gpt-5.1:               $2.50 / 1M input tokens,  $10.00 / 1M output tokens
  text-embedding-3-small: $0.02 / 1M input tokens
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Pricing table (USD per 1M tokens) ────────────────────────────────────────
MODEL_PRICING: dict[str, dict] = {
    "gpt-5.1": {
        "input":  2.50,
        "output": 10.00,
    },
    "text-embedding-3-small": {
        "input":  0.02,
        "output": 0.00,
    },
}

# Fallback pricing for unknown models
DEFAULT_PRICING = {"input": 2.50, "output": 10.00}


@dataclass
class UsageRecord:
    timestamp: str
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """
    Lightweight in-memory tracker for Azure OpenAI API usage and estimated cost.

    Usage:
        tracker = CostTracker()
        tracker.record("gpt-5.1", input_tokens=500, output_tokens=200, operation="generate")
        print(tracker.total_cost)   # → 0.000xxx
        print(tracker.total_tokens) # → 700
    """

    def __init__(self):
        self._records: list[UsageRecord] = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "unknown",
    ) -> float:
        """
        Record a single API call and return the estimated cost (USD) for that call.
        model: deployment name, e.g. "gpt-5.1" or "text-embedding-3-small"
        """
        pricing = self._get_pricing(model)
        cost = (
            (input_tokens  / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )

        record = UsageRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._records.append(record)
        return cost

    # ── Aggregated properties ─────────────────────────────────────────────────

    @property
    def total_cost(self) -> float:
        """Total estimated cost in USD across all recorded calls."""
        return sum(r.cost_usd for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all recorded calls."""
        return sum(r.input_tokens + r.output_tokens for r in self._records)

    @property
    def records(self) -> list[UsageRecord]:
        return list(self._records)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all recorded usage."""
        self._records.clear()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_pricing(self, model: str) -> dict:
        """
        Look up pricing by matching model name substring.
        Falls back to DEFAULT_PRICING if no match found.
        """
        model_lower = model.lower()
        for key, pricing in MODEL_PRICING.items():
            if key in model_lower:
                return pricing
        return DEFAULT_PRICING

    def summary_str(self) -> str:
        """Return a compact display string like '$0.0034 · 1,234 tokens'."""
        return f"${self.total_cost:.4f} · {self.total_tokens:,} tokens"
