"""In-process usage and cost tracking. Stores counts and outcomes only, never prompts or responses."""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from app.llm.models import CostEstimate, ModelPricing, UsageInfo, UsageSummary

logger = logging.getLogger(__name__)

# Verified against the official Gemini API pricing page, last updated 2026-10-13.
# Only text prices are listed. Models whose price changes by date are left out, so their cost is reported as unknown.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(input_per_million_usd=0.30, output_per_million_usd=2.50, verified_on="2026-10-13"),
    "gemini-2.5-flash-lite": ModelPricing(input_per_million_usd=0.10, output_per_million_usd=0.40, verified_on="2026-10-13"),
    "gemini-3.1-flash-lite": ModelPricing(input_per_million_usd=0.25, output_per_million_usd=1.50, verified_on="2026-10-13"),
    "gemini-3.5-flash-lite": ModelPricing(input_per_million_usd=0.30, output_per_million_usd=2.50, verified_on="2026-10-13"),
    "gemini-3-flash-preview": ModelPricing(input_per_million_usd=0.50, output_per_million_usd=3.00, verified_on="2026-10-13"),
}


def resolve_pricing(
    model: str,
    input_override: float | None,
    output_override: float | None,
) -> ModelPricing | None:
    """Configured overrides win over the built-in table. Returns None when no complete price is known."""
    base = DEFAULT_PRICING.get(model)
    input_price = input_override if input_override is not None else (base.input_per_million_usd if base else None)
    output_price = output_override if output_override is not None else (base.output_per_million_usd if base else None)
    if input_price is None or output_price is None:
        return None
    if input_override is not None or output_override is not None:
        # A partly or fully overridden price is no longer the verified table entry.
        return ModelPricing(input_per_million_usd=input_price, output_per_million_usd=output_price, source_url=None)
    return base


def estimate_cost(usage: UsageInfo, pricing: ModelPricing | None) -> CostEstimate:
    if pricing is None:
        return CostEstimate(status="unknown", reason="No pricing configured for this model.")
    if usage.input_tokens is None or usage.output_tokens is None:
        return CostEstimate(status="unknown", reason="The API did not report token usage.")

    billable_output = usage.output_tokens + (usage.thoughts_tokens or 0)
    cost = (
        usage.input_tokens * pricing.input_per_million_usd + billable_output * pricing.output_per_million_usd
    ) / 1_000_000
    return CostEstimate(status="estimated", estimated_usd=round(cost, 8))


@dataclass(frozen=True)
class UsageRecord:
    recorded_at: float
    model: str
    outcome: str
    duration_ms: int
    usage: UsageInfo
    cost: CostEstimate


class TokenTracker:
    def __init__(self, max_records: int = 1000) -> None:
        self._records: deque[UsageRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, *, model: str, outcome: str, duration_ms: int, usage: UsageInfo, cost: CostEstimate) -> None:
        entry = UsageRecord(
            recorded_at=time.time(),
            model=model,
            outcome=outcome,
            duration_ms=duration_ms,
            usage=usage,
            cost=cost,
        )
        with self._lock:
            self._records.append(entry)
        # Token counts, outcome and cost only. The prompt and response are never logged.
        logger.info(
            "llm request model=%s outcome=%s input_tokens=%s output_tokens=%s cost_status=%s duration_ms=%d",
            model,
            outcome,
            usage.input_tokens,
            usage.output_tokens,
            cost.status,
            duration_ms,
        )

    def summary(self) -> UsageSummary:
        with self._lock:
            records = list(self._records)

        failed: dict[str, int] = {}
        for record in records:
            if record.outcome != "success":
                failed[record.outcome] = failed.get(record.outcome, 0) + 1

        return UsageSummary(
            total_requests=len(records),
            successful_requests=sum(1 for r in records if r.outcome == "success"),
            failed_requests=failed,
            input_tokens=sum(r.usage.input_tokens or 0 for r in records),
            output_tokens=sum((r.usage.output_tokens or 0) + (r.usage.thoughts_tokens or 0) for r in records),
            estimated_usd=round(sum(r.cost.estimated_usd or 0.0 for r in records), 8),
            requests_with_unknown_cost=sum(1 for r in records if r.cost.status == "unknown"),
        )


@lru_cache
def get_tracker() -> TokenTracker:
    return TokenTracker()
