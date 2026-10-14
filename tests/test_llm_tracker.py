import logging

import pytest

from app.llm.models import UsageInfo
from app.llm.token_tracker import TokenTracker, estimate_cost, resolve_pricing


def test_cost_uses_verified_flash_pricing():
    pricing = resolve_pricing("gemini-2.5-flash", None, None)
    cost = estimate_cost(UsageInfo(input_tokens=2_000_000, output_tokens=1_000_000), pricing)

    # 2M * $0.30/M + 1M * $2.50/M = $3.10
    assert cost.status == "estimated"
    assert cost.estimated_usd == pytest.approx(3.10)


def test_thinking_tokens_are_billed_as_output():
    pricing = resolve_pricing("gemini-2.5-flash", None, None)
    cost = estimate_cost(UsageInfo(input_tokens=0, output_tokens=0, thoughts_tokens=1_000_000), pricing)

    assert cost.estimated_usd == pytest.approx(2.50)


def test_missing_usage_is_unknown_cost():
    pricing = resolve_pricing("gemini-2.5-flash", None, None)

    cost = estimate_cost(UsageInfo(), pricing)

    assert cost.status == "unknown"
    assert cost.estimated_usd is None


def test_missing_pricing_is_unknown_cost():
    cost = estimate_cost(UsageInfo(input_tokens=10, output_tokens=10), None)

    assert cost.status == "unknown"


def test_model_without_verified_price_has_no_pricing():
    assert resolve_pricing("gemini-3.8-flash", None, None) is None


def test_partial_override_fills_missing_side_from_table_and_drops_verified_source():
    pricing = resolve_pricing("gemini-2.5-flash", 1.0, None)

    assert pricing.input_per_million_usd == 1.0
    assert pricing.output_per_million_usd == 2.50
    assert pricing.source_url is None


def test_tracker_summarises_success_and_failures():
    tracker = TokenTracker()
    cost = estimate_cost(UsageInfo(input_tokens=100, output_tokens=50), resolve_pricing("gemini-2.5-flash", None, None))

    tracker.record(model="gemini-2.5-flash", outcome="success", duration_ms=10, usage=UsageInfo(input_tokens=100, output_tokens=50), cost=cost)
    tracker.record(model="gemini-2.5-flash", outcome="rate_limited", duration_ms=5, usage=UsageInfo(), cost=estimate_cost(UsageInfo(), None))
    tracker.record(model="gemini-2.5-flash", outcome="rate_limited", duration_ms=5, usage=UsageInfo(), cost=estimate_cost(UsageInfo(), None))

    summary = tracker.summary()

    assert summary.total_requests == 3
    assert summary.successful_requests == 1
    assert summary.failed_requests == {"rate_limited": 2}
    assert summary.input_tokens == 100
    assert summary.output_tokens == 50
    assert summary.estimated_usd == pytest.approx(cost.estimated_usd)
    assert summary.requests_with_unknown_cost == 2


def test_tracker_is_bounded():
    tracker = TokenTracker(max_records=2)
    for _ in range(5):
        tracker.record(model="m", outcome="success", duration_ms=1, usage=UsageInfo(), cost=estimate_cost(UsageInfo(), None))

    assert tracker.summary().total_requests == 2


def test_tracker_logs_metadata_without_prompt(caplog):
    tracker = TokenTracker()
    secret_prompt = "PROMPT-CONTENT-THAT-MUST-NOT-BE-LOGGED"

    with caplog.at_level(logging.INFO, logger="app.llm.token_tracker"):
        tracker.record(model="gemini-2.5-flash", outcome="success", duration_ms=3, usage=UsageInfo(input_tokens=4), cost=estimate_cost(UsageInfo(), None))

    assert "llm request" in caplog.text
    assert "outcome=success" in caplog.text
    assert secret_prompt not in caplog.text
