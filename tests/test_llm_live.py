"""Opt-in live check against the real Gemini API. Skipped unless RUN_LIVE_GEMINI=1 and a key is configured.

Run with:  RUN_LIVE_GEMINI=1 pytest tests/test_llm_live.py -s
The key is read from the project's .env file (or the environment) and is never printed.
"""

import os
from pathlib import Path

import pytest

from app.config import Settings
from app.llm.gemini_client import build_gemini_client
from app.llm.token_tracker import TokenTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]

live = pytest.mark.skipif(os.environ.get("RUN_LIVE_GEMINI") != "1", reason="set RUN_LIVE_GEMINI=1 to run")


@live
def test_live_small_request_succeeds():
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    tracker = TokenTracker()
    client = build_gemini_client(settings, tracker)

    result = client.generate("Reply with the single word: pong", max_output_tokens=32)

    assert result.text.strip()
    assert result.usage.input_tokens and result.usage.input_tokens > 0
    summary = tracker.summary()
    assert summary.successful_requests == 1
