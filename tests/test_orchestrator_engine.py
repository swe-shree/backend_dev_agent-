import hashlib
import json

import httpx
import pytest
from google.genai import errors

from app.analysis.paths import PathValidationError
from app.config import Settings
from app.llm.gemini_client import GeminiClient, GeminiConfig, GeminiError
from app.llm.models import LLMErrorKind
from app.llm.token_tracker import TokenTracker
from app.orchestrator.engine import AnalysisEngine
from app.orchestrator.task_router import TaskValidationError
from tests.conftest import requires_git, run_git, write_file
from tests.test_llm_client import FakeSDK, _api_error, _response

BUG = "compute_total returns total_with_tax which is undefined, so the call fails with NameError"


def _settings(root, **overrides):
    values = {"allowed_repo_root": root, "agent_context_window_lines": 2}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _llm(outcome, tracker=None):
    sdk = FakeSDK(outcome)
    client = GeminiClient(
        GeminiConfig(model="gemini-3.5-flash-lite", timeout_seconds=5, thinking_budget=None),
        sdk,
        tracker or TokenTracker(),
    )
    return client, sdk


def _good_findings(**overrides):
    payload = {
        "root_cause_hypotheses": [
            {
                "statement": "compute_total returns a name that is never defined.",
                "confidence": "high",
                "evidence": [
                    {"file": "app/billing.py", "line": 9, "quote": "return total_with_tax  # BUG: undefined name"},
                    {"file": "app/billing.py", "line": 1, "quote": "this quote is not in the excerpts"},
                    {"file": "secrets/elsewhere.py", "line": 3, "quote": "anything"},
                ],
            }
        ],
        "verification_steps": ["Run pytest tests/test_billing.py"],
        "limitations": ["Only the billing module was reviewed"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _analyze(bug_repo, outcome, settings=None, tracker=None, description=BUG):
    client, sdk = _llm(outcome, tracker)
    engine = AnalysisEngine(settings or _settings(bug_repo.parent), client)
    return engine.analyze("bugrepo", description), sdk


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_successful_analysis_separates_hypotheses_from_verified_evidence(bug_repo):
    report, sdk = _analyze(bug_repo, _response(text=_good_findings()))

    assert report.status == "completed"
    hypothesis = report.hypotheses[0]
    assert hypothesis.confidence == "high"
    verified = [e for e in hypothesis.evidence if e.verified]
    unverified = [e for e in hypothesis.evidence if not e.verified]
    assert [e.line for e in verified] == [9]
    assert [e.line for e in unverified] == [1]
    # Evidence pointing at a file that was never in the context is dropped, and that is reported.
    assert all(e.file == "app/billing.py" for e in hypothesis.evidence)
    assert any("dropped" in note for note in report.limitations)
    assert report.verification_steps == ["Run pytest tests/test_billing.py"]


def test_confirmed_facts_come_from_deterministic_checks_only(bug_repo):
    report, _ = _analyze(bug_repo, _response(text=_good_findings()))

    joined = " ".join(report.confirmed_facts)
    assert "compute_total" in joined
    assert "app/billing.py" in report.affected_files
    assert any("Test files reference" in fact for fact in report.confirmed_facts)
    assert any("Analysis is text-based" in note for note in report.limitations)


def test_usage_and_cost_are_reported_from_the_api_response(bug_repo):
    tracker = TokenTracker()
    report, _ = _analyze(bug_repo, _response(text=_good_findings()), tracker=tracker)

    assert report.llm.called is True
    assert report.llm.usage.input_tokens == 1000
    assert report.llm.usage.output_tokens == 500
    assert report.llm.cost.status == "estimated"
    assert tracker.summary().total_requests == 1


def test_prompt_contains_bug_and_excerpts_but_no_secrets(bug_repo):
    _, sdk = _analyze(bug_repo, _response(text=_good_findings()))

    prompt = sdk.models.last_kwargs["contents"]
    assert BUG in prompt
    assert "<bug_report>" in prompt and "<excerpts>" in prompt
    assert "app/billing.py" in prompt
    assert "abc123-secret" not in prompt
    assert "ENV-LEAK-VALUE" not in prompt
    assert "function compute_total() {}" not in prompt


def test_excerpt_size_is_bounded_by_context_limits(bug_repo):
    settings = _settings(bug_repo.parent, agent_context_max_chars=400)
    report, _ = _analyze(bug_repo, _response(text=_good_findings()), settings=settings)

    assert report.context.characters <= 400
    assert report.context.truncated is True


def test_model_is_called_once_and_output_is_capped(bug_repo):
    _, sdk = _analyze(bug_repo, _response(text=_good_findings()))

    assert sdk.models.calls == 1
    assert sdk.models.last_kwargs["config"].max_output_tokens == 1024
    assert sdk.models.last_kwargs["config"].response_mime_type == "application/json"


def test_identical_prompt_within_a_task_reuses_the_result(bug_repo):
    client, sdk = _llm(_response(text=_good_findings()))
    engine = AnalysisEngine(_settings(bug_repo.parent), client)
    memo = {}

    first = engine._generate_once("same prompt", memo)
    second = engine._generate_once("same prompt", memo)

    assert first is second
    assert sdk.models.calls == 1


def test_no_matching_code_skips_gemini(bug_repo):
    report, sdk = _analyze(bug_repo, _response(text=_good_findings()), description="crash in NoSuchSymbolAnywhere when it runs")

    assert report.status == "insufficient_context"
    assert report.llm.called is False
    assert sdk.models.calls == 0
    assert any("Gemini was not called" in note for note in report.limitations)


def test_missing_model_configuration_still_returns_deterministic_facts(bug_repo):
    engine = AnalysisEngine(_settings(bug_repo.parent), GeminiError(LLMErrorKind.NOT_CONFIGURED, "GEMINI_API_KEY is not set."))

    report = engine.analyze("bugrepo", BUG)

    assert report.status == "llm_unavailable"
    assert report.llm.called is False
    assert report.llm.error == "not_configured"
    assert report.confirmed_facts
    assert report.affected_files


@pytest.mark.parametrize(
    "failure,kind",
    [
        (_api_error(429, "RESOURCE_EXHAUSTED", "Too many requests"), "rate_limited"),
        (_api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for metric"), "quota_exhausted"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (_api_error(401, "UNAUTHENTICATED", "bad key"), "auth"),
    ],
)
def test_provider_errors_are_reported_and_not_retried(bug_repo, failure, kind):
    report, sdk = _analyze(bug_repo, failure)

    assert report.status == "llm_error"
    assert report.llm.called is True
    assert report.llm.error == kind
    assert report.llm.usage is None
    assert report.confirmed_facts
    assert sdk.models.calls == 1


def test_non_json_model_output_is_reported_as_invalid(bug_repo):
    tracker = TokenTracker()
    report, _ = _analyze(bug_repo, _response(text="Sure! The bug is in billing.py."), tracker=tracker)

    assert report.status == "invalid_model_output"
    assert report.hypotheses == []
    # Usage is still reported because the call did happen and was billed.
    assert report.llm.usage.input_tokens == 1000
    assert tracker.summary().successful_requests == 1


def test_schema_violation_is_reported_as_invalid(bug_repo):
    bad = json.dumps({"root_cause_hypotheses": [{"statement": "x", "confidence": "certain"}]})
    report, _ = _analyze(bug_repo, _response(text=bad))

    assert report.status == "invalid_model_output"


def test_format_problem_names_the_field_but_not_the_content(bug_repo):
    bad = json.dumps({"root_cause_hypotheses": [{"statement": "SECRET-ECHO-VALUE", "confidence": "certain"}]})
    report, _ = _analyze(bug_repo, _response(text=bad))

    note = next(n for n in report.limitations if "required format" in n)
    assert "root_cause_hypotheses/0/confidence" in note
    assert "SECRET-ECHO-VALUE" not in note


def test_confidence_labels_are_normalized(bug_repo):
    upper = json.dumps({"root_cause_hypotheses": [{"statement": "x", "confidence": " High "}]})
    report, _ = _analyze(bug_repo, _response(text=upper))

    assert report.status == "completed"
    assert report.hypotheses[0].confidence == "high"


def test_fenced_json_is_accepted(bug_repo):
    fenced = "```json\n" + _good_findings() + "\n```"
    report, _ = _analyze(bug_repo, _response(text=fenced))

    assert report.status == "completed"


def test_unverifiable_model_text_is_not_treated_as_confirmed(bug_repo):
    findings = json.dumps({"root_cause_hypotheses": [{"statement": "Something", "confidence": "low", "evidence": [{"file": "app/billing.py", "line": 1, "quote": "import os"}]}]})
    report, _ = _analyze(bug_repo, _response(text=findings))

    assert report.status == "completed"
    assert report.hypotheses[0].evidence[0].verified is False
    assert report.hypotheses[0].confidence == "low"


def test_invalid_path_is_rejected_before_any_model_call(bug_repo):
    client, sdk = _llm(_response(text=_good_findings()))
    engine = AnalysisEngine(_settings(bug_repo.parent), client)

    with pytest.raises(PathValidationError):
        engine.analyze("../outside", BUG)
    assert sdk.models.calls == 0


def test_non_bug_description_is_rejected_before_any_model_call(bug_repo):
    client, sdk = _llm(_response(text=_good_findings()))
    engine = AnalysisEngine(_settings(bug_repo.parent), client)

    with pytest.raises(TaskValidationError):
        engine.analyze("bugrepo", "Please summarize the README file for me.")
    assert sdk.models.calls == 0


def test_analysis_does_not_modify_the_repository(bug_repo):
    before = _snapshot(bug_repo)

    _analyze(bug_repo, _response(text=_good_findings()))
    _analyze(bug_repo, GeminiError(LLMErrorKind.NOT_CONFIGURED, "missing"))

    assert _snapshot(bug_repo) == before


@requires_git
def test_uncommitted_changes_are_reported_as_a_fact(bug_repo):
    run_git(bug_repo, "init", "-b", "main")
    run_git(bug_repo, "add", "-A")
    run_git(bug_repo, "commit", "-m", "initial")
    write_file(bug_repo / "app" / "billing.py", "# edited\n" + (bug_repo / "app" / "billing.py").read_text(encoding="utf-8"))

    report, _ = _analyze(bug_repo, _response(text=_good_findings()))

    assert any("Uncommitted Git changes exist in: app/billing.py" in fact for fact in report.confirmed_facts)


def test_non_git_directory_is_noted_in_limitations(bug_repo):
    report, _ = _analyze(bug_repo, _response(text=_good_findings()))

    assert any("Not a Git repository" in note for note in report.limitations)
