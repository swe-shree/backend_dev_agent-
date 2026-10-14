import hashlib
import json

import pytest

from app.config import Settings
from app.fixes.patch_generator import PatchProposalEngine
from app.llm.gemini_client import GeminiClient, GeminiConfig, GeminiError
from app.llm.models import LLMErrorKind
from app.llm.token_tracker import TokenTracker
from app.orchestrator.engine import AnalysisEngine
from app.orchestrator.models import AnalysisReport, ContextSummary, EvidenceRef, Hypothesis, LLMSummary
from tests.conftest import write_file
from tests.test_llm_client import FakeSDK, _api_error, _response

BUG = "compute_total returns total_with_tax which is undefined, so the call fails with NameError"
QUOTE = "return total_with_tax  # BUG: undefined name"

# Replaces the planted bug on app/billing.py line 9. Context lines 7-8 must match the file exactly.
GOOD_DIFF = (
    "--- a/app/billing.py\n"
    "+++ b/app/billing.py\n"
    "@@ -7,3 +7,3 @@\n"
    "     for item in items:\n"
    '         total += item["price"] * item["qty"]\n'
    "-    return total_with_tax  # BUG: undefined name\n"
    "+    return total\n"
)


def _settings(root, **overrides):
    values = {"allowed_repo_root": root, "agent_context_window_lines": 2}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _llm(outcome):
    sdk = FakeSDK(outcome)
    client = GeminiClient(GeminiConfig(model="gemini-3.5-flash-lite", timeout_seconds=5, thinking_budget=None), sdk, TokenTracker())
    return client, sdk


def _fix_json(**overrides):
    payload = {
        "root_cause_summary": "compute_total returns a name that is never defined.",
        "explanation": "Return the computed total instead of the undefined name.",
        "unified_diff": GOOD_DIFF,
        "verification_steps": ["Run pytest tests/test_billing.py"],
        "limitations": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _analysis(*, verified=True, status="completed", confidence="high", quote=QUOTE, hypotheses=None):
    if hypotheses is None:
        hypotheses = [
            Hypothesis(
                statement="compute_total returns an undefined name.",
                confidence=confidence,
                evidence=[EvidenceRef(file="app/billing.py", line=9, quote=quote, verified=verified)],
            )
        ]
    return AnalysisReport(
        status=status,
        repository="bugrepo",
        search_terms=["compute_total"],
        confirmed_facts=[],
        hypotheses=hypotheses,
        affected_files=["app/billing.py"],
        verification_steps=[],
        limitations=[],
        context=ContextSummary(
            files=["app/billing.py"],
            blocks=1,
            characters=100,
            terms_searched=["compute_total"],
            matched_terms={},
            total_matches=1,
            truncated=False,
        ),
        llm=LLMSummary(called=True, model="gemini-3.5-flash-lite"),
    )


def _propose(root, outcome, analysis=None, settings=None, path="bugrepo"):
    client, sdk = _llm(outcome)
    engine = PatchProposalEngine(settings if settings is not None else _settings(root.parent), client)
    return engine.propose(path, BUG, analysis if analysis is not None else _analysis()), sdk


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_valid_model_patch_is_proposed_as_unverified(bug_repo):
    report, sdk = _propose(bug_repo, _response(text=_fix_json()))

    assert report.status == "proposed"
    assert report.proposed_patch.status == "proposed_unverified"
    assert report.proposed_patch.checks.correctness == "unverified"
    assert report.proposed_patch.checks.tests_run is False
    assert report.proposed_patch.checks.python_files_parse == ["app/billing.py"]
    assert [f.path for f in report.proposed_patch.files] == ["app/billing.py"]
    assert report.proposed_patch.files[0].hunks == 1
    assert report.proposed_patch.unified_diff.endswith("\n")
    assert report.affected_files == ["app/billing.py"]
    assert any("Unverified draft" in w for w in report.warnings)
    assert report.root_cause_summary.startswith("compute_total")
    assert report.verification_steps == ["Run pytest tests/test_billing.py"]
    assert sdk.models.calls == 1


def test_request_uses_json_mode_and_the_fix_output_cap(bug_repo):
    _, sdk = _propose(bug_repo, _response(text=_fix_json()))

    assert sdk.models.last_kwargs["config"].max_output_tokens == 1536
    assert sdk.models.last_kwargs["config"].response_mime_type == "application/json"


def test_prompt_contains_only_bounded_excerpts_and_no_secrets(bug_repo):
    _, sdk = _propose(bug_repo, _response(text=_fix_json()))

    prompt = sdk.models.last_kwargs["contents"]
    assert "<prior_analysis>" in prompt and "<bug_report>" in prompt and "<excerpts>" in prompt
    assert "app/billing.py" in prompt
    assert "abc123-secret" not in prompt
    assert "API_TOKEN" not in prompt  # line 1 is outside the window around the cited line
    assert "ENV-LEAK-VALUE" not in prompt
    assert "function compute_total() {}" not in prompt


def test_existing_analysis_is_reused_without_a_second_analysis_call(bug_repo):
    findings = json.dumps(
        {
            "root_cause_hypotheses": [
                {
                    "statement": "compute_total returns an undefined name.",
                    "confidence": "high",
                    "evidence": [{"file": "app/billing.py", "line": 9, "quote": QUOTE}],
                }
            ],
            "verification_steps": [],
            "limitations": [],
        }
    )
    analysis_client, analysis_sdk = _llm(_response(text=findings))
    analysis = AnalysisEngine(_settings(bug_repo.parent), analysis_client).analyze("bugrepo", BUG)
    assert analysis.status == "completed"

    report, fix_sdk = _propose(bug_repo, _response(text=_fix_json()), analysis=analysis)

    assert report.status == "proposed"
    assert analysis_sdk.models.calls == 1  # the analysis call from before
    assert fix_sdk.models.calls == 1  # only the patch-generation call


def test_empty_patch_is_reported_as_no_patch(bug_repo):
    report, sdk = _propose(bug_repo, _response(text=_fix_json(unified_diff="")))

    assert report.status == "no_patch"
    assert report.proposed_patch is None
    assert report.explanation
    assert sdk.models.calls == 1


@pytest.mark.parametrize("diff", ["   \n", "\n"])
def test_blank_patch_is_treated_as_empty(bug_repo, diff):
    report, _ = _propose(bug_repo, _response(text=_fix_json(unified_diff=diff)))

    assert report.status == "no_patch"


def test_malformed_patch_is_rejected_and_not_returned(bug_repo):
    bad = "--- a/app/billing.py\n+++ b/app/billing.py\n@@ nonsense @@\n"

    report, _ = _propose(bug_repo, _response(text=_fix_json(unified_diff=bad)))

    assert report.status == "patch_rejected"
    assert report.proposed_patch is None
    assert any("Malformed hunk header" in w for w in report.warnings)


def test_traversal_patch_is_rejected(bug_repo):
    bad = "--- a/../outside.txt\n+++ b/../outside.txt\n@@ -1 +1 @@\n-a\n+b\n"

    report, _ = _propose(bug_repo, _response(text=_fix_json(unified_diff=bad)))

    assert report.status == "patch_rejected"
    assert report.proposed_patch is None


def test_sensitive_target_is_rejected_without_echoing_its_content(bug_repo):
    bad = "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-compute_total=ENV-LEAK-VALUE\n+x=1\n"

    report, _ = _propose(bug_repo, _response(text=_fix_json(unified_diff=bad)))

    assert report.status == "patch_rejected"
    assert "ENV-LEAK-VALUE" not in report.model_dump_json()


def test_binary_patch_is_rejected(bug_repo):
    bad = "diff --git a/app/billing.py b/app/billing.py\nBinary files a/app/billing.py and b/app/billing.py differ\n"

    report, _ = _propose(bug_repo, _response(text=_fix_json(unified_diff=bad)))

    assert report.status == "patch_rejected"


def test_patch_for_a_file_the_model_never_saw_is_rejected(bug_repo):
    bad = "--- a/app/dup.py\n+++ b/app/dup.py\n@@ -1 +1 @@\n-x\n+y\n"

    report, _ = _propose(bug_repo, _response(text=_fix_json(unified_diff=bad)))

    assert report.status == "patch_rejected"
    assert any("not in the code context" in w for w in report.warnings)


def test_patch_that_does_not_match_current_file_is_rejected(bug_repo):
    bad = GOOD_DIFF.replace("     for item in items:", "     for thing in things:")

    report, _ = _propose(bug_repo, _response(text=_fix_json(unified_diff=bad)))

    assert report.status == "patch_rejected"
    assert any("does not match the current file" in w for w in report.warnings)


def test_gemini_error_is_reported_and_not_retried(bug_repo):
    report, sdk = _propose(bug_repo, _api_error(429, "RESOURCE_EXHAUSTED", "Too many requests"))

    assert report.status == "llm_error"
    assert report.llm.error == "rate_limited"
    assert report.llm.called is True
    assert report.proposed_patch is None
    assert sdk.models.calls == 1


def test_missing_usage_metadata_marks_cost_unknown(bug_repo):
    report, _ = _propose(bug_repo, _response(text=_fix_json(), usage=False))

    assert report.status == "proposed"
    assert report.llm.usage.input_tokens is None
    assert report.llm.cost.status == "unknown"
    assert report.llm.cost.estimated_usd is None


def test_usage_and_cost_come_from_the_api_response(bug_repo):
    report, _ = _propose(bug_repo, _response(text=_fix_json()))

    assert report.llm.usage.input_tokens == 1000
    assert report.llm.usage.output_tokens == 500
    assert report.llm.cost.status == "estimated"


def test_non_json_model_output_is_invalid(bug_repo):
    report, _ = _propose(bug_repo, _response(text="Here is a patch: ..."))

    assert report.status == "invalid_model_output"
    assert report.proposed_patch is None


def test_schema_violation_names_the_field_but_not_the_value(bug_repo):
    bad = json.dumps({"root_cause_summary": "x", "unified_diff": 12345, "verification_steps": "SECRET-ECHO-VALUE"})

    report, _ = _propose(bug_repo, _response(text=bad))

    assert report.status == "invalid_model_output"
    joined = " ".join(report.limitations)
    assert "verification_steps" in joined
    assert "SECRET-ECHO-VALUE" not in joined


def test_no_verified_evidence_means_no_model_call(bug_repo):
    report, sdk = _propose(bug_repo, _response(text=_fix_json()), analysis=_analysis(verified=False))

    assert report.status == "insufficient_evidence"
    assert sdk.models.calls == 0
    assert report.llm.called is False


def test_incomplete_analysis_means_no_model_call(bug_repo):
    report, sdk = _propose(bug_repo, _response(text=_fix_json()), analysis=_analysis(status="llm_error"))

    assert report.status == "insufficient_evidence"
    assert sdk.models.calls == 0


def test_quote_that_no_longer_matches_the_file_is_not_used(bug_repo):
    write_file(bug_repo / "app" / "billing.py", "def compute_total(items):\n    return sum(items)\n")

    report, sdk = _propose(bug_repo, _response(text=_fix_json()))

    assert report.status == "insufficient_evidence"
    assert sdk.models.calls == 0
    assert any("did not match the current files" in n for n in report.limitations)


def test_unavailable_model_returns_status_without_a_call(bug_repo):
    engine = PatchProposalEngine(_settings(bug_repo.parent), GeminiError(LLMErrorKind.NOT_CONFIGURED, "GEMINI_API_KEY is not set."))

    report = engine.propose("bugrepo", BUG, _analysis())

    assert report.status == "llm_unavailable"
    assert report.llm.called is False
    assert report.llm.error == "not_configured"


def test_context_budget_is_enforced(bug_repo):
    tiny = _settings(bug_repo.parent, agent_fix_max_context_chars=1)

    report, sdk = _propose(bug_repo, _response(text=_fix_json()), settings=tiny)

    assert report.status == "insufficient_evidence"
    assert sdk.models.calls == 0
    assert report.context_truncated is True


def test_context_size_is_reported_within_the_budget(bug_repo):
    report, _ = _propose(bug_repo, _response(text=_fix_json()))

    assert 0 < report.context_characters <= 8000
    assert report.context_files == ["app/billing.py"]


def test_patch_is_limited_to_configured_changed_lines(bug_repo):
    tight = _settings(bug_repo.parent, agent_fix_max_changed_lines=1)

    report, _ = _propose(bug_repo, _response(text=_fix_json()), settings=tight)

    assert report.status == "patch_rejected"
    assert any("changes 2 lines" in w for w in report.warnings)


def test_proposals_never_modify_the_repository(bug_repo):
    before = _snapshot(bug_repo)

    _propose(bug_repo, _response(text=_fix_json()))
    _propose(bug_repo, _response(text=_fix_json(unified_diff="")))
    _propose(bug_repo, _response(text="not json"))
    _propose(bug_repo, _response(text=_fix_json(unified_diff="--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\n")))
    _propose(bug_repo, _response(text=_fix_json()), analysis=_analysis(verified=False))

    assert _snapshot(bug_repo) == before


def test_hunk_count_mismatch_is_rejected_not_repaired(bug_repo):
    # Regression for the live failure "A hunk ... ends before its line counts are reached".
    # The header claims one more old line than the body contains. The patch must be rejected, not repaired.
    bad = GOOD_DIFF.replace("@@ -7,3 +7,3 @@", "@@ -7,4 +7,3 @@")

    report, sdk = _propose(bug_repo, _response(text=_fix_json(unified_diff=bad)))

    assert report.status == "patch_rejected"
    assert report.proposed_patch is None
    assert sdk.models.calls == 1
    warnings = " ".join(report.warnings)
    assert "A hunk in app/billing.py ends before its line counts are reached" in warnings
    assert "header expects 4 old and 3 new lines, found 3 and 3" in warnings


def test_prompt_states_the_hunk_counting_rules(bug_repo):
    _, sdk = _propose(bug_repo, _response(text=_fix_json()))

    prompt = sdk.models.last_kwargs["contents"]
    assert "old_count = context lines (' ') + removed lines ('-')" in prompt
    assert "new_count = context lines (' ') + added lines ('+')" in prompt
    assert "Never write a hunk that spans a skipped number" in prompt
