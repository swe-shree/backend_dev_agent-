import pytest

from app.orchestrator.task_router import (
    MAX_DESCRIPTION_CHARS,
    TaskValidationError,
    classify_task,
    extract_terms,
    validate_description,
)


def test_description_is_stripped_of_control_characters():
    cleaned = validate_description("  compute_total raises\x00 KeyError\x07 when empty  ")

    assert cleaned == "compute_total raises KeyError when empty"


@pytest.mark.parametrize("text", ["short", "    ", "bug"])
def test_too_short_description_is_rejected(text):
    with pytest.raises(TaskValidationError, match="at least"):
        validate_description(text)


def test_too_long_description_is_rejected():
    with pytest.raises(TaskValidationError, match="at most"):
        validate_description("x" * (MAX_DESCRIPTION_CHARS + 1))


def test_non_bug_request_is_rejected():
    with pytest.raises(TaskValidationError, match="bug reports"):
        classify_task("Please write a poem about the ocean.", max_terms=5)


def test_quoted_strings_come_first():
    terms = extract_terms('Getting "invalid cart state" when calling checkout_cart')

    assert terms[0] == "invalid cart state"
    assert "checkout_cart" in terms


def test_code_like_identifiers_are_extracted():
    text = "handler() raises ValueError, see load_config and parseUser and os.path.join"

    terms = extract_terms(text)

    assert "handler" in terms
    assert "ValueError" in terms
    assert "load_config" in terms
    assert "parseUser" in terms
    assert "os.path.join" in terms


def test_trailing_sentence_punctuation_is_not_part_of_the_term():
    assert "compute_total" in extract_terms("compute_total.")


def test_plain_english_words_are_fallback_only():
    # No code-like token: fall back to plain words, dropping stopwords and bug keywords.
    terms = extract_terms("The checkout total is wrong after discounts apply")

    assert "checkout" in terms
    assert "total" in terms
    assert "wrong" not in terms
    assert "the" not in terms


def test_code_terms_suppress_plain_word_fallback():
    terms = extract_terms("checkout total is wrong in compute_total")

    assert terms == ["compute_total"]


def test_classification_caps_number_of_terms():
    plan = classify_task("fails in alpha_one, beta_two, gamma_three, delta_four and epsilon_five", max_terms=2)

    assert len(plan.terms) == 2
    assert plan.kind == "bug_analysis"
