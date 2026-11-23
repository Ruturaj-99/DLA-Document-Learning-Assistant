"""Phase 1.3: tests for the BYOK (Bring Your Own Key) answer engine.

The HTTP call is faked with unittest.mock, so no real provider is contacted
and no real key is needed. The key must never leak into error messages.
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm_providers as lp  # noqa: E402


def _response(status: int = 200, payload: dict | None = None, text: str = ""):
    """A fake requests.Response."""
    response = mock.Mock()
    response.status_code = status
    if payload is None:
        payload = {
            "choices": [{"message": {"content": "The answer is 42."}}]
        }
    response.json.return_value = payload
    response.text = text or str(payload)
    if status >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} Error", response=response
        )
    else:
        response.raise_for_status.side_effect = None
    return response


def _ask(**overrides):
    return lp.ask_openai_compatible(
        api_key=overrides.get("api_key", "sk-test-123"),
        model=overrides.get("model", "gpt-4o-mini"),
        base_url=overrides.get("base_url", "https://api.openai.com/v1"),
    )


def test_builds_and_answers_from_choices():
    ask = _ask()
    with mock.patch.object(lp.requests, "post", return_value=_response()):
        assert ask("What is the answer?") == "The answer is 42."


def test_request_shape_and_auth_header():
    ask = _ask(api_key="sk-secret")
    with mock.patch.object(lp.requests, "post", return_value=_response()) as post:
        ask("Hello")
    args, kwargs = post.call_args
    assert args[0].endswith("/chat/completions")
    assert kwargs["headers"]["Authorization"] == "Bearer sk-secret"
    assert kwargs["json"]["stream"] is False
    assert kwargs["json"]["temperature"] == 0  # Stage 5M: deterministic extraction
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "Hello"}]


def test_empty_key_model_or_base_url_fails_at_build_time():
    with pytest.raises(ValueError):  # noqa: PT012 (assertion inside)
        _ask(api_key="   ")
    with pytest.raises(ValueError):
        _ask(model="")
    with pytest.raises(ValueError):
        _ask(base_url="")


def test_rejected_key_message_does_not_echo_the_key():
    ask = _ask(api_key="sk-super-secret")
    with mock.patch.object(
        lp.requests, "post", return_value=_response(status=401)
    ):
        with pytest.raises(RuntimeError) as error:
            ask("Hello")
    assert "rejected this API key" in str(error.value)
    assert "sk-super-secret" not in str(error.value)


def test_other_http_errors_keep_provider_details():
    ask = _ask()
    with mock.patch.object(
        lp.requests, "post",
        return_value=_response(status=429, payload={"oops": "quota"}),
    ):
        with pytest.raises(RuntimeError) as error:
            ask("Hello")
    assert "429" in str(error.value)


def test_connection_problem_and_timeout_are_friendly():
    ask = _ask()
    with mock.patch.object(
        lp.requests, "post",
        side_effect=requests.exceptions.ConnectionError("no route"),
    ):
        with pytest.raises(RuntimeError, match="Cannot reach"):
            ask("Hello")
    with mock.patch.object(
        lp.requests, "post", side_effect=requests.exceptions.Timeout()
    ):
        with pytest.raises(RuntimeError, match="did not answer in time"):
            ask("Hello")


def test_malformed_or_empty_replies_are_rejected():
    ask = _ask()
    with mock.patch.object(
        lp.requests, "post",
        return_value=_response(payload={"unexpected": "shape"}),
    ):
        with pytest.raises(RuntimeError, match="not in the expected chat format"):
            ask("Hello")
    with mock.patch.object(
        lp.requests, "post",
        return_value=_response(
            payload={"choices": [{"message": {"content": "   "}}]}
        ),
    ):
        with pytest.raises(RuntimeError, match="empty answer"):
            ask("Hello")


def test_provider_registry_and_ask_fn_compatibility():
    # Every known provider has a base URL and a default model, and the
    # custom entry exists for self-hosted OpenAI-compatible servers.
    for name, base in lp.SUPPORTED_PROVIDERS.items():
        assert isinstance(base, str)
    assert lp.DEFAULT_MODELS["Groq"]
    # The built function is a drop-in for qa.answer_question's ask=:
    ask = _ask()
    with mock.patch.object(lp.requests, "post", return_value=_response()):
        assert callable(ask) and isinstance(ask("q"), str)


# --- Phase 4: BYOK and local Ollama share ONE pipeline ---------------------------
#
# Requirement: no duplicated RAG/QA logic. The provider is ONLY the ask()
# function; retrieval, the 3-tier answerability gate, refusal retry,
# grounding and citations in qa.answer_question are identical for both
# engines. These tests run the REAL pipeline with each engine's ask fn.

from src import qa  # noqa: E402
from src.chunking import Chunk  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402

BYOK_CHUNKS = [
    Chunk("manual", 2, "Deliveries take five working days, no matter how long the order list is.", 0, 76),
    Chunk("manual", 1, "Store the manual in a dry place.", 80, 112),
]
BYOK_EMBEDDED = embed_chunks(BYOK_CHUNKS)

GROUNDED = {"choices": [{"message": {"content": "Deliveries take five working days."}}]}


def _grounded_response():
    return _response(payload=GROUNDED)


def test_byok_gets_the_full_pipeline():
    """A BYOK answer is gated, grounded and cited exactly like a local one."""
    ask = _ask()
    with mock.patch.object(lp.requests, "post", return_value=_grounded_response()):
        bundle = qa.answer_question(
            "How long do deliveries take?", BYOK_EMBEDDED, ask=ask
        )
    assert bundle.not_found is False
    assert bundle.reason == "ok"
    assert bundle.sources, "citations attached by the shared pipeline"


def test_byok_answer_passes_the_grounding_check():
    """An invented value is discarded for BOTH engines - same rule."""
    ask = _ask()
    with mock.patch.object(
        lp.requests, "post",
        return_value=_response(
            payload={"choices": [{"message": {"content": "Deliveries take 99 days."}}]}
        ),
    ):
        bundle = qa.answer_question(
            "How long do deliveries take?", BYOK_EMBEDDED, ask=ask
        )
    assert bundle.not_found is True
    assert bundle.reason == "ungrounded_answer"
    assert bundle.sources == []


def test_byok_refusal_enters_the_same_retry_path():
    """A hosted model that wrongly refuses gets the one evidence-backed
    retry - identical to the local Ollama behaviour."""
    ask = _ask()
    responses = iter(
        [
            _response(payload={"choices": [{"message": {"content": qa.NOT_FOUND_MARKER}}]}),
            _grounded_response(),  # second call: the real answer
        ]
    )
    with mock.patch.object(lp.requests, "post", side_effect=lambda *a, **k: next(responses)):
        bundle = qa.answer_question(
            "How long do deliveries take?", BYOK_EMBEDDED, ask=ask
        )
    assert bundle.not_found is False
    assert bundle.reason == "ok"


def test_local_and_byok_produce_equivalent_bundles():
    """Same evidence, same truthful answer -> same bundle shape, whichever
    engine produced the text."""
    answers = {}
    for name, ask_fn in (
        ("local", lambda prompt: "Deliveries take five working days."),
        ("byok", _ask()),
    ):
        if name == "byok":
            with mock.patch.object(lp.requests, "post", return_value=_grounded_response()):
                bundle = qa.answer_question(
                    "How long do deliveries take?", BYOK_EMBEDDED, ask=ask_fn
                )
        else:
            bundle = qa.answer_question(
                "How long do deliveries take?", BYOK_EMBEDDED, ask=ask_fn
            )
        answers[name] = bundle
    assert answers["local"].not_found == answers["byok"].not_found
    assert answers["local"].reason == answers["byok"].reason == "ok"
    assert [s.page_number for s in answers["local"].sources] == [
        s.page_number for s in answers["byok"].sources
    ]
