"""Step 4B: tests for src/ollama.py.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

Most tests MOCK the HTTP call (a fake response object stands in for the real
one), so no model download and no real Ollama call is needed for them - they
fail fast and never touch the network. One optional LIVE test talks to the
real local model, and politely skips itself when Ollama or the model is
absent.
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest
import requests

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ollama  # noqa: E402


def fake_response(status_code=200, json_value=None, raw_text=""):
    """Build a stand-in object shaped like requests' real Response."""
    fake = mock.Mock()
    fake.status_code = status_code
    fake.text = raw_text
    if json_value is None:
        fake.json.side_effect = ValueError("not JSON")  # like a broken reply
    else:
        fake.json.return_value = json_value
    if status_code >= 400:
        fake.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error"
        )
    else:
        fake.raise_for_status.return_value = None
    return fake


def test_empty_question_raises_without_any_http_call():
    with mock.patch.object(ollama.requests, "post") as post_mock:
        for blank in ["", "   "]:
            with pytest.raises(ValueError, match="non-empty question"):
                ollama.ask_ollama(blank)
        post_mock.assert_not_called()  # a blank question never reaches the network


def test_connection_error_becomes_friendly_runtime_error():
    with mock.patch.object(
        ollama.requests, "post", side_effect=requests.exceptions.ConnectionError()
    ):
        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            ollama.ask_ollama("Why does the fan spin up?")


def test_timeout_becomes_friendly_runtime_error():
    with mock.patch.object(
        ollama.requests, "post", side_effect=requests.exceptions.Timeout()
    ):
        with pytest.raises(RuntimeError, match="did not answer in time"):
            ollama.ask_ollama("Why does the fan spin up?")


def test_http_error_includes_status_info():
    with mock.patch.object(
        ollama.requests, "post", return_value=fake_response(status_code=404, raw_text="model not found")
    ):
        with pytest.raises(RuntimeError, match="404"):
            ollama.ask_ollama("What is inside the manual?")


def test_non_json_response_is_reported():
    with mock.patch.object(
        ollama.requests, "post", return_value=fake_response(status_code=200, json_value=None, raw_text="hello world")
    ):
        with pytest.raises(RuntimeError, match="not in the expected JSON format"):
            ollama.ask_ollama("What is inside the manual?")


def test_reply_without_answer_text_is_reported():
    broken_json = {"message": {"role": "assistant"}}  # no "content" key
    with mock.patch.object(
        ollama.requests, "post", return_value=fake_response(status_code=200, json_value=broken_json)
    ):
        with pytest.raises(RuntimeError, match="no usable answer text"):
            ollama.ask_ollama("What is inside the manual?")


def test_success_returns_plain_answer_text():
    good_json = {"message": {"role": "assistant", "content": "Two-speed fan, cleaned quarterly."}}
    with mock.patch.object(
        ollama.requests, "post", return_value=fake_response(status_code=200, json_value=good_json)
    ) as post_mock:
        answer = ollama.ask_ollama("How often is the filter cleaned?")
        assert answer == "Two-speed fan, cleaned quarterly."
        # Check what was actually sent: the right model, one message, not streamed.
        sent = post_mock.call_args.kwargs["data"]
        assert "llama3.2:3b" in sent
        assert '"stream":false' in sent.replace(" ", "")


def test_passages_are_included_in_the_sent_message():
    good_json = {"message": {"role": "assistant", "content": "ok"}}
    with mock.patch.object(
        ollama.requests, "post", return_value=fake_response(status_code=200, json_value=good_json)
    ) as post_mock:
        ollama.ask_ollama("Summarize", passages=["The filter needs cleaning.", "The fan has two speeds."])
        sent = post_mock.call_args.kwargs["data"]
        assert "The filter needs cleaning." in sent


def test_ollama_available_detects_running_service():
    # Directly exercising the availability probe against the REAL local
    # service is itself a safe, read-only check. If Ollama is not running,
    # this just documents the offline behaviour. Never downloads anything.
    result = ollama.ollama_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(
    not (sys.modules.get("src.ollama") and ollama.ollama_available()),
    reason="live model test: local Ollama or llama3.2:3b unavailable",
)
def test_live_local_model_answers():
    """Real handshake with the local model. Skipped when unavailable."""
    answer = ollama.ask_ollama("What is the capital of France? Answer in one word.")
    assert answer.strip(), "the live model must return some text"
    assert "Paris" in answer
