"""Phase 4: tests for the deployment client bridge (src/ollama.py).

No real network: requests is mocked. Verifies the Streamlit Secret names,
the bearer-token plumbing and the graceful tunnel-offline messages.
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ollama  # noqa: E402

# --- base URL resolution ---------------------------------------------------------


def test_base_url_defaults_to_local_ollama(monkeypatch):
    for name in ("OLLAMA_BASE_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    assert ollama.base_url() == "http://localhost:11434"


def test_legacy_host_override_still_works(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost-custom:11434")
    assert ollama.base_url() == "http://localhost-custom:11434"


def test_deployment_secret_takes_precedence(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://old:11434")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://random.trycloudflare.com/")
    assert ollama.base_url() == "https://random.trycloudflare.com"


# --- bearer token plumbing ---------------------------------------------------------


def test_no_token_means_no_auth_header(monkeypatch):
    for name in ollama._TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert ollama._auth_headers() == {}


def test_new_secret_name_sets_bearer_header(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_TOKEN", "tok-123")
    assert ollama._auth_headers() == {"Authorization": "Bearer tok-123"}


def test_legacy_secret_name_still_works(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_TOKEN", raising=False)
    monkeypatch.setenv("DLA_OLLAMA_TOKEN", "legacy-tok")
    assert ollama._auth_headers() == {"Authorization": "Bearer legacy-tok"}


def test_new_secret_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_TOKEN", "new-tok")
    monkeypatch.setenv("DLA_OLLAMA_TOKEN", "old-tok")
    assert ollama._auth_headers() == {"Authorization": "Bearer new-tok"}


# --- graceful tunnel / offline handling --------------------------------------------


def _fake_response(status: int, text: str = ""):
    response = mock.Mock()
    response.status_code = status
    response.text = text
    if status >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} Error", response=response
        )
    else:
        response.raise_for_status.side_effect = None
    response.json.return_value = {"message": {"content": "ok"}}
    return response


def _ask_with(monkeypatch, side_effect, response=None):
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://random.trycloudflare.com")
    monkeypatch.setenv("OLLAMA_API_TOKEN", "tok-123")
    with mock.patch.object(
        ollama.requests, "post",
        side_effect=side_effect, return_value=response,
    ):
        return ollama.ask_ollama("Hello")


def test_tunnel_offline_connection_error_is_friendly(monkeypatch):
    with pytest.raises(RuntimeError) as error:
        _ask_with(monkeypatch, requests.exceptions.ConnectionError("refused"))
    message = str(error.value)
    assert "tunnel" in message and "offline" in message
    assert "https://random.trycloudflare.com" in message  # URL is public info
    assert "tok-123" not in message  # the token NEVER appears


def test_cloudflare_5xx_is_reported_as_local_offline(monkeypatch):
    for status in (502, 503, 530):
        with pytest.raises(RuntimeError) as error:
            _ask_with(monkeypatch, None, response=_fake_response(status))
        assert "local PC did not answer" in str(error.value)


def test_real_ollama_http_errors_keep_their_details(monkeypatch):
    with pytest.raises(RuntimeError) as error:
        _ask_with(
            monkeypatch, None,
            response=_fake_response(400, text='{"error":"bad model"}'),
        )
    message = str(error.value)
    assert "400" in message and "bad model" in message
    assert "tok-123" not in message


def test_timeout_message_is_friendly(monkeypatch):
    with pytest.raises(RuntimeError, match="did not answer in time"):
        _ask_with(monkeypatch, requests.exceptions.Timeout())


def test_token_reaches_the_request_headers(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_TOKEN", "tok-123")
    with mock.patch.object(
        ollama.requests, "post", return_value=_fake_response(200)
    ) as post:
        ollama.ask_ollama("Hello")
    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok-123"


def test_health_check_fails_silently_when_tunnel_is_down(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://random.trycloudflare.com")
    with mock.patch.object(
        ollama.requests, "get",
        side_effect=requests.exceptions.ConnectionError("refused"),
    ):
        assert ollama.ollama_available() is False


def test_model_config_is_untouched():
    # Phase 4 preserves the approved model configuration.
    assert ollama.DEFAULT_MODEL == "llama3.2:3b"
    assert ollama.TIMEOUT_SECONDS == 300
