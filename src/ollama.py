"""Python bridge to the local Ollama service.

Step 4B of the AI Document Learning Assistant.

In simple words: this file is the "phone" our project uses to call the AI
model that runs on your own computer. Our code sends a small packet of JSON
(the question) to http://localhost:11434 - the address where Ollama listens -
and gets a packet back containing the answer as text.

IMPORTANT boundaries (per the project plan):
    - This module sends ONE simple message. The clever "cheat sheet" prompt
      with citations is built in Stage 4C, NOT here.
    - No secrets, no API keys: everything stays on your machine.
    - Only the already-installed `requests` package is used for HTTP.
"""

import json
import os

import requests

# Where the Ollama service listens. By default this app asks the Ollama on
# YOUR computer (http://localhost:11434).
#
# Phase 4 (deployment): the deployed Streamlit app reads its model
# connection from Streamlit Secrets instead of a hardcoded address:
#     OLLAMA_BASE_URL   - the HTTPS Cloudflare Tunnel URL, e.g.
#                         https://<random>.trycloudflare.com
#     OLLAMA_API_TOKEN  - the shared bearer token (the SAME value the
#                         local auth proxy validates, see proxy/README.md)
# The tunnel URL is NEVER committed - it lives only in the cloud secrets.
# The older OLLAMA_HOST name still works so the plain-local setup is
# unchanged, and the default remains your own machine.
_DEFAULT_HOST = "http://localhost:11434"


def base_url() -> str:
    """The model server address, resolved per call.

    Precedence: OLLAMA_BASE_URL (deployment secret) > OLLAMA_HOST
    (classic local override) > http://localhost:11434 (plain local).
    """
    configured = (
        os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or _DEFAULT_HOST
    )
    return configured.strip().rstrip("/")


# The model you approved and pulled (see `ollama list`).
DEFAULT_MODEL = "llama3.2:3b"

# Seconds to wait for an answer on a small CPU model before giving up.
# CPU answering is slow, so we are generous here.
TIMEOUT_SECONDS = 300

# When a token is configured (deployment secret), every request carries
# "Authorization: Bearer <token>" - the local auth proxy
# (proxy/ollama_proxy.py) requires it. When NO token is set (the
# plain-local setup) NO header is sent and behaviour is exactly as before.
# Secrets are only ever read from the environment: never hardcoded, never
# logged, and never written to an exception message.
_TOKEN_ENVS = ("OLLAMA_API_TOKEN", "DLA_OLLAMA_TOKEN")  # new > legacy


def _auth_token() -> str:
    """The configured bearer token, or "" when running without a proxy."""
    for name in _TOKEN_ENVS:
        token = os.environ.get(name, "").strip()
        if token:
            return token
    return ""


def _auth_headers() -> dict[str, str]:
    """Bearer header for the auth proxy, or {} when running without one."""
    token = _auth_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def ask_ollama(
    question: str,
    passages: list[str] | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send a question to the local Ollama model and return its answer text.

    Args:
        question: what you want the model to answer (required, not blank).
        passages: optional list of document passages to include AFTER the
            question, so the model has something to look at. NOTE: the final,
            careful RAG prompt design happens in Stage 4C - this is just the
            raw plumbing.
        model: which model to ask (default: the approved llama3.2:3b).

    Returns:
        The model's answer as a plain Python string.

    Raises:
        ValueError: when the question is empty/blank.
        RuntimeError: when something is wrong with reaching or reading from
            Ollama - each message explains what to do next.
    """
    # --- 1. Basic input check ---------------------------------------------
    if not question.strip():
        raise ValueError("Please provide a non-empty question.")
    passages = passages or []

    # --- 2. Build the one simple message ----------------------------------
    # Stage 4C will replace this basic layout with the real RAG prompt.
    content = question
    if passages:
        content += "\n\nReference passages:\n" + "\n".join(
            f"- {p.strip()}" for p in passages
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,  # we want one complete answer, not streamed pieces
    }

    # --- 3. Send the request and translate failures into friendly errors ---
    try:
        response = requests.post(
            f"{base_url()}/api/chat",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json", **_auth_headers()},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()  # turns HTTP error codes into exceptions
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach the model server at {base_url()}. If this app is "
            "deployed in the cloud, the local PC, its Ollama app, or the "
            "Cloudflare tunnel may be offline right now - the tunnel URL "
            "also changes whenever the quick tunnel is restarted. "
            "Locally: start the Ollama app and try again."
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The model did not answer in time. Small CPUs can be slow; try again "
            "or use a smaller model."
        ) from None
    except requests.exceptions.HTTPError as error:
        # 502/503/530 are what Cloudflare returns when the tunnel is up in
        # DNS but the local machine (proxy/Ollama) is not answering.
        if response.status_code in (502, 503, 530):
            raise RuntimeError(
                "The tunnel was reached, but the local PC did not answer "
                f"(HTTP {response.status_code}). The local machine, its "
                "Ollama app or the auth proxy is probably offline. Start "
                "them and try again."
            ) from None
        raise RuntimeError(
            f"Ollama rejected the request (HTTP error {error}). Details: "
            f"{response.text[:300]}"
        ) from None

    # --- 4. Read the JSON answer safely ------------------------------------
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            "Ollama replied, but not in the expected JSON format. "
            f"It said: {response.text[:300]}"
        ) from None

    message = data.get("message") or {}
    answer = message.get("content", "")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(
            f"Ollama's reply had no usable answer text. Raw reply: {json.dumps(data)[:300]}"
        )

    return answer


def ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """True when Ollama is running AND the given model is downloaded.

    Used by tests to skip live-model tests politely instead of failing.
    Never downloads anything - it only asks what exists.
    """
    try:
        response = requests.get(
            f"{base_url()}/api/tags", timeout=5, headers=_auth_headers()
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return False
    try:
        names = [entry["name"] for entry in response.json().get("models", [])]
    except ValueError:
        return False
    return model in names
