

import json
import os

import requests

# Where the Ollama service listens. By default this app asks the Ollama on
# YOUR computer (http://localhost:11434). On a deployed server set the
# OLLAMA_HOST environment variable to point at a running Ollama, e.g.:
#     OLLAMA_HOST=http://your-ollama-server:11434
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# The model you approved and pulled (see `ollama list`).
DEFAULT_MODEL = "llama3.2:3b"

# Seconds to wait for an answer on a small CPU model before giving up.
# CPU answering is slow, so we are generous here.
TIMEOUT_SECONDS = 300


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
            f"{OLLAMA_HOST}/api/chat",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()  # turns HTTP error codes into exceptions
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Is the Ollama app running? "
            "Start it (check the small llama icon in your system tray) and try again."
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The model did not answer in time. Small CPUs can be slow; try again "
            "or use a smaller model."
        ) from None
    except requests.exceptions.HTTPError as error:
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
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return False
    try:
        names = [entry["name"] for entry in response.json().get("models", [])]
    except ValueError:
        return False
    return model in names
