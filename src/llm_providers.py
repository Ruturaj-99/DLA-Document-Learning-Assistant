"""BYOK (Bring Your Own Key) answer engines.

Phase 1.3 of the AI Document Learning Assistant.

In simple words: besides our local Ollama model, the user may prefer to
answer with a hosted AI they already pay for. They paste THEIR OWN API key
into the app, and we call that provider's chat API with it.

IMPORTANT boundaries:
    - The key exists only in the browser session and this process's memory:
      it is never written to disk, never logged, never echoed in errors.
    - We support OpenAI-compatible chat APIs (`/chat/completions`), which
      covers OpenAI, Groq, OpenRouter, Together and Mistral with one
      implementation - each just has a different base URL.
    - The returned function has the SAME signature as ollama.ask_ollama
      (prompt in, answer text out), so qa.answer_question works unchanged.
"""

import requests

# Seconds to wait for a hosted model before giving up. Generous like the
# Ollama bridge: hosted APIs are usually fast, but not always.
TIMEOUT_SECONDS = 300

# Providers that expose an OpenAI-compatible chat API. The value is the
# API base URL the user's key belongs to; "Custom" lets the user paste a
# base URL for any other OpenAI-compatible service.
SUPPORTED_PROVIDERS = {
    "OpenAI": "https://api.openai.com/v1",
    "Groq": "https://api.groq.com/openai/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Together": "https://api.together.xyz/v1",
    "Mistral": "https://api.mistral.ai/v1",
    "Custom (OpenAI-compatible)": "",
}

# Sensible small/fast default model per provider (the user can type any
# model name their key supports).
DEFAULT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Groq": "llama-3.1-8b-instant",
    "OpenRouter": "meta-llama/llama-3.1-8b-instruct",
    "Together": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "Mistral": "mistral-small-latest",
    "Custom (OpenAI-compatible)": "",
}


def ask_openai_compatible(
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: int = TIMEOUT_SECONDS,
):
    """Build an `ask(prompt) -> str` function for one BYOK provider.

    The returned function is a drop-in replacement for ollama.ask_ollama:
    qa.answer_question(ask=...) treats both identically.

    Args:
        api_key: the user's own secret key (never stored or logged).
        model: the model name to request, e.g. "gpt-4o-mini".
        base_url: the provider's OpenAI-compatible API base URL.
        timeout_seconds: request timeout, matching the Ollama bridge.

    Raises:
        ValueError: for an empty key, model or base URL - raised when the
            function is BUILT, so the UI can complain before any call.

    Returns:
        An ask(prompt) -> str function.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Please paste your API key first.")
    if not model or not model.strip():
        raise ValueError("Please choose the model name your key supports.")
    if not base_url or not base_url.strip():
        raise ValueError("Please provide the provider's API base URL.")
    base = base_url.strip().rstrip("/")

    def ask(prompt: str) -> str:
        """One chat completion against the user's provider."""
        payload = {
            "model": model.strip(),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }
        try:
            response = requests.post(
                f"{base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach {base}. Check the API base URL and your "
                "internet connection."
            ) from None
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"{model.strip()} did not answer in time. Try again."
            ) from None
        except requests.exceptions.HTTPError as error:
            # Providers reject a bad key with 401 - say so WITHOUT echoing
            # the key itself back into the visible message.
            status = response.status_code
            detail = response.text[:200]
            if status in (401, 403):
                raise RuntimeError(
                    "The provider rejected this API key (HTTP "
                    f"{status}). Check the key and its permissions."
                ) from None
            raise RuntimeError(
                f"The provider rejected the request (HTTP {status}). "
                f"Details: {detail}"
            ) from None

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise RuntimeError(
                "The provider replied, but not in the expected chat "
                f"format. Raw reply: {response.text[:200]}"
            ) from None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The provider returned an empty answer.")
        return content

    return ask
