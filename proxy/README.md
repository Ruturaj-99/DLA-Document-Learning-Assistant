# Secure local Ollama behind a Cloudflare Tunnel (Phase 1.3)

The deployed Streamlit app needs your **local** Ollama model. The public
door is a Cloudflare Tunnel pointing at the auth proxy in this folder -
**never** at Ollama itself.

```
[Streamlit Cloud] --HTTPS+Bearer--> [Cloudflare Tunnel] --> 127.0.0.1:8765
                                                            (proxy) -->
                                                            127.0.0.1:11434
                                                            (Ollama)
```

## 1. Create the shared token (once)

On your PC (PowerShell):

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save the value. It goes into TWO places (same value, both sides):

- **Proxy side** (this PC): Windows environment variable `DLA_OLLAMA_TOKEN`
- **App side** (Streamlit Cloud): secret `DLA_OLLAMA_TOKEN` (Streamlit
  Cloud -> your app -> Settings -> Secrets, or an environment variable)

## 2. Start the proxy (on your PC, whenever you want the app answerable)

```powershell
$env:DLA_OLLAMA_TOKEN = "<token from step 1>"
python proxy\ollama_proxy.py
```

It listens **only** on `127.0.0.1:8765` and forwards just two endpoints to
Ollama (`POST /api/chat`, `GET /api/tags`); everything else gets HTTP 403,
unauthenticated requests get HTTP 401. Ollama itself must be running
(`ollama serve` or the tray app) with `llama3.2:3b` pulled.

## 3. Expose the proxy through Cloudflare (on your PC)

Quick tunnel (random URL, changes each run - good for trying it):

```powershell
cloudflared tunnel --url http://127.0.0.1:8765
```

Named tunnel (stable URL):

```powershell
cloudflared tunnel login
cloudflared tunnel create dla-ollama
cloudflared tunnel route dns dla-ollama ollama.yourdomain.com
cloudflared tunnel run --url http://127.0.0.1:8765 dla-ollama
```

Note the printed `https://...trycloudflare.com` / your domain URL.

## 4. Point the Streamlit app at it

In Streamlit Cloud (app -> Settings -> Secrets):

```toml
OLLAMA_BASE_URL = "https://<your-tunnel-url>"
OLLAMA_API_TOKEN = "<token from step 1>"
```

`OLLAMA_BASE_URL` is the HTTPS tunnel URL - the public door to the proxy,
NEVER a direct route to port 11434. `OLLAMA_API_TOKEN` is the same secret
the proxy validates. The client bridge (`src/ollama.py`) then sends
`Authorization: Bearer <token>` on every call - and sends nothing when no
token is configured, so a plain local setup keeps working unchanged
(the legacy names `OLLAMA_HOST` / `DLA_OLLAMA_TOKEN` also still work).

Quick Tunnel reality check (see step 3): the `trycloudflare.com` URL is
**temporary** - it changes every time `cloudflared` restarts, so after a
restart you must update `OLLAMA_BASE_URL` in the secrets. The quick
tunnel is NOT a permanent, always-available link; for a stable URL use a
named tunnel (also in step 3), which gets a fixed domain.

## What the proxy guarantees

| Requirement | Implementation |
|---|---|
| Listen only on 127.0.0.1:8765 | `ThreadingHTTPServer(("127.0.0.1", 8765))` |
| Bearer token required | every request, checked before anything else |
| Secret from `DLA_OLLAMA_TOKEN` | read per-request, never hardcoded |
| Constant-time compare | `hmac.compare_digest` |
| Invalid/missing token -> 401 | and the upstream is never called |
| Endpoint allowlist | only `/api/chat` (POST) + `/api/tags` (GET); `/api/pull`, `/api/delete`, ... -> 403 |
| Host/Origin normalization | the upstream sees ITS OWN host, never the public tunnel domain (measured: real Ollama 403s public-hostname Host headers with an empty body - the tunnel-403 root cause); browser-only `Origin` is stripped (measured 403 trigger) |
| Streaming supported | response chunks are relayed and flushed as they arrive |
| Timeouts / upstream down | 300 s to match the client bridge; upstream unreachable -> 502 with a friendly message |
| Robust error relay | empty upstream error bodies are relayed without inventing a body under the upstream's Content-Length; client disconnects mid-relay are swallowed, no tracebacks |
| No secrets in logs | logs contain method, path, status only - never headers or bodies |
| Stdlib only | `http.server`, `urllib`, `hmac`, `json`, `os` |
| Separate from DLA | imports nothing from `src/`; standalone file |

## BYOK alternative

Users without access to your machine can answer with their own hosted API
key instead: in the Ask tab choose "Bring your own API key (BYOK)", pick a
provider (OpenAI, Groq, OpenRouter, Together, Mistral, or any
OpenAI-compatible URL), paste the key and model name. The key lives in the
browser session only - it is never written to disk, logged or echoed in
error messages (a rejected key says "check the key", not the key).
