"""Standalone authentication proxy in front of a local Ollama server.

Phase 1.3 of the AI Document Learning Assistant.

In simple words: the deployed Streamlit app (in the cloud) must NOT be able
to reach your PC's Ollama directly. Instead, this small program runs on the
PC, accepts ONLY authenticated, approved requests on 127.0.0.1:8765, and
forwards them to the real Ollama on 127.0.0.1:11434. A Cloudflare Tunnel
then exposes 127.0.0.1:8765 to the internet, so the public entry point is
the proxy - not Ollama itself.

Security boundaries (per the phase 1.3 requirements):
    - Listens ONLY on 127.0.0.1:8765 - never on a public interface.
    - Requires "Authorization: Bearer <token>" on every request; the token
      comes from the Windows environment variable DLA_OLLAMA_TOKEN and is
      NEVER hardcoded or logged.
    - Token comparison is constant-time (hmac.compare_digest).
    - Missing/invalid token -> HTTP 401 (the upstream is never called).
    - ONLY the endpoints DLA actually needs are forwarded:
          POST /api/chat   (the one request qa.py's bridge makes)
          GET  /api/tags   (read-only; used by the health check)
      Model management (/api/pull, /api/delete, ...) is refused with 403.
    - Reasonable upstream timeouts; upstream down -> HTTP 502.
    - Streaming responses are passed through chunk by chunk (DLA itself
      uses stream:false, but the proxy does not buffer streamed replies).
    - Logs contain method, path, status and duration only - never
      authorization headers, API keys or document contents.
    - Standard library only, and fully separate from the DLA application
      (this file imports nothing from src/).

Run:
    set DLA_OLLAMA_TOKEN=<same token as the Streamlit secret>
    python proxy\\ollama_proxy.py
"""

import hmac
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Where the proxy listens. localhost ONLY: the tunnel is the public door,
# this port must never be exposed by a port-forward.
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8765

# Where the real Ollama runs. Override with DLA_OLLAMA_UPSTREAM if needed.
UPSTREAM_ENV = "DLA_OLLAMA_UPSTREAM"
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"

# Windows environment variable holding the shared secret. The SAME value
# must be configured as DLA_OLLAMA_TOKEN on the Streamlit Cloud side.
TOKEN_ENV = "DLA_OLLAMA_TOKEN"

# Endpoints DLA actually uses. Anything else is refused with 403 - the
# proxy must never expose Ollama's model-management surface.
ALLOWED_ENDPOINTS = {
    ("POST", "/api/chat"),
    ("GET", "/api/tags"),
}

# Safety valve: DLA's largest request is one RAG prompt (passages included),
# a few hundred KB. 8 MB is far above that and still blocks trivial abuse.
MAX_BODY_BYTES = 8 * 1024 * 1024

# Matches the client bridge (src/ollama.py TIMEOUT_SECONDS): small local
# models can be slow on CPU, so we stay generous.
UPSTREAM_TIMEOUT_SECONDS = 300

# Headers we never copy in either direction (RFC 7230 hop-by-hop + auth,
# which must stop at the proxy: Ollama sees a local request, not a token).
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "authorization",
}

# Headers stripped from the forwarded request, measured against real
# Ollama 0.40.16: it answers 403 (empty body) to requests whose Host is a
# public hostname or whose Origin is not localhost - and through a tunnel
# the client-supplied Host IS the public tunnel domain. The proxy is a
# server-to-server relay: the upstream sees its own host, and browser-only
# context headers (Origin) are meaningless here and only harmful.
_STRIPPED_HEADERS = {"origin"}


def _upstream_base() -> str:
    return os.environ.get(UPSTREAM_ENV, DEFAULT_UPSTREAM).rstrip("/")


def _expected_token() -> str:
    """The shared secret, read per-request (tests can reconfigure it)."""
    return os.environ.get(TOKEN_ENV, "")


def require_token() -> str:
    """Fail fast at startup: a proxy without a secret is an open door."""
    token = _expected_token()
    if not token.strip():
        raise SystemExit(
            f"Refusing to start: set the {TOKEN_ENV} environment variable "
            "first (e.g. a 32-byte random string from "
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\")."
        )
    return token


def _authorized(header_value: str | None) -> bool:
    """Constant-time Bearer-token check. Never compares plain ==."""
    expected = _expected_token()
    if not expected or not header_value:
        return False
    scheme, _, provided = header_value.partition(" ")
    if scheme.strip().casefold() != "bearer":
        return False
    return hmac.compare_digest(provided.strip(), expected)


class OllamaAuthProxyHandler(BaseHTTPRequestHandler):
    """One request = authenticate -> allowlist -> forward -> stream back."""

    # HTTP/1.1 lets us stream chunked replies back to the client.
    protocol_version = "HTTP/1.1"
    server_version = "DLA-OllamaProxy/1.0"
    sys_version = ""

    # --- helpers -----------------------------------------------------------

    def _send_json(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Error paths (401/403/413) may reject BEFORE reading the request
        # body; with HTTP/1.1 keep-alive the un-read body bytes would then
        # be misparsed as a new request. Closing the connection removes
        # that confusion entirely.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes | None:
        """The request body, or None when it must be refused outright."""
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length else 0
        except ValueError:
            self._send_json(400, "Malformed Content-Length header.")
            return None
        if size > MAX_BODY_BYTES:
            self._send_json(413, f"Request body exceeds {MAX_BODY_BYTES} bytes.")
            return None
        return self.rfile.read(size) if size else b""

    def _forward(self, body: bytes) -> None:
        """Send the request to Ollama and relay the reply (streamed)."""
        url = f"{_upstream_base()}{self.path}"
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.casefold() not in _HOP_BY_HOP
            and name.casefold() not in _STRIPPED_HEADERS
        }
        # Root-cause fix for the tunnel 403 (measured on Ollama 0.40.16):
        # Ollama answers 403 with an EMPTY body to any request whose Host
        # header is a public hostname, and behind a tunnel the
        # client-supplied Host IS the public tunnel domain. A reverse
        # proxy must not forward the client's Host anyway: the upstream
        # sees its own host instead.
        headers["Host"] = urllib.parse.urlsplit(_upstream_base()).netloc
        request = urllib.request.Request(url, data=body if body else None,
                                         headers=headers, method=self.command)
        try:
            upstream = urllib.request.urlopen(
                request, timeout=UPSTREAM_TIMEOUT_SECONDS
            )
        except urllib.error.HTTPError as error:
            # Ollama's own 4xx/5xx carry useful JSON - relay them as-is,
            # still without ever echoing headers here.
            try:
                relay = error.read(MAX_BODY_BYTES)
            except OSError:
                relay = b""
            error.close()
            if relay:
                self._relay(error.status if hasattr(error, "status") else error.code,
                            error.headers, relay)
            else:
                # The upstream answered WITHOUT a body (Ollama's 403 does
                # exactly this: Content-Length: 0). Its headers claim zero
                # bytes, so an invented replacement body would violate
                # Content-Length and corrupt keep-alive clients (measured:
                # cloudflared reset the connection). Answer with our own
                # JSON, under OUR correct Content-Length, keeping the
                # upstream status.
                self._send_json(
                    error.status if hasattr(error, "status") else error.code,
                    f"Upstream returned HTTP "
                    f"{error.status if hasattr(error, 'status') else error.code} "
                    "without a body.",
                )
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            self._send_json(
                502,
                f"Cannot reach Ollama at {_upstream_base()}. "
                "Is the Ollama app running on this machine?",
            )
            return

        with upstream:
            self._relay(upstream.status, upstream.headers,
                        lambda: upstream.read(64 * 1024))

    def _relay(self, status, headers, read_chunk) -> None:
        """Stream `read_chunk()`'s bytes to the client with sane headers."""
        try:
            self.send_response(status)
            for name, value in headers.items():
                if name.casefold() in _HOP_BY_HOP:
                    continue
                self.send_header(name, value)
            length_header = headers.get("Content-Length") if headers else None
            if not length_header:
                # Unknown length (e.g. Ollama's NDJSON streaming mode): the
                # body ends when the connection closes - valid HTTP/1.1.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            while True:
                chunk = read_chunk() if callable(read_chunk) else read_chunk
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The client went away mid-relay (e.g. the tunnel closed the
            # stream). Nothing can be sent back to a disconnected client -
            # drop the connection quietly instead of spraying a traceback.
            self.close_connection = True
        except OSError:
            # The upstream died mid-stream: headers may already be on the
            # wire, so the cleanest signal left is closing the (truncated)
            # response instead of crashing the handler thread.
            self.close_connection = True

    # --- HTTP verbs ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        self._handle()

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        self._handle()

    # --- the single entry point ----------------------------------------------

    def _handle(self) -> None:
        # 1. Authentication FIRST: unauthenticated requests learn nothing.
        if not _authorized(self.headers.get("Authorization")):
            self._send_json(401, "Missing or invalid bearer token.")
            return

        # 2. Allowlist: only what DLA needs, nothing administrative.
        if (self.command, self.path) not in ALLOWED_ENDPOINTS:
            self._send_json(
                403, f"Endpoint {self.command} {self.path} is not allowed by DLA."
            )
            return

        # 3. Forward.
        body = self._read_body() if self.command == "POST" else b""
        if body is None:  # refused inside _read_body (413/400 already sent)
            return
        self._forward(body)

    # --- privacy-first logging -----------------------------------------------

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Delegate to the default one-liner, which records ONLY the request
        line ("POST /api/chat") and the status code - BaseHTTPRequestHandler's
        log never includes headers or bodies, so the token and document
        contents can never reach the console."""
        super().log_message(format, *args)


def main() -> None:
    require_token()
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT),
                                 OllamaAuthProxyHandler)
    print(
        f"DLA Ollama proxy listening on http://{LISTEN_HOST}:{LISTEN_PORT} "
        f"-> {_upstream_base()} | allowed: "
        + ", ".join(f"{method} {path}" for method, path in sorted(ALLOWED_ENDPOINTS))
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Proxy stopped.")


if __name__ == "__main__":
    main()
