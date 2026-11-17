"""Phase 1.3: tests for the standalone Ollama auth proxy.

The proxy and a stub "Ollama" both run in-process on 127.0.0.1 port 0
(OS-chosen ports), so tests never touch the real Ollama or the network.
"""

import http.client
import ipaddress
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy.ollama_proxy import (  # noqa: E402
    ALLOWED_ENDPOINTS,
    MAX_BODY_BYTES,
    OllamaAuthProxyHandler,
    TOKEN_ENV,
    UPSTREAM_ENV,
    require_token,
)

TOKEN = "test-token-abc123"

CHAT_REPLY = {"message": {"content": "The heater resets in five seconds."}}
TAGS_REPLY = {"models": ["llama3.2:3b"]}

# The public Host header a Cloudflare quick tunnel produces.
TUNNEL_HOST = "random-words.trycloudflare.com"


def _host_allowed_like_real_ollama(host: str) -> bool:
    """Mimic the MEASURED behaviour of Ollama 0.40.16 on this machine:

    localhost names and loopback/private IP hosts are allowed (any port);
    public hostnames get 403 with an EMPTY body (Content-Length: 0).
    Measured via curl: Host localhost:11434 -> 200, 127.0.0.1:8765 -> 200,
    192.168.1.5:8765 -> 200, example.com -> 403, <random>.trycloudflare.com
    -> 403 empty.
    """
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


class _StubOllama(BaseHTTPRequestHandler):
    """Minimal Ollama stand-in that records what actually reached it."""

    calls = []

    def _record(self, body: bytes) -> None:
        _StubOllama.calls.append(
            {
                "method": self.command,
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "host": self.headers.get("Host"),
                "origin": self.headers.get("Origin"),
                "body": body,
            }
        )

    def _refuse_like_real_ollama(self) -> bool:
        """403 with an EMPTY body when the Host is a public hostname -
        exactly what the real Ollama does behind a tunnel."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if not _host_allowed_like_real_ollama(host):
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True
        return False

    def _empty_403(self) -> None:
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _respond(self, payload: dict, stream: bool = False) -> None:
        data = "\n".join(json.dumps(chunk) for chunk in (payload, payload)) \
            if stream else json.dumps(payload).encode("utf-8")
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if stream:
            # No Content-Length: body ends with the connection (NDJSON style).
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            half = len(data) // 2
            self.wfile.write(data[:half])
            self.wfile.flush()
            self.wfile.write(data[half:])
            self.wfile.flush()
        else:
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self._record(body)
        if self._refuse_like_real_ollama():
            return
        if b"trigger-403" in body:
            # Simulates ANY upstream empty-body error (the proxy must relay
            # it without corrupting the keep-alive connection).
            self._empty_403()
            return
        self._respond(CHAT_REPLY, stream=b"true" in body)

    def do_GET(self):  # noqa: N802
        self._record(b"")
        if self._refuse_like_real_ollama():
            return
        self._respond(TAGS_REPLY)

    def log_message(self, format, *args):  # silence the stub
        pass


@pytest.fixture()
def proxy_url(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubOllama)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    monkeypatch.setenv(UPSTREAM_ENV, f"http://127.0.0.1:{stub.server_address[1]}")

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), OllamaAuthProxyHandler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{proxy.server_address[1]}"

    proxy.shutdown()
    stub.shutdown()
    _StubOllama.calls.clear()


def _raw_request(host, port, target, token, body, content_length=None,
                 extra_headers=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        headers["Content-Length"] = str(content_length) if content_length else str(len(body))
    if extra_headers:
        headers.update(extra_headers)
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("POST" if body is not None else "GET", target,
                 body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_missing_token_is_401_and_upstream_is_never_called(proxy_url):
    status, data = _raw_request(_host(proxy_url), _port(proxy_url),
                                "/api/chat", None, json.dumps({"model": "m"}).encode())
    assert status == 401
    assert "bearer token" in data.decode()
    assert _StubOllama.calls == []


def test_wrong_token_is_401(proxy_url):
    status, _ = _raw_request(_host(proxy_url), _port(proxy_url),
                             "/api/chat", "wrong-token", b"{}")
    assert status == 401
    assert _StubOllama.calls == []


def test_valid_token_forwards_chat_and_strips_auth_upstream(proxy_url):
    body = json.dumps({"model": "llama3.2:3b", "stream": False}).encode()
    status, data = _raw_request(_host(proxy_url), _port(proxy_url),
                                "/api/chat", TOKEN, body)
    assert status == 200
    assert json.loads(data)["message"]["content"] == CHAT_REPLY["message"]["content"]
    assert len(_StubOllama.calls) == 1
    call = _StubOllama.calls[0]
    assert call["path"] == "/api/chat"
    assert call["body"] == body
    # The token must STOP at the proxy - Ollama never sees it.
    assert call["auth"] is None


def test_tags_health_check_is_allowed(proxy_url):
    status, data = _raw_request(_host(proxy_url), _port(proxy_url),
                                "/api/tags", TOKEN, None)
    assert status == 200
    assert json.loads(data) == TAGS_REPLY


@pytest.mark.parametrize("path", ["/api/pull", "/api/delete", "/api/ps", "/root"])
def test_unallowed_endpoints_are_refused_with_403(proxy_url, path):
    status, data = _raw_request(_host(proxy_url), _port(proxy_url),
                                path, TOKEN,
                                json.dumps({"name": "m"}).encode())
    assert status == 403
    assert _StubOllama.calls == []


def test_oversized_body_is_413_without_reading_it(proxy_url):
    status, _ = _raw_request(_host(proxy_url), _port(proxy_url),
                             "/api/chat", TOKEN, b"x",
                             content_length=MAX_BODY_BYTES + 1)
    assert status == 413
    assert _StubOllama.calls == []


def test_streamed_upstream_reply_is_relayed(proxy_url):
    body = json.dumps({"model": "llama3.2:3b", "stream": True}).encode()
    status, data = _raw_request(_host(proxy_url), _port(proxy_url),
                                "/api/chat", TOKEN, body)
    assert status == 200
    lines = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    assert lines == [CHAT_REPLY, CHAT_REPLY]


def test_upstream_down_returns_502(proxy_url, monkeypatch):
    # A port with (almost certainly) nothing listening.
    with socket.socket() as sleeper:
        sleeper.bind(("127.0.0.1", 0))
        dead_port = sleeper.getsockname()[1]
    monkeypatch.setenv(UPSTREAM_ENV, f"http://127.0.0.1:{dead_port}")
    status, data = _raw_request(_host(proxy_url), _port(proxy_url),
                                "/api/chat", TOKEN, b"{}")
    assert status == 502
    assert "Cannot reach Ollama" in data.decode()


def test_proxy_refuses_to_start_without_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit):
        require_token()


def test_allowed_endpoints_contract():
    # The proxy must expose exactly what DLA uses - no administration.
    assert ALLOWED_ENDPOINTS == {("POST", "/api/chat"), ("GET", "/api/tags")}


def test_logs_never_contain_the_token(proxy_url, capsys):
    body = json.dumps({"model": "llama3.2:3b"}).encode()
    _raw_request(_host(proxy_url), _port(proxy_url), "/api/chat", TOKEN, body)
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


# --- Regression: the measured tunnel 403 (root cause) -----------------------------
#
# Incident: local proxy -> 200, but the same authenticated request through
# a Cloudflare quick tunnel -> 403 (empty body). Root cause (measured
# against real Ollama 0.40.16): Ollama answers 403 to requests whose Host
# header is a public hostname; behind the tunnel the client-supplied Host
# IS the tunnel domain, and the proxy used to forward it verbatim.


def test_tunnel_style_host_header_is_normalized_to_upstream(proxy_url):
    """THE root-cause regression: a tunneled request (Host: the public
    tunnel domain) must reach the upstream with the upstream's own host,
    so the real Ollama's host allowlist lets it through."""
    body = json.dumps({"model": "llama3.2:3b", "stream": False}).encode()
    status, data = _raw_request(
        _host(proxy_url), _port(proxy_url), "/api/chat", TOKEN, body,
        extra_headers={"Host": TUNNEL_HOST},
    )
    assert status == 200, data
    assert json.loads(data)["message"]["content"] == CHAT_REPLY["message"]["content"]
    forwarded_host = _StubOllama.calls[0]["host"]
    assert forwarded_host != TUNNEL_HOST
    assert forwarded_host.split(":")[0] in ("127.0.0.1", "localhost")


def test_origin_header_is_never_forwarded(proxy_url):
    """Ollama 403s non-localhost Origin headers too (measured); the proxy
    is a server-to-server relay and strips browser-only context."""
    body = json.dumps({"model": "llama3.2:3b"}).encode()
    status, _ = _raw_request(
        _host(proxy_url), _port(proxy_url), "/api/chat", TOKEN, body,
        extra_headers={"Origin": f"https://{TUNNEL_HOST}"},
    )
    assert status == 200
    assert _StubOllama.calls[0]["origin"] is None


def test_empty_upstream_error_body_relayed_without_length_corruption(proxy_url):
    """Old bug: an empty upstream error body was REPLACED with an invented
    JSON body while the upstream's Content-Length: 0 header was kept -
    a Content-Length violation that made cloudflared reset the connection.
    After the fix: the empty 403 is relayed cleanly and the keep-alive
    connection stays usable for the next request."""
    conn = http.client.HTTPConnection(_host(proxy_url), _port(proxy_url),
                                      timeout=10)
    headers = {"Authorization": f"Bearer {TOKEN}",
               "Content-Type": "application/json"}
    # 1. The empty-body upstream 403 ...
    body = json.dumps({"model": "llama3.2:3b", "trigger": "trigger-403"}).encode()
    conn.request("POST", "/api/chat", body=body, headers=headers)
    response = conn.getresponse()
    assert response.status == 403
    # The status is relayed; the body is either the (empty) upstream one or
    # the proxy's own JSON - either way under a CORRECT Content-Length. The
    # old bug sent an invented body under the upstream's "Content-Length:
    # 0", so the client-side parse here would already be inconsistent.
    payload = response.read()
    if payload:
        json.loads(payload)  # must be well-formed, correctly delimited JSON
    # 2. ... and the SAME connection still works for the next request
    # (before the fix, the length mismatch corrupted the keep-alive stream).
    body2 = json.dumps({"model": "llama3.2:3b"}).encode()
    conn.request("POST", "/api/chat", body=body2, headers=headers)
    response2 = conn.getresponse()
    assert response2.status == 200
    assert json.loads(response2.read())["message"]["content"]
    conn.close()


def test_relay_swallows_client_disconnects_without_traceback():
    """Unit test: a client that vanishes mid-relay must not raise out of
    the handler (measured incident: ConnectionResetError traceback)."""
    class _VanishingWFile:
        def write(self, _chunk):
            raise ConnectionResetError("client gone")

        def flush(self):
            pass

    fake = type("F", (), {})()
    fake.send_response = lambda *a, **k: None
    fake.send_header = lambda *a, **k: None
    fake.end_headers = lambda: None
    fake.wfile = _VanishingWFile()
    fake.close_connection = False

    OllamaAuthProxyHandler._relay(fake, 200, {"Content-Length": "5"}, b"hello")
    assert fake.close_connection is True


# --- Live-Ollama regression (skips politely when Ollama isn't running) -------------


def test_live_ollama_accepts_tunnel_style_host_via_proxy():
    """End-to-end against the REAL Ollama on 127.0.0.1:11434: the exact
    incident request (Host: <random>.trycloudflare.com) through the proxy
    must succeed, because the proxy normalizes Host."""
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",
                                    timeout=3) as probe:
            probe.read()
    except OSError:
        pytest.skip("Ollama is not running on this machine")

    import os as _os

    _os.environ[TOKEN_ENV] = TOKEN
    _os.environ.pop(UPSTREAM_ENV, None)  # use the real default 11434
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), OllamaAuthProxyHandler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        status, data = _raw_request(
            _host(f"http://127.0.0.1:{proxy.server_address[1]}"),
            _port(f"http://127.0.0.1:{proxy.server_address[1]}"),
            "/api/tags", TOKEN, None,
            extra_headers={"Host": TUNNEL_HOST},
        )
        assert status == 200
        assert "llama3.2:3b" in json.dumps(json.loads(data))
    finally:
        proxy.shutdown()


# --- helpers ----------------------------------------------------------------------


def _host(url: str) -> str:
    return url.split("//")[1].split(":")[0]


def _port(url: str) -> int:
    return int(url.split(":")[2])
