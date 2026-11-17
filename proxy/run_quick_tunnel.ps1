# Cloudflare Quick Tunnel - MANUAL runner (Phase 4).
#
# Run this script yourself, in a visible window, when you want the deployed
# Streamlit app to reach your local Ollama. NOTHING here starts in the
# background or at boot - by design (requires explicit approval per run).
#
#   PowerShell:  .\proxy\run_quick_tunnel.ps1
#
# Hard rules this script enforces by construction:
#   - It tunnels ONLY to the authenticated proxy on 127.0.0.1:8765.
#     NEVER to 11434 - Ollama itself must stay private.
#   - It runs cloudflared in the FOREGROUND. Closing this window (or
#     Ctrl+C) closes the tunnel.
#
# Know before you run:
#   * A Quick Tunnel URL is TEMPORARY: every restart gives a NEW
#     <random>.trycloudflare.com URL. After each restart, update
#     OLLAMA_BASE_URL in the Streamlit Cloud secrets.
#   * The quick tunnel is NOT a permanent or always-available link - it is
#     for trying things out. For a stable URL use a named tunnel
#     (see proxy/README.md step 3, "Named tunnel").

$ErrorActionPreference = "Stop"

# --- pre-flight checks ------------------------------------------------------
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared.exe not found on PATH." -ForegroundColor Red
    Write-Host "Install it first, e.g.:  winget install Cloudflare.cloudflared"
    exit 1
}

# Is the authenticated proxy up on 8765? If not, tunneling is pointless.
$proxyUp = $false
try {
    $client = New-Object Net.Sockets.TcpClient
    $client.Connect("127.0.0.1", 8765)
    $proxyUp = $client.Connected
    $client.Close()
} catch { }
if (-not $proxyUp) {
    Write-Host "Nothing is listening on 127.0.0.1:8765." -ForegroundColor Yellow
    Write-Host "Start the auth proxy FIRST (it needs the tunnel, not vice versa):"
    Write-Host '    $env:DLA_OLLAMA_TOKEN = "<your token>"'
    Write-Host "    python proxy\ollama_proxy.py"
    Write-Host "(Ollama itself must also be running with llama3.2:3b pulled.)"
    exit 1
}

Write-Host ""
Write-Host "Tunneling  https://<random>.trycloudflare.com  ->  http://127.0.0.1:8765 (auth proxy)"
Write-Host "NOT 127.0.0.1:11434 - Ollama stays private behind the proxy." -ForegroundColor Cyan
Write-Host "When cloudflared prints the https URL, set it in Streamlit Cloud secrets as:"
Write-Host '    OLLAMA_BASE_URL = "https://<that-url>"'
Write-Host "The URL changes on every restart of this command." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor Gray
Write-Host ""

# --- foreground tunnel ------------------------------------------------------
cloudflared tunnel --url http://127.0.0.1:8765
