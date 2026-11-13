# AI Document Learning Assistant

A private-by-design PDF workspace: upload a document, refine its pages, add
comments, replace text, run OCR, and ask questions answered from the document's
own contents. Processing stays on your computer.

Built with **Streamlit** + **pypdf / PyMuPDF / Tesseract OCR / sentence-transformers**,
with local AI answering via **Ollama** (`llama3.2:3b`).

## Run locally

> **Important:** the `.venv` already exists in this project folder. Only run
> the `python -m venv .venv` command when it is MISSING (or after stopping
> every running instance of the app). Running venv setup/pip surgery while
> the app is running leaves locked files behind and silently empties
> installed packages - the cause of `ImportError: cannot import name
> 'descriptor' from 'google.protobuf'`-type breakages.

```bash
python -m venv .venv                # only if .venv does not exist yet!
.venv\Scripts\Activate.ps1          # (Windows PowerShell)
pip install -r requirements.txt     # only needed once / after changes
streamlit run app.py --server.address 127.0.0.1
```

Optional, for the "Ask" tab: install [Ollama](https://ollama.com) locally and
pull the model (`ollama pull llama3.2:3b`). Point `OLLAMA_HOST` env variable at a
different Ollama server if needed (default `http://localhost:11434`).

Optional, for scanned-page OCR: install [Tesseract](https://github.com/tesseract-ocr/tesseract)
locally.

## Deploy to Streamlit Community Cloud (free)

The app deploys straight from this GitHub repository:

1. Push this repo to GitHub.
2. Sign in at https://share.streamlit.io with your GitHub account, choose this
   repo, main branch, and `app.py` as the entry point, then Deploy.
3. `requirements.txt` installs the Python libraries; `packages.txt` installs
   the Tesseract engine for OCR on the cloud host.

On the cloud host set the environment variable `OLLAMA_HOST` if you want the
"Ask" feature to talk to a running Ollama server; otherwise Ask simply reports
it cannot reach Ollama, and every other feature works.

## Docs / guides / OCR notes

Tip: uploaded documents are kept in memory and (for example the working copy
and its comments) in the browser session only - the repository contains no
user data (`data/` is git-ignored on purpose).
