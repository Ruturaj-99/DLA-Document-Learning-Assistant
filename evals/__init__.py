"""RAG evaluation framework (offline, deterministic).

This package evaluates the existing RAG pipeline WITHOUT touching any
production code. It only CALLS src.pdf_loader, src.chunking,
src.embeddings, src.retrieval and src.qa - the same code the app runs.

What is offline / deterministic here:
    - the evaluation document is a synthetic PDF built in code,
    - extraction runs WITHOUT OCR (native text only),
    - embeddings are the real MiniLM model (cached locally, CPU),
    - the LLM is replaced by a scripted oracle function (`ask=`),
      exactly like tests/test_qa.py does.

NOTE: submodules are imported lazily (no eager imports here) so that
`python -m evals.run_evaluation` does not re-execute the module twice.
"""

