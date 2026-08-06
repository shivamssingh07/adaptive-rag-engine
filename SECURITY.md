# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in this project, please **do not**
open a public GitHub issue. Instead:

1. Open a [private security advisory](../../security/advisories/new) on this
   repository (GitHub's built-in private disclosure mechanism), or
2. If that's not available, contact the maintainers directly (see the
   repository's contact information) with a description of the issue,
   steps to reproduce, and potential impact.

Please allow a reasonable amount of time for a fix to be developed and
released before any public disclosure.

## Supported versions

This is an actively-developed portfolio/reference project without formal
LTS branches. Security fixes are applied to the latest version on the
default branch.

## Scope and known considerations

A few things worth knowing about this project's security posture, since
it's designed for local/self-hosted use rather than as a multi-tenant SaaS:

- **No built-in authentication.** `POST /reset` and `DELETE /documents/{id}`
  are destructive and unauthenticated by default. If you deploy this
  publicly, put it behind your own auth layer (a reverse proxy with basic
  auth, an API gateway, etc.) — this is called out explicitly in
  `docs/deployment.md`'s production checklist.
- **`allow_dangerous_deserialization=True`** is used when loading the
  persisted FAISS index (`backend/rag/indexing/faiss_store.py`). This is
  safe *only* because the index is always one this application wrote
  itself to a local path under its own control — it is never loaded from
  an untrusted or user-supplied source. Do not point `FAISS_INDEX_DIR` at
  a location writable by an untrusted party.
- **Uploaded files are parsed by PyMuPDF / python-docx / pandas.** These
  are mature, widely-used libraries, but as with any file-parsing code
  path, don't run this service with elevated privileges, and keep
  dependencies up to date (`pip list --outdated`).
- **Secrets** (`GROQ_API_KEY`, `TAVILY_API_KEY`) are read from environment
  variables / `.env` only, never logged, and stored in memory as
  `pydantic.SecretStr`. `.env` is git-ignored by default — never commit it.
- **CORS** defaults to `*` for local development convenience. Set
  `CORS_ORIGINS` to your actual frontend origin(s) before deploying
  publicly (see `docs/deployment.md`).

## Dependency updates

This project pins exact versions in `requirements.txt` for reproducibility.
Periodically check for security advisories on key dependencies (FastAPI,
LangChain/LangGraph, Streamlit) and update pinned versions accordingly.
