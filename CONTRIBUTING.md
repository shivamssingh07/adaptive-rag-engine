# Contributing

Thanks for considering a contribution to Adaptive RAG Engine.

## Getting set up

```bash
git clone <this-repo-url>
cd adaptive-rag-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # add your GROQ_API_KEY
```

## Before opening a PR

```bash
ruff check backend/ frontend/ tests/     # lint
ruff format --check backend/ frontend/ tests/   # formatting (or: ruff format .)
mypy backend/                             # type-check
pytest                                    # tests (see docs/testing.md)
```

All four run in CI (`.github/workflows/ci.yml`) on every push and PR — please make sure they pass locally first.

## Code style

- Every function/method has a docstring (Args/Returns/Raises where relevant) and full type hints.
- Domain errors are raised as a subclass of `backend.core.exceptions.AppException`, never a bare `Exception`.
- New RAG components (retrievers, loaders, providers) should accept their dependencies as constructor parameters with a lazy-singleton default — see `backend/rag/llms/groq_provider.py` for the pattern — so they stay independently testable.
- New API routes should use `Depends(...)` on the dependency providers in `backend/api/dependencies.py`, not import and call singleton getters directly (see `docs/testing.md` for why this matters — it's what makes `app.dependency_overrides` work in tests).

## Adding a new RAG capability

If you're adding a new retriever/loader/provider:
1. Implement it as its own module under the relevant `backend/rag/` subpackage, following the existing lazy-singleton pattern.
2. Add unit tests in `tests/unit/` using the fakes in `tests/conftest.py` — no real model downloads or API calls in tests.
3. If it should be reachable from the adaptive graph, wire it into the relevant node in `backend/core/graph/nodes/`.
4. Update `docs/architecture.md`'s dependency graph and design-decisions table if the change is structural.

## Adding a new API endpoint

1. Add a Pydantic schema in `backend/api/schemas/`.
2. Add the route in `backend/api/routes/`, using `Depends(...)` for every dependency.
3. Register the router in `backend/api/main.py`.
4. Add an integration test in `tests/integration/` using the `api_client` fixture.
5. Update `docs/api_reference.md`.

## Commit messages

Conventional, imperative mood: `fix: correct BM25 score normalization`, `feat: add self-query date filtering`, `docs: update deployment guide for Railway`.

## Reporting bugs / requesting features

Open an issue with: what you expected, what happened instead, and steps to reproduce (including whether `TAVILY_API_KEY` is set, since some behavior branches on it).

## Security issues

Please see [`SECURITY.md`](SECURITY.md) — do not open a public issue for a security vulnerability.
