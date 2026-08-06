"""Adaptive RAG Engine — Streamlit dashboard entrypoint.

Run with:
    streamlit run frontend/streamlit_app/app.py

Configure the backend URL via the `API_BASE_URL` environment variable
(defaults to `http://localhost:8000/api/v1`).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit's script runner puts only this script's own directory
# (frontend/streamlit_app/) on sys.path — NOT the project root. Since this
# app uses absolute imports (`from frontend.streamlit_app... import ...`)
# so that shared code reads identically whether imported by Streamlit,
# by a test, or by another tool, the project root must be added to
# sys.path explicitly before any such import is attempted.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import UTC, datetime  # noqa: E402

import streamlit as st  # noqa: E402

from frontend.streamlit_app.components.chat_panel import render_chat_panel  # noqa: E402
from frontend.streamlit_app.components.sidebar import render_sidebar  # noqa: E402
from frontend.streamlit_app.components.upload_panel import render_upload_panel  # noqa: E402
from frontend.streamlit_app.services.api_client import (  # noqa: E402
    APIClient,
    APIError,
    get_api_client,
)
from frontend.streamlit_app.state.session_state import init_state  # noqa: E402

st.set_page_config(
    page_title="Adaptive RAG Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_documents_page(client: APIClient) -> None:
    """Render the Documents management page: list + per-document delete."""
    st.header("📄 Indexed documents")

    try:
        data = client.list_documents()
    except APIError as exc:
        st.error(f"Could not load documents: {exc.message}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach the backend: {exc}")
        return

    st.caption(f"{data['total_documents']} document(s), {data['total_chunks']} chunk(s) total.")

    if not data["documents"]:
        st.info("No documents indexed yet. Upload some from the Chat page.")
        return

    for doc in data["documents"]:
        col1, col2, col3, col4 = st.columns([4, 1, 2, 1])
        col1.markdown(f"**{doc['filename']}**")
        col2.caption(doc["file_type"].upper())
        col3.caption(f"{doc['chunk_count']} chunks · {doc['size_bytes'] / 1024:.1f} KB")
        if col4.button("🗑️ Delete", key=f"delete_{doc['document_id']}"):
            try:
                client.delete_document(doc["document_id"])
                st.success(f"Deleted '{doc['filename']}'.")
                st.rerun()
            except APIError as exc:
                st.error(f"Delete failed: {exc.message}")
        st.divider()


def render_settings_page(client: APIClient) -> None:
    """Render the Settings page: current backend configuration (read-only)."""
    st.header("⚙️ Settings")
    st.caption(
        "Configuration is set via environment variables (`.env`) on the backend. "
        "This page is read-only; edit `.env` and restart the backend to change these."
    )

    try:
        config = client.get_config()
    except APIError as exc:
        st.error(f"Could not load configuration: {exc.message}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach the backend: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Models")
        st.write(f"**LLM:** `{config['llm_model']}`")
        st.write(f"**Embedding model:** `{config['embedding_model']}`")
        st.write(f"**Reranker model:** `{config['reranker_model']}`")
        if config["tavily_web_search_enabled"]:
            web_search_status = "✅ enabled"
        else:
            web_search_status = "❌ disabled (no TAVILY_API_KEY)"
        st.write(f"**Web search fallback:** {web_search_status}")

        st.subheader("Environment")
        st.write(f"**App:** {config['app_name']} v{config['app_version']}")
        st.write(f"**Environment:** `{config['environment']}`")

    with col2:
        st.subheader("Retrieval tuning")
        st.write(f"**Chunk size / overlap:** {config['chunk_size']} / {config['chunk_overlap']}")
        st.write(f"**Top-k retrieval:** {config['top_k_retrieval']}")
        st.write(f"**Top-k reranked:** {config['top_k_rerank']}")
        st.write(
            f"**Hybrid weights:** {config['hybrid_bm25_weight']:.1f} BM25 / "
            f"{config['hybrid_vector_weight']:.1f} vector"
        )

        st.subheader("Self-correction limits")
        st.write(f"**Max document-relevance retries:** {config['max_document_grade_retries']}")
        st.write(f"**Max groundedness retries:** {config['max_groundedness_retries']}")

        st.subheader("Uploads")
        st.write(f"**Allowed extensions:** {', '.join(config['allowed_extensions'])}")
        st.write(f"**Max upload size:** {config['max_upload_size_mb']} MB")

    with st.expander("Raw configuration (JSON)"):
        st.json(config)


def render_about_page() -> None:
    """Render the static About page."""
    st.header("ℹ️ About")
    st.markdown(
        """
### Adaptive RAG Engine

A self-correcting, hybrid-retrieval Adaptive RAG system built entirely on
**free, open-source infrastructure** — no OpenAI, no paid vector database,
no paid search API required.

#### What makes it "adaptive"?

1. **Adaptive routing** — an LLM classifies each question and the graph
   takes a different path (local documents / web search / direct answer)
   depending on the answer.
2. **Adaptive retrieval** — the retrieval strategy (hybrid BM25+vector,
   MMR, multi-query expansion, or self-query filtering) is chosen
   per-query based on its characteristics.
3. **Adaptive correction** — a relevance grader can trigger a bounded
   query-rewrite-and-retry loop, and a groundedness grader can trigger a
   bounded regenerate loop, before ever falling back to a best-effort
   answer.

#### Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq — Llama 3.3 70B Versatile |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (local, persisted to disk) |
| Lexical search | BM25 (`rank-bm25`) |
| Reranking | Cross-encoder — `BAAI/bge-reranker-base` |
| Backend | FastAPI |
| Frontend | Streamlit |
| Web search (optional) | Tavily |

#### Links

- API documentation: `/docs` on the backend (Swagger UI)
- Source: see the project's `README.md` for architecture diagrams and
  the full setup guide.
        """
    )
    st.caption(f"Page rendered {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")


def main() -> None:
    """Application entrypoint: initialize state, render sidebar, route to
    the selected page."""
    init_state()
    client = get_api_client()

    render_sidebar(client)

    page = st.session_state["page"]
    if page == "Chat":
        st.title("💬 Chat with your documents")
        render_upload_panel(client)
        render_chat_panel(client)
    elif page == "Documents":
        render_documents_page(client)
    elif page == "Settings":
        render_settings_page(client)
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
