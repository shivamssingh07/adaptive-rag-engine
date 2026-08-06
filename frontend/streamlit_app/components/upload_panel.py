"""Upload panel component: drag-and-drop multi-file upload with per-file
success/failure feedback."""

from __future__ import annotations

import streamlit as st

from frontend.streamlit_app.services.api_client import APIClient, APIError

_MIME_BY_EXTENSION = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
}


def render_upload_panel(client: APIClient) -> None:
    """Render the drag-and-drop upload widget and process submissions.

    Args:
        client: Backend API client.
    """
    with st.expander("📤 Upload documents", expanded=not st.session_state["messages"]):
        st.caption("Drag and drop, or browse. Supports PDF, DOCX, TXT, Markdown, and CSV.")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded_files and st.button("Index uploaded files", type="primary"):
            _process_uploads(client, uploaded_files)


def _process_uploads(client: APIClient, uploaded_files: list) -> None:
    """Send uploaded files to the backend and render a per-file result summary.

    Args:
        client: Backend API client.
        uploaded_files: Files returned by `st.file_uploader`.
    """
    files_payload = []
    for uploaded in uploaded_files:
        extension = uploaded.name.rsplit(".", 1)[-1].lower()
        mime = _MIME_BY_EXTENSION.get(extension, "application/octet-stream")
        files_payload.append((uploaded.name, uploaded.getvalue(), mime))

    with st.spinner(f"Indexing {len(files_payload)} file(s)... this may take a moment."):
        try:
            summary = client.upload_files(files_payload)
        except APIError as exc:
            st.error(f"Upload failed: {exc.message}")
            return
        except Exception as exc:  # noqa: BLE001 - surface any connectivity issue plainly
            st.error(f"Could not reach the backend: {exc}")
            return

    st.session_state["last_upload_summary"] = summary

    if summary["successful_files"] > 0:
        st.success(
            f"✅ Indexed {summary['successful_files']}/{summary['total_files']} file(s) "
            f"({summary['total_chunks_added']} chunk(s) added)."
        )
    if summary["failed_files"] > 0:
        st.warning(f"⚠️ {summary['failed_files']} file(s) could not be indexed.")

    for result in summary["results"]:
        if result["duplicate"]:
            st.info(f"↩️ **{result['filename']}** — identical content already indexed, skipped.")
        elif result["success"]:
            st.write(f"✅ **{result['filename']}** — {result['chunks_added']} chunk(s)")
        else:
            st.write(f"❌ **{result['filename']}** — {result['error']}")
