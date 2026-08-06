"""Unit tests for `backend.rag.loaders.splitter`."""

from __future__ import annotations

from langchain_core.documents import Document

from backend.config.settings import Settings
from backend.rag.loaders.splitter import DocumentSplitter


class TestDocumentSplitter:
    def test_splits_long_document_into_multiple_chunks(self, settings: Settings) -> None:
        splitter = DocumentSplitter(settings)
        long_text = "This is a sentence about topic A. " * 100
        doc = Document(page_content=long_text, metadata={"document_id": "doc_1"})

        chunks = splitter.split([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= settings.chunk_size + 50  # small tolerance

    def test_short_document_stays_one_chunk(self, settings: Settings) -> None:
        splitter = DocumentSplitter(settings)
        doc = Document(page_content="A short sentence.", metadata={"document_id": "doc_1"})

        chunks = splitter.split([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == "A short sentence."

    def test_chunk_ids_are_deterministic_and_sequential(self, settings: Settings) -> None:
        splitter = DocumentSplitter(settings)
        long_text = "Sentence about topic B. " * 100
        doc = Document(page_content=long_text, metadata={"document_id": "doc_42"})

        chunks = splitter.split([doc])

        for index, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_id"] == f"doc_42_chunk_{index}"
            assert chunk.metadata["chunk_index"] == index

    def test_preserves_parent_metadata(self, settings: Settings) -> None:
        splitter = DocumentSplitter(settings)
        doc = Document(
            page_content="Some content here.",
            metadata={"document_id": "doc_1", "source": "report.pdf", "page": 3},
        )

        chunks = splitter.split([doc])

        assert chunks[0].metadata["source"] == "report.pdf"
        assert chunks[0].metadata["page"] == 3

    def test_multiple_input_documents_all_split(self, settings: Settings) -> None:
        splitter = DocumentSplitter(settings)
        docs = [
            Document(page_content="First document content.", metadata={"document_id": "doc_1"}),
            Document(page_content="Second document content.", metadata={"document_id": "doc_2"}),
        ]

        chunks = splitter.split(docs)

        assert len(chunks) == 2
        assert {c.metadata["document_id"] for c in chunks} == {"doc_1", "doc_2"}
