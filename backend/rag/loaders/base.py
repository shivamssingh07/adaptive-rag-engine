"""Abstract interface for document loaders.

Every format-specific loader (PDF, DOCX, TXT/Markdown, CSV) implements
this protocol so `backend.rag.loaders.registry` can dispatch to the right
one purely by file extension, and so `backend.rag.indexing.indexer` never
needs to know about format-specific parsing details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class DocumentLoader(Protocol):
    """Structural interface every document loader must satisfy."""

    def load(self, file_path: Path) -> list[Document]:
        """Parse a file into one or more LangChain `Document` objects.

        Implementations should raise `backend.core.exceptions.CorruptedFileError`
        (not a bare exception) when the file matches the expected format but
        cannot actually be parsed, so the ingestion pipeline can isolate the
        failure to this one file in a batch upload rather than crashing.

        Args:
            file_path: Path to the file on disk.

        Returns:
            One or more `Document` objects. Multi-page/row formats (PDF,
            CSV) typically return one `Document` per page/row; single-blob
            formats (TXT, DOCX) typically return exactly one.
        """
        ...
