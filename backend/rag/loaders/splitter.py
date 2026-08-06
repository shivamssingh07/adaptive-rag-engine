"""Document chunking, backed by LangChain's `RecursiveCharacterTextSplitter`.

Chunk size and overlap are configurable via `Settings` (`CHUNK_SIZE`,
`CHUNK_OVERLAP` in `.env`), defaulting to 1000/200 characters.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config.settings import Settings, get_settings
from backend.utils.ids import generate_chunk_id

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """Splits loaded documents into overlapping chunks suitable for
    embedding and retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the splitter with the configured chunk size/overlap.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
        """
        self._settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split a list of loaded documents into chunks.

        Each output chunk inherits and extends its parent document's
        metadata with `chunk_id` (a deterministic ID derived from the
        parent `document_id` and global chunk position) and `chunk_index`.

        Args:
            documents: Documents produced by a loader (see
                `backend.rag.loaders`), one per page/row/file depending on
                format.

        Returns:
            The full list of chunked `Document` objects across all inputs.
        """
        chunks: list[Document] = []
        global_index = 0

        for document in documents:
            document_id = str(document.metadata.get("document_id", "unknown"))
            texts = self._splitter.split_text(document.page_content)

            for _, text in enumerate(texts):
                metadata = dict(document.metadata)

                metadata["chunk_id"] = generate_chunk_id(
                    document_id,
                    global_index,
                )
                metadata["chunk_index"] = global_index

                chunks.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

                global_index += 1

        logger.info(
            "Split %d document(s) into %d chunk(s) (chunk_size=%d, overlap=%d).",
            len(documents),
            len(chunks),
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )

        return chunks
