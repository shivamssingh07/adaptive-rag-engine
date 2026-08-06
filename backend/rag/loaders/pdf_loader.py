"""PDF document loader, backed by PyMuPDF (`fitz`).

PyMuPDF was chosen over `pypdf`/`PyPDF2` for its significantly better text
extraction fidelity (layout-aware ordering, fewer garbled ligatures) and
its speed on large documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from backend.core.exceptions import CorruptedFileError
from backend.utils.ids import generate_document_id

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads a PDF file into one `Document` per page."""

    def load(self, file_path: Path) -> list[Document]:
        """Extract text from every page of a PDF.

        Args:
            file_path: Path to the `.pdf` file.

        Returns:
            One `Document` per page that contains extractable text. Pages
            with detected text (e.g. purely graphical/scanned pages) are
            silently skipped.

        Raises:
            CorruptedFileError: If the PDF cannot be opened at all, or
                contains no extractable text on any page (e.g. it is a
                scanned image with no OCR layer).
        """
        document_id = generate_document_id()

        try:
            pdf = fitz.open(file_path)
        except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
            raise CorruptedFileError(
                f"Could not open PDF '{file_path.name}': {exc}",
                details={"file": file_path.name},
            ) from exc

        documents: list[Document] = []
        try:
            for page_number in range(pdf.page_count):
                page = pdf.load_page(page_number)
                text = page.get_text("text").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "document_id": document_id,
                            "page": page_number + 1,
                            "file_type": "pdf",
                        },
                    )
                )
        except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
            raise CorruptedFileError(
                f"Failed while extracting text from '{file_path.name}': {exc}",
                details={"file": file_path.name},
            ) from exc
        finally:
            pdf.close()

        if not documents:
            raise CorruptedFileError(
                f"PDF '{file_path.name}' contains no extractable text "
                f"(it may be a scanned image without a text layer).",
                details={"file": file_path.name},
            )

        logger.info("Parsed PDF '%s': %d page(s) with text.", file_path.name, len(documents))
        return documents
