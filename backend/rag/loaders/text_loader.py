"""Plain text and Markdown document loader.

Markdown is intentionally loaded as plain text rather than parsed into an
AST — the `RecursiveCharacterTextSplitter` used downstream already
respects Markdown's blank-line paragraph structure reasonably well via its
default separator hierarchy, and preserving raw Markdown syntax (headers,
links) in the chunk text gives the LLM useful structural signal at
generation time.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from backend.core.exceptions import CorruptedFileError
from backend.utils.ids import generate_document_id

logger = logging.getLogger(__name__)


class TextLoader:
    """Loads a `.txt` or `.md` file into a single `Document`."""

    def load(self, file_path: Path) -> list[Document]:
        """Read a plain text or Markdown file.

        Attempts UTF-8 first, then falls back to Latin-1 for files with
        non-UTF-8 encoding rather than failing outright, since plain text
        files in the wild are not reliably UTF-8.

        Args:
            file_path: Path to the `.txt` or `.md` file.

        Returns:
            A single-element list containing one `Document`.

        Raises:
            CorruptedFileError: If the file cannot be decoded in either
                encoding, or is empty.
        """
        document_id = generate_document_id()

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding="latin-1")
                logger.warning(
                    "File '%s' was not valid UTF-8; decoded as Latin-1 instead.",
                    file_path.name,
                )
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise CorruptedFileError(
                    f"Could not decode text file '{file_path.name}': {exc}",
                    details={"file": file_path.name},
                ) from exc
        except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
            raise CorruptedFileError(
                f"Could not read text file '{file_path.name}': {exc}",
                details={"file": file_path.name},
            ) from exc

        if not text.strip():
            raise CorruptedFileError(
                f"File '{file_path.name}' is empty.",
                details={"file": file_path.name},
            )

        file_type = "markdown" if file_path.suffix.lower() == ".md" else "text"
        logger.info("Parsed %s file '%s' (%d characters).", file_type, file_path.name, len(text))
        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "document_id": document_id,
                    "file_type": file_type,
                },
            )
        ]
