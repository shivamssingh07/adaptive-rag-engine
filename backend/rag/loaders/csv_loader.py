"""CSV document loader, backed by `pandas`.

Each row is converted into its own `Document`, formatted as
`"column: value"` lines — this keeps individual rows small enough to be
precisely retrievable (rather than dumping an entire spreadsheet into one
oversized chunk) while preserving column context that a raw comma-joined
row would lose.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from backend.core.exceptions import CorruptedFileError
from backend.utils.ids import generate_document_id

logger = logging.getLogger(__name__)


class CSVLoader:
    """Loads a `.csv` file into one `Document` per row."""

    def load(self, file_path: Path) -> list[Document]:
        """Parse a CSV file, one `Document` per non-empty row.

        Args:
            file_path: Path to the `.csv` file.

        Returns:
            One `Document` per row that has at least one non-null cell.

        Raises:
            CorruptedFileError: If the file cannot be parsed as CSV, has no
                rows, or every row is empty.
        """
        document_id = generate_document_id()

        try:
            frame = pd.read_csv(file_path)
        except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
            raise CorruptedFileError(
                f"Could not parse CSV '{file_path.name}': {exc}",
                details={"file": file_path.name},
            ) from exc

        if frame.empty:
            raise CorruptedFileError(
                f"CSV '{file_path.name}' contains no rows.",
                details={"file": file_path.name},
            )

        columns = list(frame.columns)
        documents: list[Document] = []
        for row_index, row in frame.iterrows():
            row_lines = [f"{column}: {row[column]}" for column in columns if pd.notna(row[column])]
            if not row_lines:
                continue
            documents.append(
                Document(
                    page_content="\n".join(row_lines),
                    metadata={
                        "source": file_path.name,
                        "document_id": document_id,
                        "row": int(row_index) + 1,
                        "file_type": "csv",
                    },
                )
            )

        if not documents:
            raise CorruptedFileError(
                f"CSV '{file_path.name}' contains no non-empty rows.",
                details={"file": file_path.name},
            )

        logger.info("Parsed CSV '%s': %d row(s).", file_path.name, len(documents))
        return documents
