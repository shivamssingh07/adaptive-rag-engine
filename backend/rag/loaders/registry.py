"""Extension-to-loader registry.

Provides a single dispatch point (`get_loader_for`) so the rest of the
ingestion pipeline never needs an `if/elif` chain over file extensions.
"""

from __future__ import annotations

from pathlib import Path

from backend.core.exceptions import UnsupportedFileTypeError
from backend.rag.loaders.base import DocumentLoader
from backend.rag.loaders.csv_loader import CSVLoader
from backend.rag.loaders.docx_loader import DOCXLoader
from backend.rag.loaders.pdf_loader import PDFLoader
from backend.rag.loaders.text_loader import TextLoader

_LOADER_REGISTRY: dict[str, type[DocumentLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
}


def get_loader_for(file_path: Path) -> DocumentLoader:
    """Return a loader instance appropriate for the given file's extension.

    Args:
        file_path: Path to the file to be loaded (only the suffix is
            inspected).

    Returns:
        A freshly-constructed loader instance implementing `DocumentLoader`.

    Raises:
        UnsupportedFileTypeError: If the file's extension has no registered
            loader.
    """
    extension = file_path.suffix.lower()
    loader_cls = _LOADER_REGISTRY.get(extension)
    if loader_cls is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}' for '{file_path.name}'.",
            details={"extension": extension, "supported": sorted(_LOADER_REGISTRY)},
        )
    return loader_cls()


def supported_extensions() -> list[str]:
    """Return the sorted list of all file extensions with a registered loader."""
    return sorted(_LOADER_REGISTRY)
