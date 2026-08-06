"""Application settings.

All runtime configuration is centralized here and loaded from environment
variables (optionally via a `.env` file — see `.env.example`). This is the
single source of truth for configuration; no other module should call
`os.getenv` directly.

Only `GROQ_API_KEY` is required. Every other setting has a sensible default
so the application boots successfully out of the box.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Strongly-typed, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application identity
    # ------------------------------------------------------------------
    app_name: str = Field(default="Adaptive RAG Engine")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    cors_origins: str = Field(default="*")

    # ------------------------------------------------------------------
    # LLM (Groq)
    # ------------------------------------------------------------------
    groq_api_key: SecretStr = Field(
        ...,
        description=(
            "Required. Free API key from https://console.groq.com/keys. "
            "Copy .env.example to .env and set this value."
        ),
    )
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # ------------------------------------------------------------------
    # Optional web-search fallback
    # ------------------------------------------------------------------
    tavily_api_key: SecretStr | None = Field(default=None)

    # ------------------------------------------------------------------
    # Embeddings & reranking
    # ------------------------------------------------------------------
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Keep the normal default for local development/tests.
    # Render Free can disable it through:
    # RERANKER_MODEL_NAME=
    reranker_model_name: str = Field(default="BAAI/bge-reranker-base")

    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------
    faiss_index_dir: Path = Field(default=Path("data/faiss_index"))
    bm25_index_dir: Path = Field(default=Path("data/bm25_index"))
    upload_dir: Path = Field(default=Path("data/uploads"))
    session_db_path: Path = Field(default=Path("data/sessions.db"))
    document_registry_db_path: Path = Field(default=Path("data/documents.db"))

    # ------------------------------------------------------------------
    # Ingestion / chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_upload_size_mb: int = Field(default=25, gt=0)

    allowed_upload_extensions: str = Field(default=".pdf,.docx,.txt,.md,.csv")

    # ------------------------------------------------------------------
    # Retrieval / reranking tuning
    # ------------------------------------------------------------------
    top_k_retrieval: int = Field(default=10, gt=0)
    top_k_rerank: int = Field(default=4, gt=0)
    hybrid_bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Adaptive graph self-correction limits
    # ------------------------------------------------------------------
    max_document_grade_retries: int = Field(default=2, ge=0)
    max_groundedness_retries: int = Field(default=2, ge=0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(default="INFO")

    log_dir: Path = Field(default=Path("logs"))
    log_json: bool = Field(default=True)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def cors_origins_list(self) -> list[str]:
        """CORS allow-list parsed from comma-separated setting."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_extensions(self) -> frozenset[str]:
        """Allowed upload extensions normalized to lowercase."""
        normalized: set[str] = set()

        for raw_ext in self.allowed_upload_extensions.split(","):
            ext = raw_ext.strip().lower()

            if not ext:
                continue

            if not ext.startswith("."):
                ext = f".{ext}"

            normalized.add(ext)

        return frozenset(normalized)

    @property
    def tavily_enabled(self) -> bool:
        """Whether Tavily web-search fallback is configured."""
        return self.tavily_api_key is not None and bool(
            self.tavily_api_key.get_secret_value().strip()
        )

    @property
    def hybrid_vector_weight(self) -> float:
        """Vector-search weight in hybrid retrieval."""
        return 1.0 - self.hybrid_bm25_weight

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size converted to bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def model_post_init(self, __context: object) -> None:
        """Create required application directories."""
        for directory in (
            self.faiss_index_dir,
            self.bm25_index_dir,
            self.upload_dir,
            self.log_dir,
            self.session_db_path.parent,
            self.document_registry_db_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    try:
        return Settings()  # type: ignore[call-arg]

    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            "Application settings are invalid or incomplete. "
            "Copy .env.example to .env and ensure GROQ_API_KEY is set.",
            details={"validation_error": str(exc)},
        ) from exc
