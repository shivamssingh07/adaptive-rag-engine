"""Unit tests for `backend.config.settings`."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from backend.config.settings import Settings
from backend.core.exceptions import ConfigurationError


class TestSettingsValidation:
    def test_missing_groq_api_key_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from backend.config import settings as settings_module

        settings_module.get_settings.cache_clear()
        with pytest.raises(ConfigurationError):
            settings_module.get_settings()
        settings_module.get_settings.cache_clear()

    def test_defaults_are_sane(self, settings: Settings) -> None:
        assert settings.groq_model == "llama-3.3-70b-versatile"
        assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.reranker_model_name == "BAAI/bge-reranker-base"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.max_document_grade_retries == 2
        assert settings.max_groundedness_retries == 2


class TestSettingsComputedProperties:
    def test_tavily_enabled_false_when_unset(self, settings: Settings) -> None:
        assert settings.tavily_enabled is False

    def test_tavily_enabled_true_when_set(self, tmp_path) -> None:
        s = Settings(
            groq_api_key=SecretStr("x"),
            tavily_api_key=SecretStr("some-real-key"),
            faiss_index_dir=tmp_path / "f",
            bm25_index_dir=tmp_path / "b",
            upload_dir=tmp_path / "u",
            session_db_path=tmp_path / "s.db",
            document_registry_db_path=tmp_path / "d.db",
            log_dir=tmp_path / "l",
        )
        assert s.tavily_enabled is True

    def test_hybrid_weights_sum_to_one(self, settings: Settings) -> None:
        assert settings.hybrid_bm25_weight + settings.hybrid_vector_weight == pytest.approx(1.0)

    def test_max_upload_size_bytes_conversion(self, settings: Settings) -> None:
        assert settings.max_upload_size_bytes == settings.max_upload_size_mb * 1024 * 1024

    def test_cors_origins_list_parses_comma_separated(self, tmp_path) -> None:
        s = Settings(
            groq_api_key=SecretStr("x"),
            cors_origins="https://a.com, https://b.com,https://c.com",
            faiss_index_dir=tmp_path / "f",
            bm25_index_dir=tmp_path / "b",
            upload_dir=tmp_path / "u",
            session_db_path=tmp_path / "s.db",
            document_registry_db_path=tmp_path / "d.db",
            log_dir=tmp_path / "l",
        )
        assert s.cors_origins_list == ["https://a.com", "https://b.com", "https://c.com"]

    def test_allowed_extensions_normalized(self, tmp_path) -> None:
        s = Settings(
            groq_api_key=SecretStr("x"),
            allowed_upload_extensions="PDF,.docx, txt",
            faiss_index_dir=tmp_path / "f",
            bm25_index_dir=tmp_path / "b",
            upload_dir=tmp_path / "u",
            session_db_path=tmp_path / "s.db",
            document_registry_db_path=tmp_path / "d.db",
            log_dir=tmp_path / "l",
        )
        assert s.allowed_extensions == frozenset({".pdf", ".docx", ".txt"})

    def test_directories_are_created(self, settings: Settings) -> None:
        assert settings.faiss_index_dir.exists()
        assert settings.bm25_index_dir.exists()
        assert settings.upload_dir.exists()
        assert settings.session_db_path.parent.exists()
        assert settings.document_registry_db_path.parent.exists()
