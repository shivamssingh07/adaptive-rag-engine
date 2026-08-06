"""Unit tests for `backend.rag.rerankers.cross_encoder_reranker`.

The real `CrossEncoder` model is never loaded — `_get_model` is
monkeypatched to return a fake object exposing the same `.predict(pairs)`
interface, so these tests exercise the wrapper's own scoring/sorting/
truncation logic in isolation.
"""

from __future__ import annotations

from langchain_core.documents import Document

from backend.config.settings import Settings
from backend.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker


class _FakeCrossEncoderModel:
    """Mimics `sentence_transformers.CrossEncoder`'s `.predict()` interface."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._scores[: len(pairs)]


class TestCrossEncoderReranker:
    def test_empty_input_returns_empty_list(self, settings: Settings) -> None:
        reranker = CrossEncoderReranker(settings)
        assert reranker.rerank("query", []) == []

    def test_sorts_by_descending_score(self, settings: Settings, monkeypatch) -> None:
        reranker = CrossEncoderReranker(settings)
        docs = [
            Document(page_content="low relevance"),
            Document(page_content="high relevance"),
            Document(page_content="medium relevance"),
        ]
        # Scores intentionally out of order relative to `docs`.
        monkeypatch.setattr(reranker, "_get_model", lambda: _FakeCrossEncoderModel([0.1, 0.9, 0.5]))

        results = reranker.rerank("query", docs)

        assert [doc.page_content for doc, _ in results] == [
            "high relevance",
            "medium relevance",
            "low relevance",
        ]
        assert results[0][1] == 0.9

    def test_truncates_to_top_k(self, settings: Settings, monkeypatch) -> None:
        reranker = CrossEncoderReranker(settings)
        docs = [Document(page_content=f"doc {i}") for i in range(5)]
        monkeypatch.setattr(
            reranker, "_get_model", lambda: _FakeCrossEncoderModel([0.5, 0.9, 0.1, 0.7, 0.3])
        )

        results = reranker.rerank("query", docs, top_k=2)

        assert len(results) == 2
        assert results[0][1] == 0.9
        assert results[1][1] == 0.7

    def test_defaults_top_k_to_settings_value(self, settings: Settings, monkeypatch) -> None:
        reranker = CrossEncoderReranker(settings)
        docs = [Document(page_content=f"doc {i}") for i in range(10)]
        monkeypatch.setattr(reranker, "_get_model", lambda: _FakeCrossEncoderModel([0.5] * 10))

        results = reranker.rerank("query", docs)

        assert len(results) == settings.top_k_rerank
