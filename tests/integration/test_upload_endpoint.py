"""Integration tests for `POST /upload`, `GET /documents`,
`DELETE /documents/{id}`, and `POST /reset`."""

from __future__ import annotations

import io


class TestUploadEndpoint:
    def test_upload_single_valid_file(self, api_client) -> None:
        files = [
            (
                "files",
                (
                    "notes.txt",
                    io.BytesIO(b"This is a document about company policy."),
                    "text/plain",
                ),
            )
        ]
        response = api_client.post("/api/v1/upload", files=files)

        assert response.status_code == 200
        body = response.json()
        assert body["total_files"] == 1
        assert body["successful_files"] == 1
        assert body["failed_files"] == 0
        assert body["results"][0]["chunks_added"] > 0

    def test_upload_isolates_per_file_failures(self, api_client) -> None:
        files = [
            ("files", ("good.txt", io.BytesIO(b"Valid content about refunds."), "text/plain")),
            ("files", ("bad.png", io.BytesIO(b"not supported"), "image/png")),
        ]
        response = api_client.post("/api/v1/upload", files=files)

        body = response.json()
        assert body["total_files"] == 2
        assert body["successful_files"] == 1
        assert body["failed_files"] == 1
        results_by_name = {r["filename"]: r for r in body["results"]}
        assert results_by_name["good.txt"]["success"] is True
        assert results_by_name["bad.png"]["success"] is False

    def test_duplicate_upload_is_skipped(self, api_client) -> None:
        content = b"Identical content for duplicate detection."
        files = [("files", ("a.txt", io.BytesIO(content), "text/plain"))]
        api_client.post("/api/v1/upload", files=files)

        files_again = [("files", ("a.txt", io.BytesIO(content), "text/plain"))]
        response = api_client.post("/api/v1/upload", files=files_again)

        result = response.json()["results"][0]
        assert result["duplicate"] is True
        assert result["chunks_added"] == 0


class TestDocumentsEndpoint:
    def test_list_documents_empty(self, api_client) -> None:
        response = api_client.get("/api/v1/documents")
        assert response.status_code == 200
        assert response.json()["total_documents"] == 0

    def test_list_documents_after_upload(self, api_client) -> None:
        files = [
            ("files", ("notes.txt", io.BytesIO(b"Some content here about pricing."), "text/plain"))
        ]
        api_client.post("/api/v1/upload", files=files)

        response = api_client.get("/api/v1/documents")

        body = response.json()
        assert body["total_documents"] == 1
        assert body["documents"][0]["filename"] == "notes.txt"

    def test_delete_document(self, api_client) -> None:
        files = [
            ("files", ("notes.txt", io.BytesIO(b"Content to be deleted later."), "text/plain"))
        ]
        upload_response = api_client.post("/api/v1/upload", files=files)
        document_id = upload_response.json()["results"][0]["document_id"]

        delete_response = api_client.delete(f"/api/v1/documents/{document_id}")

        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"] is True
        assert delete_response.json()["chunks_removed"] >= 1

        list_response = api_client.get("/api/v1/documents")
        assert list_response.json()["total_documents"] == 0

    def test_delete_nonexistent_document_returns_404(self, api_client) -> None:
        response = api_client.delete("/api/v1/documents/doc_does_not_exist")
        assert response.status_code == 404
        assert response.json()["error_code"] == "document_not_found"


class TestResetEndpoint:
    def test_reset_clears_knowledge_base(self, api_client) -> None:
        files = [("files", ("notes.txt", io.BytesIO(b"Content that will be reset."), "text/plain"))]
        api_client.post("/api/v1/upload", files=files)
        assert api_client.get("/api/v1/documents").json()["total_documents"] == 1

        response = api_client.post("/api/v1/reset")

        assert response.status_code == 200
        assert response.json()["documents_removed"] == 1
        assert api_client.get("/api/v1/documents").json()["total_documents"] == 0
