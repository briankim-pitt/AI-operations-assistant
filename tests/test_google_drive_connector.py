from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.connectors.google_drive import (
    GOOGLE_DOCUMENT_MIME_TYPE,
    GOOGLE_PRESENTATION_MIME_TYPE,
    PDF_MIME_TYPE,
    GoogleDriveConnector,
)


def test_fetches_docs_slides_and_pdf_pages() -> None:
    connector = GoogleDriveConnector.__new__(GoogleDriveConnector)
    connector.folder_id = "folder-id"
    connector.service = MagicMock()
    connector._list_folder_files = MagicMock(
        return_value=[
            _drive_file("doc-id", "Policy", GOOGLE_DOCUMENT_MIME_TYPE),
            _drive_file("slides-id", "Incident Review", GOOGLE_PRESENTATION_MIME_TYPE),
            _drive_file("pdf-id", "Handbook.pdf", PDF_MIME_TYPE),
        ]
    )
    connector._export_workspace_text = MagicMock(
        side_effect=["Document text", "Slide text"]
    )
    connector._download_file = MagicMock(return_value=b"pdf-bytes")

    first_page = MagicMock()
    first_page.extract_text.return_value = "First PDF page"
    empty_page = MagicMock()
    empty_page.extract_text.return_value = ""
    third_page = MagicMock()
    third_page.extract_text.return_value = "Third PDF page"

    with patch(
        "app.connectors.google_drive.PdfReader",
        return_value=MagicMock(pages=[first_page, empty_page, third_page]),
    ):
        documents = connector.fetch_documents()

    assert [document.title for document in documents] == [
        "Policy",
        "Incident Review",
        "Handbook.pdf, page 1",
        "Handbook.pdf, page 3",
    ]
    assert documents[1].metadata["mime_type"] == GOOGLE_PRESENTATION_MIME_TYPE
    assert documents[2].metadata["page_number"] == "1"
    assert documents[2].external_id == "pdf-id#page=1"


def _drive_file(file_id: str, name: str, mime_type: str) -> dict:
    return {
        "id": file_id,
        "name": name,
        "mimeType": mime_type,
        "modifiedTime": datetime(2026, 1, 1, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "webViewLink": f"https://drive.google.com/file/d/{file_id}/view",
    }
