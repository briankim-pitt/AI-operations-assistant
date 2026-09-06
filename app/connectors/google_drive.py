from datetime import datetime
from io import BytesIO
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from pypdf import PdfReader

from app.models import SourceDocument

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_DOCUMENT_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_PRESENTATION_MIME_TYPE = "application/vnd.google-apps.presentation"
PDF_MIME_TYPE = "application/pdf"
TEXT_MIME_TYPE = "text/plain"


class GoogleDriveConnector:
    def __init__(self, credentials_path: Path, folder_id: str) -> None:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=[DRIVE_READONLY_SCOPE],
        )

        self.service = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )
        self.folder_id = folder_id

    def fetch_documents(
        self,
        since: datetime | None = None,
    ) -> list[SourceDocument]:
        files = self._list_folder_files()
        documents = []

        for file in files:
            updated_at = datetime.fromisoformat(
                file["modifiedTime"].replace("Z", "+00:00")
            )

            if since is not None and updated_at <= since:
                continue

            if file["mimeType"] in {
                GOOGLE_DOCUMENT_MIME_TYPE,
                GOOGLE_PRESENTATION_MIME_TYPE,
            }:
                content = self._export_workspace_text(file["id"])
                if content.strip():
                    documents.append(
                        self._source_document(
                            file=file,
                            content=content,
                            updated_at=updated_at,
                        )
                    )
            elif file["mimeType"] == PDF_MIME_TYPE:
                pdf_content = self._download_file(file["id"])
                documents.extend(
                    self._pdf_page_documents(
                        file=file,
                        content=pdf_content,
                        updated_at=updated_at,
                    )
                )

        return documents

    def _source_document(
        self,
        file: dict,
        content: str,
        updated_at: datetime,
        *,
        title: str | None = None,
        external_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SourceDocument:
        document_metadata = {"mime_type": file["mimeType"]}
        if metadata:
            document_metadata.update(metadata)

        return SourceDocument(
            external_id=external_id or file["id"],
            provider="google_drive",
            title=title or file["name"],
            content=content.strip(),
            source_url=file.get("webViewLink", ""),
            updated_at=updated_at,
            metadata=document_metadata,
        )

    def _list_folder_files(self) -> list[dict]:
        response = (
            self.service.files()
            .list(
                q=f"'{self.folder_id}' in parents and trashed = false",
                fields=(
                    "files(id,name,mimeType,modifiedTime,webViewLink)"
                ),
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        return response.get("files", [])

    def _export_workspace_text(self, file_id: str) -> str:
        content = (
            self.service.files()
            .export(
                fileId=file_id,
                mimeType=TEXT_MIME_TYPE,
            )
            .execute()
        )

        return content.decode("utf-8")

    def _download_file(self, file_id: str) -> bytes:
        return self.service.files().get_media(fileId=file_id).execute()

    def _pdf_page_documents(
        self,
        file: dict,
        content: bytes,
        updated_at: datetime,
    ) -> list[SourceDocument]:
        reader = PdfReader(BytesIO(content))
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text().strip()
            if not page_text:
                continue

            documents.append(
                self._source_document(
                    file=file,
                    content=page_text,
                    updated_at=updated_at,
                    title=f"{file['name']}, page {page_number}",
                    external_id=f"{file['id']}#page={page_number}",
                    metadata={"page_number": str(page_number)},
                )
            )

        return documents
