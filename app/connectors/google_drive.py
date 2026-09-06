from datetime import datetime
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.models import SourceDocument

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_DOCUMENT_MIME_TYPE = "application/vnd.google-apps.document"


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
            if file["mimeType"] != GOOGLE_DOCUMENT_MIME_TYPE:
                continue

            updated_at = datetime.fromisoformat(
                file["modifiedTime"].replace("Z", "+00:00")
            )

            if since is not None and updated_at <= since:
                continue

            content = self._export_google_doc(file["id"])

            documents.append(
                SourceDocument(
                    external_id=file["id"],
                    provider="google_drive",
                    title=file["name"],
                    content=content.strip(),
                    source_url=file.get("webViewLink", ""),
                    updated_at=updated_at,
                    metadata={"mime_type": file["mimeType"]},
                )
            )

        return documents

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

    def _export_google_doc(self, file_id: str) -> str:
        content = (
            self.service.files()
            .export(
                fileId=file_id,
                mimeType="text/plain",
            )
            .execute()
        )

        return content.decode("utf-8")