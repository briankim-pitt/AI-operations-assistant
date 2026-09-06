from datetime import datetime
from typing import Protocol

from app.models import SourceDocument


class DocumentConnector(Protocol):
    def fetch_documents(
        self,
        since: datetime | None = None,
    ) -> list[SourceDocument]:
        ...