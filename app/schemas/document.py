"""Belge endpoint'leri için Pydantic şemaları."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    file_size: int
    page_count: int
    status: str  # pending | processing | completed | failed
    error_message: str | None = None
    summary: str | None = None
    doc_metadata: dict[str, Any] | None = None
    summary_status: str = "pending"
    created_at: datetime
    chunk_count: int = 0


class DocumentList(BaseModel):
    total: int
    items: list[DocumentOut]
