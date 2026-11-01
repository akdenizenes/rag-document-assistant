"""Sohbet oturumu endpoint'leri için Pydantic şemaları."""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.config import MAX_SCOPE_DOCUMENTS

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class SessionCreate(BaseModel):
    title: Title | None = Field(default=None, description="Boşsa ilk sorudan otomatik üretilir.")
    document_ids: list[int] | None = Field(
        default=None, max_length=MAX_SCOPE_DOCUMENTS, description="Belge kapsamı (None = tüm belgeler)."
    )


class SessionUpdate(BaseModel):
    title: Title


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str  # user | assistant
    content: str
    sources: list[dict[str, Any]] | None = None
    model: str | None = None
    route: str | None = None
    created_at: datetime


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    document_ids: list[int] | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionList(BaseModel):
    total: int
    items: list[SessionOut]


class SessionDetail(SessionOut):
    messages: list[MessageOut] = Field(default_factory=list)
