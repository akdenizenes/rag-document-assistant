"""Soru-cevap (RAG) endpoint'i için Pydantic şemaları."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

RouteType = Literal["summary", "full_text", "hybrid"]


class ChatMessage(BaseModel):
    """Konuşma geçmişindeki tek bir mesaj."""

    role: Literal["user", "assistant"]
    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Bu belgede teslim süresi kaç gün olarak belirtilmiş?",
                "document_id": 1,
                "history": [],
                "top_k": 5,
            }
        }
    )

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    document_id: int | None = Field(
        default=None,
        description="Verilirse yalnızca bu belgede arar; verilmezse kullanıcının tüm belgelerinde arar.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Önceki mesajlar (eskiden yeniye). Son birkaçı prompt'a eklenir.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Getirilecek parça sayısı. Boşsa sunucu varsayılanı kullanılır.",
    )


class SourceChunk(BaseModel):
    """Cevabın dayandığı kaynak parça."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: int
    document_id: int
    document_name: str
    chunk_index: int
    page_number: int | None = None
    score: float = Field(description="Benzerlik veya RRF skoru")
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str = Field(description="Cevabı üreten LLM modeli")
    route: RouteType = Field(
        default="hybrid",
        description="Sorgunun yönlendirildiği rota: summary | full_text | hybrid",
    )
    document_id: int | None = None
