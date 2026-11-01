"""Soru-cevap (RAG) endpoint'i için Pydantic şemaları."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.config import MAX_SCOPE_DOCUMENTS


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
        description="Tek belge kapsamı (geriye dönük uyumluluk için; `document_ids` tercih edilir).",
    )
    document_ids: list[int] | None = Field(
        default=None,
        max_length=MAX_SCOPE_DOCUMENTS,
        description=(
            "Aramanın yapılacağı belgeler. Boş/verilmezse (ve `document_id` de yoksa) kullanıcının "
            "tüm belgelerinde aranır."
        ),
    )
    session_id: int | None = Field(
        default=None,
        description=(
            "Verilirse konuşma geçmişi sunucudaki oturumdan okunur (`history` yok sayılır) ve "
            "soru + cevap bu oturuma kaydedilir."
        ),
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Oturumsuz kullanım için önceki mesajlar (eskiden yeniye). `session_id` varsa yok sayılır.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Getirilecek parça sayısı. Boşsa sunucu varsayılanı kullanılır.",
    )


    def scope(self) -> list[int] | None:
        """`document_id` ve `document_ids` birleştirilmiş, tekilleştirilmiş belge kapsamı (None = hepsi)."""
        ids: list[int] = []
        for value in [self.document_id, *(self.document_ids or [])]:
            if value is not None and value not in ids:
                ids.append(value)
        return ids or None


class SourceChunk(BaseModel):
    """Cevabın dayandığı kaynak parça."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: int
    document_id: int
    document_name: str
    chunk_index: int
    page_number: int | None = None
    score: float = Field(description="Kosinüs benzerliği (1'e yakın = daha ilgili)")
    content: str
    rrf_score: float | None = Field(
        default=None, description="Hibrit aramada RRF birleştirme skoru (sıralama bununla yapılır)"
    )
    rerank_score: float | None = Field(
        default=None, description="Cross-encoder yeniden sıralama skoru (yalnızca RERANK_MODE=cross_encoder)"
    )
    matched_by: str | None = Field(
        default=None,
        description="Parçanın kaynağı: vector | fts | both (hibrit arama), summary | full_text (özet/tam metin rotaları)",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str = Field(description="Cevabı üreten LLM modeli")
    document_id: int | None = Field(default=None, description="Tek belge kapsamıysa onun id'si")
    document_ids: list[int] | None = Field(default=None, description="Kullanılan belge kapsamı (None = tümü)")
    session_id: int | None = Field(default=None, description="Soru-cevabın kaydedildiği oturum")
    route: Literal["summary", "full_text", "hybrid"] | None = Field(
        default=None, description="Sorunun yanıtlandığı rota (kural tabanlı yönlendirme sonucu)"
    )
