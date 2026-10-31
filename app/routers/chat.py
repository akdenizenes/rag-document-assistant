"""RAG soru-cevap endpoint'i."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models import Document, DocumentStatus, User
from app.schemas.chat import QueryRequest, QueryResponse, SourceChunk
from app.services.llm_service import LLMError
from app.services.rag_service import answer_question

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=QueryResponse)
async def chat(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryResponse:
    """Belge(ler)e dayanarak soruyu yanıtlar.

    - `document_id` verilirse yalnızca o belgede, verilmezse tüm belgelerde aranır.
    - `history` ile önceki mesajlar gönderilerek takip soruları sorulabilir.
    - Yanıtla birlikte dayanılan kaynak parçalar (`sources`) döner.
    """
    user_id = user.id  # answer_question içinde oturum sıfırlanacağı için önceden al

    if payload.document_id is not None:
        doc = await db.scalar(
            select(Document).where(
                Document.id == payload.document_id, Document.user_id == user_id
            )
        )
        if doc is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Belge bulunamadı.")
        if doc.status != DocumentStatus.COMPLETED.value:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Belge henüz sorgulanamaz (durum: {doc.status}).",
            )

    try:
        result = await answer_question(
            db,
            user_id=user_id,
            question=payload.question,
            document_id=payload.document_id,
            history=payload.history,
            top_k=payload.top_k,
        )
    except LLMError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        sources=[SourceChunk.model_validate(c, from_attributes=True) for c in result.sources],
        model=result.model,
        document_id=payload.document_id,
    )
