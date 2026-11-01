"""RAG soru-cevap endpoint'leri (/chat ve SSE /chat/stream)."""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models import Document, DocumentStatus, User
from app.schemas.chat import QueryRequest, QueryResponse, SourceChunk
from app.services.llm_service import LLMError, generate_stream
from app.services.prompts import NO_ANSWER
from app.services.rag_service import answer_question, prepare_query

router = APIRouter(tags=["chat"])


async def _validate_target_document(
    db: AsyncSession, user_id: int, document_id: int | None
) -> None:
    if document_id is not None:
        doc = await db.scalar(
            select(Document).where(
                Document.id == document_id, Document.user_id == user_id
            )
        )
        if doc is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Belge bulunamadı.")
        if doc.status != DocumentStatus.COMPLETED.value:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Belge henüz sorgulanamaz (durum: {doc.status}).",
            )


@router.post("/chat", response_model=QueryResponse)
async def chat(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryResponse:
    """Belge(ler)e dayanarak soruyu tek seferde yanıtlar."""
    user_id = user.id
    await _validate_target_document(db, user_id, payload.document_id)

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
        route=result.route,
        document_id=payload.document_id,
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """SSE (Server-Sent Events) ile canlı cevap akışı.

    - `prepare_query` ile veritabanı sorguları akış başlamadan tamamlanır ve bağlantı serbest bırakılır.
    - İlk SSE olayında (`sources`) kullanılan kaynaklar ve seçilen rota (`route: "summary" | "full_text" | "hybrid"`) iletilir.
    """
    user_id = user.id
    await _validate_target_document(db, user_id, payload.document_id)

    prepared = await prepare_query(
        db,
        user_id=user_id,
        question=payload.question,
        document_id=payload.document_id,
        history=payload.history,
        top_k=payload.top_k,
    )

    sources_out = [
        SourceChunk.model_validate(c, from_attributes=True).model_dump()
        for c in prepared.sources
    ]

    async def event_stream() -> AsyncGenerator[str, None]:
        # 1. İlk olay: Kaynaklar + seçilen rota (route) + model metaverisi
        sources_event = {
            "type": "sources",
            "route": prepared.route,
            "model": prepared.model,
            "sources": sources_out,
            "metadata": {
                "route": prepared.route,
                "model": prepared.model,
                "document_id": payload.document_id,
            },
        }
        yield f"event: sources\ndata: {json.dumps(sources_event, ensure_ascii=False)}\n\n"

        # 2. İlgili parça bulunamadıysa doğrudan NO_ANSWER akıt
        if prepared.empty_result:
            token_event = {"type": "token", "content": NO_ANSWER}
            yield f"event: token\ndata: {json.dumps(token_event, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return

        # 3. LLM canlı metin akışı
        try:
            async for chunk in generate_stream(prepared.messages):
                token_event = {"type": "token", "content": chunk}
                yield f"event: token\ndata: {json.dumps(token_event, ensure_ascii=False)}\n\n"
        except LLMError as exc:
            err_event = {"type": "error", "detail": str(exc)}
            yield f"event: error\ndata: {json.dumps(err_event, ensure_ascii=False)}\n\n"
            return

        yield f"event: done\ndata: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
