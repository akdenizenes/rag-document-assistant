"""RAG soru-cevap endpoint'leri.

  POST /chat         -> cevap tek seferde (JSON)
  POST /chat/stream  -> cevap parça parça (Server-Sent Events)

Kapsam: `document_ids` (çoklu seçim) ya da eski `document_id`; ikisi de yoksa tüm belgeler.
Geçmiş: `session_id` verilirse konuşma geçmişi sunucudaki oturumdan okunur ve soru + cevap
o oturuma kaydedilir; verilmezse istemcinin gönderdiği `history` kullanılır (oturumsuz mod).

/chat/stream olay akışı (her olay: `event: <ad>` + `data: <tek satır JSON>`):

  sources : {"sources": [...], "model": "...", "document_id": 1 | null, "document_ids": [..] | null,
             "session_id": 7 | null, "route": "summary" | "full_text" | "hybrid"}  (her zaman ilk olay)
  token   : {"text": "..."}                                          (0..n kez)
  done    : {}                                                       (başarılı bitiş; oturum kaydı bundan ÖNCE yapılır)
  error   : {"detail": "...", "status": 503}                         (akış sırasında hata)

Kaynaklar akışın BAŞINDA gönderilir: arama cevap üretiminden önce biter, bu yüzden
kaynaklar zaten bellidir ve akış yarıda kesilse bile istemcide kaybolmaz.
Belge/oturum bulunamadı (404), belge henüz hazır değil (409), istek sınırı aşıldı (429) gibi
hatalar akış başlamadan, normal HTTP durum koduyla döner. Akış yarıda kesilirse (hata ya da
istemci bağlantıyı kapatırsa) soru-cevap oturuma KAYDEDİLMEZ.

`route`, sorunun hangi yoldan yanıtlandığını söyler (kural tabanlı yönlendirme):
  summary   : belge özeti bağlama verildi (arama yok)
  full_text : belgenin tam metni bağlama verildi (sayma / genel tarama)
  hybrid    : hibrit arama (pgvector + Türkçe tam metin araması, RRF) + yeniden sıralama
"""

import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import CHAT_HISTORY_DB_LIMIT, LLM_MODEL
from app.database import get_db
from app.dependencies import get_current_user
from app.models import Document, DocumentStatus, User
from app.rate_limit import limit_chat
from app.schemas.chat import ChatMessage, QueryRequest, QueryResponse, SourceChunk
from app.services.citations import build_sources
from app.services.llm_service import LLMError, generate_stream
from app.services.prompts import NO_ANSWER
from app.services.rag_service import PreparedQuery, answer_question, prepare_query
from app.services.retrieval_service import RetrievedChunk
from app.services.session_service import get_owned_session, load_history, save_exchange

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@dataclass(slots=True)
class ChatContext:
    """Bir isteğin çözümlenmiş kapsamı, oturumu ve konuşma geçmişi."""

    scope: list[int] | None  # None = tüm belgeler
    session_id: int | None
    history: Sequence[ChatMessage]

    @property
    def single_document_id(self) -> int | None:
        """Eski istemciler için: kapsam tek belgeyse onun id'si."""
        return self.scope[0] if self.scope and len(self.scope) == 1 else None


async def _ensure_queryable(db: AsyncSession, user_id: int, document_ids: list[int] | None) -> None:
    """Seçili belgelerin var, kullanıcıya ait ve hazır olduğunu doğrular."""
    if not document_ids:
        return
    rows = (
        await db.execute(
            select(Document.id, Document.filename, Document.status).where(
                Document.user_id == user_id, Document.id.in_(document_ids)
            )
        )
    ).all()

    found = {row.id for row in rows}
    missing = [doc_id for doc_id in document_ids if doc_id not in found]
    if missing:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Belge bulunamadı." if len(document_ids) == 1 else f"Belge bulunamadı (id: {missing}).",
        )
    not_ready = [r for r in rows if r.status != DocumentStatus.COMPLETED.value]
    if not_ready:
        names = ", ".join(f"{r.filename} ({r.status})" for r in not_ready)
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Belge henüz sorgulanamaz: {names}"
            if len(not_ready) == 1
            else f"Şu belgeler henüz sorgulanamaz: {names}",
        )


async def _resolve_context(db: AsyncSession, user_id: int, payload: QueryRequest) -> ChatContext:
    """Kapsamı doğrular; `session_id` varsa oturumu bulur ve geçmişi veritabanından okur."""
    scope = payload.scope()
    await _ensure_queryable(db, user_id, scope)

    if payload.session_id is None:
        return ChatContext(scope=scope, session_id=None, history=payload.history)

    chat_session = await get_owned_session(db, user_id, payload.session_id)
    if chat_session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sohbet bulunamadı.")
    history = await load_history(db, chat_session.id, CHAT_HISTORY_DB_LIMIT)
    return ChatContext(scope=scope, session_id=chat_session.id, history=history)


def _serialize_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Parçaları `ref` (atıf no) + `snippet` (kısa özet) ile birlikte JSON'a çevirir."""
    return build_sources(chunks)


@router.post("/chat", response_model=QueryResponse, dependencies=[Depends(limit_chat)])
async def chat(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryResponse:
    """Belge(ler)e dayanarak soruyu yanıtlar.

    - `document_ids` (ya da `document_id`) verilirse yalnızca o belgelerde, verilmezse tüm belgelerde aranır.
    - `session_id` verilirse geçmiş sunucudan okunur ve soru-cevap oturuma kaydedilir;
      verilmezse `history` ile önceki mesajlar gönderilerek takip soruları sorulabilir.
    - Yanıtla birlikte dayanılan kaynak parçalar (`sources`) döner: her biri `ref` (cevaptaki [n]
      atfıyla eşleşir), belge adı, chunk id, sayfa/bölüm no, skor, `snippet` (kısa özet) ve tam `content` içerir.
    """
    user_id = user.id  # answer_question içinde oturum sıfırlanacağı için önceden al
    ctx = await _resolve_context(db, user_id, payload)

    try:
        result = await answer_question(
            db,
            user_id=user_id,
            question=payload.question,
            document_ids=ctx.scope,
            history=ctx.history,
            top_k=payload.top_k,
        )
    except LLMError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc

    sources = _serialize_sources(result.sources)
    if ctx.session_id is not None:
        await save_exchange(
            ctx.session_id,
            question=payload.question,
            answer=result.answer,
            sources=sources,
            model=result.model,
            route=result.route,
            document_ids=ctx.scope,
        )

    return QueryResponse(
        answer=result.answer,
        sources=[SourceChunk.model_validate(s) for s in sources],
        model=result.model,
        document_id=ctx.single_document_id,
        document_ids=ctx.scope,
        session_id=ctx.session_id,
        route=result.route,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    """Tek bir SSE olayı üretir. json.dumps satır sonlarını kaçışladığı için data tek satırdır."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _event_stream(
    prepared: PreparedQuery, ctx: ChatContext, question: str
) -> AsyncIterator[str]:
    """Hazırlanmış sorguyu LLM'e gönderir ve SSE olayları üretir.

    LLM beklenirken veritabanı kullanılmaz (arama zaten bitti). Akış başarıyla bittiyse ve
    oturum varsa soru-cevap, `done` olayından ÖNCE kendi kısa DB oturumuyla kaydedilir.
    """
    sources = _serialize_sources(prepared.sources)
    yield _sse(
        "sources",
        {
            "sources": sources,
            "model": LLM_MODEL,
            "document_id": ctx.single_document_id,
            "document_ids": ctx.scope,
            "session_id": ctx.session_id,
            "route": prepared.route,
        },
    )

    parts: list[str] = []
    if prepared.messages is None:
        text = prepared.direct_answer or NO_ANSWER
        parts.append(text)
        yield _sse("token", {"text": text})
    else:
        try:
            # aclosing: istemci bağlantıyı keserse LLM akışı da hemen kapatılır (çöp toplayıcıya kalmaz).
            async with aclosing(generate_stream(prepared.messages)) as pieces:
                async for piece in pieces:
                    parts.append(piece)
                    yield _sse("token", {"text": piece})
        except LLMError as exc:
            yield _sse("error", {"detail": str(exc), "status": exc.status_code})
            return
        except Exception:
            logger.exception("Akış sırasında beklenmeyen hata.")
            yield _sse(
                "error",
                {"detail": "Cevap üretilirken beklenmeyen bir hata oluştu.", "status": 500},
            )
            return

    answer = "".join(parts)
    if ctx.session_id is not None and answer.strip():
        # Hata fırlatmaz (save_exchange içinde yakalanır): kayıt başarısız olsa da cevap iletilmiştir.
        await save_exchange(
            ctx.session_id,
            question=question,
            answer=answer,
            sources=sources,
            model=LLM_MODEL,
            route=prepared.route,
            document_ids=ctx.scope,
        )
    yield _sse("done", {})


@router.post("/chat/stream", dependencies=[Depends(limit_chat)])
async def chat_stream(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """`/chat` ile aynı girdiyi alır, cevabı Server-Sent Events olarak parça parça döner."""
    user_id = user.id
    ctx = await _resolve_context(db, user_id, payload)

    # Arama ve prompt hazırlığı akıştan ÖNCE yapılır: DB oturumu bu noktada serbest kalır,
    # akış sırasında (saniyeler sürebilir) bağlantı tutulmaz.
    prepared = await prepare_query(
        db,
        user_id=user_id,
        question=payload.question,
        document_ids=ctx.scope,
        history=ctx.history,
        top_k=payload.top_k,
    )

    return StreamingResponse(
        _event_stream(prepared, ctx, payload.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx gibi ara katmanların akışı tamponlamasını engeller
        },
    )
