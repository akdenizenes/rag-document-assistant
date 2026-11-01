"""RAG hattı: kural tabanlı yönlendirme (summary | full_text | hybrid) + prepare_query + LLM."""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import LLM_MODEL, RETRIEVAL_TOP_K
from app.models import Document, DocumentStatus
from app.schemas.chat import ChatMessage, RouteType
from app.services.llm_service import Messages, generate
from app.services.prompts import (
    NO_ANSWER,
    build_context,
    build_messages,
    build_retrieval_query,
)
from app.services.retrieval_service import RetrievedChunk, hybrid_search_chunks

logger = logging.getLogger(__name__)

# Adım 2: Türkçe Kural Tabanlı Soru Yönlendirme Kalıpları
_SUMMARY_PATTERNS = (
    r"\bözet(le|i|ini|ler misin|leyebilir misin)?\b",
    r"\bne anlat(ıyor|makta|ılmaktadır|ır)\b",
    r"\bgenel olarak ne(dir|den bahsediyor| anlatıyor)?\b",
    r"\bana (fikir|fikri|konusu|hatları)\b",
    r"\bbelge(nin|lerin)? konusu ne\b",
    r"\bkısaca (anlat|bahset|açıkla)\b",
)

_FULL_TEXT_PATTERNS = (
    r"\bkaç (tane |adet )?(soru|bölüm|konu|başlık|madde|örnek|deney|problem|alıştırma)\b",
    r"\b(bölümleri|konuları|soruları|başlıkları|maddeleri) (listele|sırala|yaz|göster)\b",
    r"\btüm (konular|konuları|sorular|soruları|bölümler|bölümleri|başlıklar|başlıkları)\b",
    r"\btoplam (kaç )?(soru|bölüm|konu|madde)\b",
    r"\bsoru sayısı\b",
    r"\biçindekiler\b",
)


def classify_query_route(question: str) -> RouteType:
    """Kullanıcı sorusunu Türkçe kurallara göre 'summary', 'full_text' veya 'hybrid' olarak sınıflandırır."""
    q = question.lower().strip()

    for pattern in _FULL_TEXT_PATTERNS:
        if re.search(pattern, q):
            return "full_text"

    for pattern in _SUMMARY_PATTERNS:
        if re.search(pattern, q):
            return "summary"

    return "hybrid"


@dataclass(slots=True)
class PreparedQuery:
    route: RouteType
    messages: Messages
    sources: list[RetrievedChunk]
    model: str
    empty_result: bool = False


@dataclass(slots=True)
class RagResult:
    answer: str
    sources: list[RetrievedChunk]
    model: str
    route: RouteType = "hybrid"


async def _fetch_target_documents(
    db: AsyncSession,
    *,
    user_id: int,
    document_id: int | None,
) -> list[Document]:
    stmt = (
        select(Document)
        .where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.COMPLETED.value,
        )
        .order_by(Document.created_at.desc(), Document.id.desc())
    )
    if document_id is not None:
        stmt = stmt.where(Document.id == document_id)
    return list((await db.execute(stmt)).scalars().all())


async def prepare_query(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_id: int | None = None,
    history: Sequence[ChatMessage] = (),
    top_k: int | None = None,
) -> PreparedQuery:
    """Adım 2: Veritabanı kilitlemesini önlemek için LLM akışından önce çalışır.
    Rotayı belirler, kaynakları/bağlamı hazırlar ve DB bağlantısını serbest bırakır.
    """
    route = classify_query_route(question)
    chunks: list[RetrievedChunk] = []
    unlimited_context = False

    if route == "summary":
        docs = await _fetch_target_documents(db, user_id=user_id, document_id=document_id)
        for doc in docs:
            summary_text = doc.summary or (doc.full_text[:8000] if doc.full_text else "")
            if not summary_text:
                continue
            meta_json = json.dumps(doc.doc_metadata or {}, ensure_ascii=False)
            content_block = (
                f"Belge Özeti:\n{summary_text}\n\n"
                f"Belge Metaverisi: {meta_json}"
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=0,
                    document_id=doc.id,
                    document_name=doc.filename,
                    chunk_index=0,
                    page_number=1,
                    content=content_block,
                    score=1.0,
                )
            )
        if chunks:
            unlimited_context = True
        else:
            # Belgenin özeti henüz yoksa hibrit aramaya düş
            route = "hybrid"

    elif route == "full_text":
        docs = await _fetch_target_documents(db, user_id=user_id, document_id=document_id)
        for doc in docs:
            if not doc.full_text:
                continue
            meta_json = json.dumps(doc.doc_metadata or {}, ensure_ascii=False)
            content_block = (
                f"Belge Metaverisi: {meta_json}\n"
                f"Toplam Sayfa: {doc.page_count}\n\n"
                f"{doc.full_text}"
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=0,
                    document_id=doc.id,
                    document_name=doc.filename,
                    chunk_index=0,
                    page_number=doc.page_count or 1,
                    content=content_block,
                    score=1.0,
                )
            )
        if chunks:
            unlimited_context = True
        else:
            route = "hybrid"

    if route == "hybrid":
        chunks = await hybrid_search_chunks(
            db,
            user_id=user_id,
            query=build_retrieval_query(question, history),
            document_id=document_id,
            top_k=top_k or RETRIEVAL_TOP_K,
        )

    # LLM yanıtı saniyeler sürebilir; bu sırada DB bağlantısını havuza geri bırak.
    await db.rollback()

    if not chunks:
        logger.info("İlgili parça bulunamadı (route=%s), standart yanıt dönülecek.", route)
        return PreparedQuery(
            route=route,
            messages=[],
            sources=[],
            model=LLM_MODEL,
            empty_result=True,
        )

    context, used = build_context(chunks, unlimited=unlimited_context)
    used_sources: list[RetrievedChunk] = []
    for c in chunks[:used]:
        # UI tarafında full_text akordeonu çok şişirmesin diye kaynak önizlemesini kısalt
        preview_content = (
            f"[Tam Metin Taraması - {c.page_number} sayfa]\n" + c.content[:600] + "…"
            if route == "full_text" and len(c.content) > 600
            else c.content
        )
        used_sources.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name=c.document_name,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                content=preview_content,
                score=c.score,
            )
        )

    messages = build_messages(question, context, history, route=route)
    logger.info("Sorgu hazırlandı: route=%s, kaynak_sayısı=%d", route, len(used_sources))
    return PreparedQuery(
        route=route,
        messages=messages,
        sources=used_sources,
        model=LLM_MODEL,
        empty_result=False,
    )


async def answer_question(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_id: int | None = None,
    history: Sequence[ChatMessage] = (),
    top_k: int | None = None,
) -> RagResult:
    """Soruyu kullanıcının belgelerine dayanarak yanıtlar (/chat endpoint'i için)."""
    prepared = await prepare_query(
        db,
        user_id=user_id,
        question=question,
        document_id=document_id,
        history=history,
        top_k=top_k,
    )
    if prepared.empty_result:
        return RagResult(
            answer=NO_ANSWER,
            sources=[],
            model=prepared.model,
            route=prepared.route,
        )

    answer = await generate(prepared.messages)
    return RagResult(
        answer=answer,
        sources=prepared.sources,
        model=prepared.model,
        route=prepared.route,
    )
