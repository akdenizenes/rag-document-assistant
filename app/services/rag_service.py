"""RAG hattı: soru -> rota -> bağlam -> prompt -> LLM -> cevap + kaynaklar.

İki aşamalıdır, böylece hem tek seferlik hem akışlı cevap aynı hazırlığı kullanır:
  1. prepare_query  : yönlendirme + (arama | özet | tam metin) + prompt hazırlığı
                      (veritabanı BURADA kullanılır)
  2. LLM çağrısı    : generate / generate_stream (veritabanı kullanılmaz)

Yönlendirme (kural tabanlı, LLM çağrısı yok; bkz. query_router.py):
  summary   -> `documents.summary` doğrudan bağlama verilir, arama yapılmaz
  full_text -> `documents.full_text` bağlama verilir (sayma / genel tarama), arama yapılmaz
  hybrid    -> hibrit arama (pgvector + Türkçe FTS, RRF) + yeniden sıralama / daraltma

Belge kapsamı (`document_ids`): None/boş = kullanıcının tüm belgeleri, dolu = yalnızca o belgeler.

Bir rota için gerekli veri yoksa (özet henüz üretilmemiş, eski belge için tam metin
doldurulmamış vb.) bir sonraki uygun rotaya düşülür; `PreparedQuery.route` gerçekte
kullanılan rotayı gösterir.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    FULL_TEXT_CONTEXT_MAX_CHARS,
    FULL_TEXT_MAX_DOCS,
    HYBRID_SEARCH_ENABLED,
    LLM_MODEL,
    RETRIEVAL_TOP_K,
    ROUTE_SUMMARY_MAX_DOCS,
)
from app.models import Document, DocumentStatus
from app.schemas.chat import ChatMessage
from app.services.llm_service import generate
from app.services.prompts import (
    FULL_TEXT_ROUTE_RULES,
    NO_ANSWER,
    SUMMARY_ROUTE_RULES,
    DocContext,
    build_context,
    build_full_text_context,
    build_messages,
    build_multi_full_text_context,
    build_retrieval_query,
    build_summary_context,
)
from app.services.query_router import (
    ROUTE_FULL_TEXT,
    ROUTE_HYBRID,
    ROUTE_SUMMARY,
    decide_route,
)
from app.services.rerank_service import candidates_needed, rerank_chunks
from app.services.retrieval_service import (
    RetrievedChunk,
    hybrid_search_chunks,
    search_similar_chunks,
)

logger = logging.getLogger(__name__)

TOO_MANY_DOCS_FULL_TEXT_MESSAGE = (
    "Sayma ve listeleme gibi tüm belgeyi tarayan sorularda en fazla {max} belge birlikte "
    "taranabilir (şu anda kapsamda {count} belge var). Sohbet kapsamını daraltıp aynı soruyu "
    "tekrar sorabilirsiniz."
)
FULL_TEXT_TOO_LONG_MESSAGE = (
    "Seçili belgelerin metni birlikte çok uzun olduğu için tam tarama yapılamıyor. "
    "Lütfen kapsamı daha az belgeye indirip tekrar deneyin."
)
# Geriye dönük uyumluluk için eski ad
MULTI_DOC_FULL_TEXT_MESSAGE = TOO_MANY_DOCS_FULL_TEXT_MESSAGE


@dataclass(slots=True)
class PreparedQuery:
    """LLM'e gitmeye hazır sorgu."""

    sources: list[RetrievedChunk]
    # None => LLM çağrılmayacak: `direct_answer` varsa o, yoksa NO_ANSWER dönülür
    messages: list[dict[str, str]] | None
    route: str = ROUTE_HYBRID  # gerçekte kullanılan rota: summary | full_text | hybrid
    direct_answer: str | None = None


@dataclass(slots=True)
class RagResult:
    answer: str
    sources: list[RetrievedChunk]
    model: str
    route: str = ROUTE_HYBRID


# --------------------------------------------------------------------------- #
# Veri yükleme
# --------------------------------------------------------------------------- #
def _in_scope(stmt, user_id: int, document_ids: Sequence[int] | None):
    """Hazır (completed) belgeleri kullanıcıya ve (varsa) seçili belgelere göre sınırlar."""
    stmt = stmt.where(
        Document.user_id == user_id,
        Document.status == DocumentStatus.COMPLETED.value,
    )
    if document_ids:
        stmt = stmt.where(Document.id.in_(list(document_ids)))
    return stmt


async def _load_docs(
    db: AsyncSession,
    *,
    user_id: int,
    document_ids: Sequence[int] | None,
    need_summary: bool = False,
    need_full_text: bool = False,
    limit: int = 10,
) -> list[DocContext]:
    """Kapsamdaki hazır belgeleri yükler. Büyük `full_text` yalnızca istenirse seçilir."""
    columns = [
        Document.id,
        Document.filename,
        Document.page_count,
        Document.summary,
        Document.doc_metadata,
    ]
    if need_full_text:
        columns.append(Document.full_text)

    stmt = _in_scope(select(*columns), user_id, document_ids)
    if need_summary:
        stmt = stmt.where(Document.summary.is_not(None))
    if need_full_text:
        stmt = stmt.where(Document.full_text.is_not(None))
    stmt = stmt.order_by(Document.created_at.desc(), Document.id.desc()).limit(limit)

    return [
        DocContext(
            document_id=row.id,
            filename=row.filename,
            page_count=row.page_count,
            summary=row.summary,
            doc_metadata=row.doc_metadata,
            full_text=row.full_text if need_full_text else None,
        )
        for row in (await db.execute(stmt)).all()
    ]


def _doc_source(doc: DocContext, *, content: str, matched_by: str) -> RetrievedChunk:
    """Özet / tam metin rotalarında arayüzde gösterilecek, belgeyi temsil eden kaynak kaydı."""
    return RetrievedChunk(
        chunk_id=0,
        document_id=doc.document_id,
        document_name=doc.filename,
        chunk_index=-1,
        page_number=None,
        content=content,
        score=1.0,
        matched_by=matched_by,
    )


# --------------------------------------------------------------------------- #
# Rotalar
# --------------------------------------------------------------------------- #
async def _prepare_summary(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_ids: Sequence[int] | None,
    history: Sequence[ChatMessage],
) -> PreparedQuery | None:
    """Özet rotası. None döner => uygun veri yok, çağıran bir sonraki rotayı dener."""
    docs = await _load_docs(
        db,
        user_id=user_id,
        document_ids=document_ids,
        need_summary=True,
        limit=ROUTE_SUMMARY_MAX_DOCS,
    )
    if not docs:
        logger.info("Özet rotası: kayıtlı özet yok, geri dönüş yapılacak.")
        if document_ids:  # belgeler seçili, özet hazır değil ama tam metin olabilir
            return await _prepare_full_text(
                db,
                user_id=user_id,
                question=question,
                document_ids=document_ids,
                history=history,
            )
        return None

    return PreparedQuery(
        sources=[
            _doc_source(d, content=d.summary or "", matched_by=ROUTE_SUMMARY) for d in docs
        ],
        messages=build_messages(
            question, build_summary_context(docs), history, extra_rules=SUMMARY_ROUTE_RULES
        ),
        route=ROUTE_SUMMARY,
    )


async def _prepare_full_text(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_ids: Sequence[int] | None,
    history: Sequence[ChatMessage],
) -> PreparedQuery | None:
    """Tam metin rotası (sayma / genel tarama). None döner => hibrit aramaya düşülür.

    Kapsamda en fazla FULL_TEXT_MAX_DOCS belge varsa hepsinin tam metni birlikte bağlama verilir;
    daha fazlaysa (ya da metinler birlikte çok uzunsa) kullanıcıdan kapsamı daraltması istenir.
    """
    ready = await db.scalar(
        _in_scope(select(func.count()).select_from(Document), user_id, document_ids)
    )
    if not ready:
        return None
    if ready > FULL_TEXT_MAX_DOCS:
        logger.info("Tam metin rotası: kapsamda %d belge var (sınır %d).", ready, FULL_TEXT_MAX_DOCS)
        return PreparedQuery(
            sources=[],
            messages=None,
            route=ROUTE_FULL_TEXT,
            direct_answer=TOO_MANY_DOCS_FULL_TEXT_MESSAGE.format(
                count=ready, max=FULL_TEXT_MAX_DOCS
            ),
        )

    docs = await _load_docs(
        db,
        user_id=user_id,
        document_ids=document_ids,
        need_full_text=True,
        limit=FULL_TEXT_MAX_DOCS,
    )
    if len(docs) < ready:
        # Bazı belgelerin full_text'i yok (backfill yapılmamış): eksik sayım vermektense aramaya düş.
        logger.warning(
            "Tam metin rotası: %d belgeden %d tanesinin full_text'i var "
            "(scripts/backfill_metadata.py çalıştırılmamış olabilir), hibrit aramaya düşülüyor.",
            ready,
            len(docs),
        )
        return None

    if len(docs) == 1:
        doc = docs[0]
        context, truncated = build_full_text_context(doc, FULL_TEXT_CONTEXT_MAX_CHARS)
        note = " (bağlam sınırı nedeniyle kısaltıldı)" if truncated else ""
        sources = [
            _doc_source(
                doc,
                content=f"Belgenin tam metni bağlama verildi: {doc.page_count} sayfa, "
                f"{len(doc.full_text or ''):,} karakter{note}.",
                matched_by=ROUTE_FULL_TEXT,
            )
        ]
    else:
        context, fits = build_multi_full_text_context(docs, FULL_TEXT_CONTEXT_MAX_CHARS)
        if not fits:
            return PreparedQuery(
                sources=[],
                messages=None,
                route=ROUTE_FULL_TEXT,
                direct_answer=FULL_TEXT_TOO_LONG_MESSAGE,
            )
        sources = [
            _doc_source(
                d,
                content=f"Belgenin tam metni bağlama verildi: {d.page_count} sayfa, "
                f"{len(d.full_text or ''):,} karakter.",
                matched_by=ROUTE_FULL_TEXT,
            )
            for d in docs
        ]

    return PreparedQuery(
        sources=sources,
        messages=build_messages(question, context, history, extra_rules=FULL_TEXT_ROUTE_RULES),
        route=ROUTE_FULL_TEXT,
    )


async def _prepare_hybrid(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_ids: Sequence[int] | None,
    history: Sequence[ChatMessage],
    top_k: int | None,
) -> PreparedQuery:
    """Varsayılan rota: hibrit arama + yeniden sıralama/daraltma; sonuç yoksa LLM çağrılmaz."""
    top = top_k or RETRIEVAL_TOP_K
    search = hybrid_search_chunks if HYBRID_SEARCH_ENABLED else search_similar_chunks
    retrieval_query = build_retrieval_query(question, history)
    chunks = await search(
        db,
        user_id=user_id,
        query=retrieval_query,
        document_ids=document_ids,
        top_k=candidates_needed(top),
    )
    # Arama bitti: parçalar düz nesnelerdir, yeniden sıralama (CPU) sırasında DB bağlantısını bırak.
    await db.rollback()

    # Sorgunun sonu (güncel soru) korunur: cross-encoder (sorgu + parça) uzunluğuna duyarlıdır.
    chunks = await rerank_chunks(retrieval_query[-400:], chunks, top_k=top)

    if not chunks:
        logger.info("İlgili parça bulunamadı, LLM çağrılmadan standart yanıt dönülecek.")
        return PreparedQuery(sources=[], messages=None, route=ROUTE_HYBRID)

    context, used = build_context(chunks)
    return PreparedQuery(
        sources=chunks[:used],
        messages=build_messages(question, context, history),
        route=ROUTE_HYBRID,
    )


async def prepare_query(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_ids: Sequence[int] | None = None,
    history: Sequence[ChatMessage] = (),
    top_k: int | None = None,
) -> PreparedQuery:
    """Soruyu yönlendirir, gerekli bağlamı toplar ve LLM'e gidecek mesajları hazırlar."""
    document_ids = list(document_ids) if document_ids else None
    decision = decide_route(question)
    logger.info(
        "Soru yönlendirildi: rota=%s (kural: %s), kapsam=%s",
        decision.route,
        decision.reason,
        document_ids or "tüm belgeler",
    )

    try:
        prepared: PreparedQuery | None = None
        if decision.route == ROUTE_SUMMARY:
            prepared = await _prepare_summary(
                db,
                user_id=user_id,
                question=question,
                document_ids=document_ids,
                history=history,
            )
        elif decision.route == ROUTE_FULL_TEXT:
            prepared = await _prepare_full_text(
                db,
                user_id=user_id,
                question=question,
                document_ids=document_ids,
                history=history,
            )
        if prepared is None:
            prepared = await _prepare_hybrid(
                db,
                user_id=user_id,
                question=question,
                document_ids=document_ids,
                history=history,
                top_k=top_k,
            )
    finally:
        # LLM yanıtı saniyeler sürebilir; bu sırada DB bağlantısını havuza geri bırak.
        await db.rollback()

    if prepared.route != decision.route:
        logger.info("Rota değişti: %s -> %s (veri yoktu).", decision.route, prepared.route)
    return prepared


async def answer_question(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_ids: Sequence[int] | None = None,
    history: Sequence[ChatMessage] = (),
    top_k: int | None = None,
) -> RagResult:
    """Soruyu kullanıcının belgelerine dayanarak yanıtlar (cevap tek seferde döner)."""
    prepared = await prepare_query(
        db,
        user_id=user_id,
        question=question,
        document_ids=document_ids,
        history=history,
        top_k=top_k,
    )
    if prepared.messages is None:
        return RagResult(
            answer=prepared.direct_answer or NO_ANSWER,
            sources=[],
            model=LLM_MODEL,
            route=prepared.route,
        )

    answer = await generate(prepared.messages)
    return RagResult(
        answer=answer, sources=prepared.sources, model=LLM_MODEL, route=prepared.route
    )
