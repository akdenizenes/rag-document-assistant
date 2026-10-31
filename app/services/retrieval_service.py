"""Vektör benzerlik araması (pgvector, kosinüs mesafesi)."""

import logging
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import HNSW_EF_SEARCH, RETRIEVAL_MIN_SCORE, RETRIEVAL_TOP_K
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.embedding_service import embed_texts_async

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_name: str
    chunk_index: int
    page_number: int | None
    content: str
    score: float  # kosinüs benzerliği = 1 - kosinüs mesafesi


async def search_similar_chunks(
    db: AsyncSession,
    *,
    user_id: int,
    query: str,
    document_id: int | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
) -> list[RetrievedChunk]:
    """Sorguyu embed eder ve kullanıcının belgelerindeki en yakın `top_k` parçayı döner.

    - Yalnızca `completed` durumundaki belgeler aranır.
    - `document_id` verilirse arama o belgeyle sınırlanır.
    - `min_score` altındaki parçalar elenir.
    """
    [query_vector] = await embed_texts_async([query])

    distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            Document.filename,
            DocumentChunk.chunk_index,
            DocumentChunk.page_number,
            DocumentChunk.content,
            distance,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.COMPLETED.value,
        )
        .order_by(distance)
        .limit(top_k)
    )
    if document_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == document_id)

    # HNSW indeksinin bu sorgu için baktığı aday sayısı (yalnızca bu transaction için).
    await db.execute(text(f"SET LOCAL hnsw.ef_search = {int(HNSW_EF_SEARCH)}"))
    rows = (await db.execute(stmt)).all()

    results = [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_name=row.filename,
            chunk_index=row.chunk_index,
            page_number=row.page_number,
            content=row.content,
            score=round(1.0 - float(row.distance), 4),
        )
        for row in rows
    ]
    kept = [r for r in results if r.score >= min_score]
    logger.info(
        "Arama: %d sonuç, %d tanesi eşiği (%.2f) geçti. Skorlar: %s",
        len(results),
        len(kept),
        min_score,
        [r.score for r in results],
    )
    return kept
