"""Hibrit Arama: pgvector (kosinüs benzerliği) + PostgreSQL Full-Text Search (Türkçe) + RRF."""

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
    score: float  # kosinüs benzerliği veya RRF birleşik skoru


async def search_similar_chunks(
    db: AsyncSession,
    *,
    user_id: int,
    query: str,
    document_id: int | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
) -> list[RetrievedChunk]:
    """Yalnızca vektör benzerlik araması yapar (fallback veya saf semantik arama için)."""
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
    return [r for r in results if r.score >= min_score]


async def hybrid_search_chunks(
    db: AsyncSession,
    *,
    user_id: int,
    query: str,
    document_id: int | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Adım 3: Vektör benzerliği (pgvector) ile Türkçe Full-Text Search (tsvector)
    sonuçlarını Reciprocal Rank Fusion (RRF) ile birleştirir.
    """
    [query_vector] = await embed_texts_async([query])
    vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"
    pool_limit = max(top_k * 4, 20)

    try:
        await db.execute(text(f"SET LOCAL hnsw.ef_search = {int(HNSW_EF_SEARCH)}"))

        sql = text(
            """
            WITH vector_candidates AS (
                SELECT
                    dc.id AS chunk_id,
                    1.0 - (dc.embedding <=> CAST(:query_vec AS vector)) AS vec_sim,
                    ROW_NUMBER() OVER (
                        ORDER BY dc.embedding <=> CAST(:query_vec AS vector)
                    ) AS vec_rank
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE d.user_id = CAST(:user_id AS INTEGER)
                  AND d.status = CAST(:completed_status AS VARCHAR)
                  AND (CAST(:doc_id AS INTEGER) IS NULL OR dc.document_id = CAST(:doc_id AS INTEGER))
                  AND (1.0 - (dc.embedding <=> CAST(:query_vec AS vector))) >= CAST(:min_score AS FLOAT)
                ORDER BY dc.embedding <=> CAST(:query_vec AS vector)
                LIMIT CAST(:pool_limit AS INTEGER)
            ),
            fts_candidates AS (
                SELECT
                    dc.id AS chunk_id,
                    ts_rank_cd(dc.content_tsv, websearch_to_tsquery('turkish', CAST(:query_text AS TEXT))) AS fts_rank_score,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(dc.content_tsv, websearch_to_tsquery('turkish', CAST(:query_text AS TEXT))) DESC
                    ) AS fts_rank
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE d.user_id = CAST(:user_id AS INTEGER)
                  AND d.status = CAST(:completed_status AS VARCHAR)
                  AND (CAST(:doc_id AS INTEGER) IS NULL OR dc.document_id = CAST(:doc_id AS INTEGER))
                  AND dc.content_tsv @@ websearch_to_tsquery('turkish', CAST(:query_text AS TEXT))
                ORDER BY fts_rank_score DESC
                LIMIT CAST(:pool_limit AS INTEGER)
            ),
            rrf_fused AS (
                SELECT
                    COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
                    COALESCE(1.0 / (CAST(:rrf_k AS FLOAT) + v.vec_rank), 0.0)
                      + COALESCE(1.0 / (CAST(:rrf_k AS FLOAT) + f.fts_rank), 0.0) AS rrf_score,
                    v.vec_sim,
                    f.fts_rank_score
                FROM vector_candidates v
                FULL OUTER JOIN fts_candidates f ON v.chunk_id = f.chunk_id
            )
            SELECT
                dc.id AS chunk_id,
                dc.document_id,
                d.filename,
                dc.chunk_index,
                dc.page_number,
                dc.content,
                r.rrf_score,
                r.vec_sim
            FROM rrf_fused r
            JOIN document_chunks dc ON dc.id = r.chunk_id
            JOIN documents d ON d.id = dc.document_id
            ORDER BY r.rrf_score DESC, COALESCE(r.vec_sim, 0.0) DESC
            LIMIT CAST(:top_k AS INTEGER)
            """
        )

        rows = (
            await db.execute(
                sql,
                {
                    "query_vec": vector_literal,
                    "query_text": query,
                    "user_id": int(user_id),
                    "completed_status": DocumentStatus.COMPLETED.value,
                    "doc_id": int(document_id) if document_id is not None else None,
                    "min_score": float(min_score),
                    "pool_limit": int(pool_limit),
                    "rrf_k": int(rrf_k),
                    "top_k": int(top_k),
                },
            )
        ).all()

        results = [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                document_name=row.filename,
                chunk_index=row.chunk_index,
                page_number=row.page_number,
                content=row.content,
                score=round(
                    float(row.vec_sim) if row.vec_sim is not None else float(row.rrf_score),
                    4,
                ),
            )
            for row in rows
        ]

        logger.info(
            "Hibrit Arama (RRF): %d sonuç bulundu. RRF skorları: %s",
            len(results),
            [round(float(r.rrf_score), 4) for r in rows],
        )
        return results

    except Exception:
        logger.exception(
            "Hibrit arama sırasında SQL hatası oluştu; standart vektör aramasına (fallback) geçiliyor."
        )
        await db.rollback()
        return await search_similar_chunks(
            db,
            user_id=user_id,
            query=query,
            document_id=document_id,
            top_k=top_k,
            min_score=min_score,
        )