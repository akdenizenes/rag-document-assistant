"""Arama katmanı: vektör benzerliği (pgvector) + Türkçe tam metin araması (PostgreSQL FTS).

İki giriş noktası vardır:
  - search_similar_chunks : yalnızca vektör araması (kosinüs mesafesi)
  - hybrid_search_chunks  : vektör + FTS, Reciprocal Rank Fusion (RRF) ile birleştirilir

Hibrit arama neden? Vektör araması anlamı yakalar ama "Soru 7", "Ohm", "RS-232" gibi tam
terimlerde zayıf kalabilir; tam metin araması tam terimi yakalar ama eş anlamlıyı kaçırır.
RRF yalnızca SIRALARI birleştirdiği için iki farklı skor ölçeğini normalize etmek gerekmez.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Float, Text, cast, func, literal_column, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import UserDefinedType

from app.config import (
    HNSW_EF_SEARCH,
    HYBRID_CANDIDATES,
    HYBRID_FTS_MIN_SCORE,
    HYBRID_FTS_WEIGHT,
    HYBRID_RRF_K,
    HYBRID_VECTOR_WEIGHT,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_TOP_K,
)
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.embedding_service import embed_texts_async
from app.services.fusion import fuse_ranked_chunks

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
    # Hibrit aramada RRF skoru (sıralama bununla yapılır); vektör aramada None.
    rrf_score: float | None = None
    # Parçanın nereden geldiği: vector | fts | both | summary | full_text
    matched_by: str = "vector"
    # cross_encoder modunda yeniden sıralama skoru (model ölçeğinde ham skor); diğerlerinde None
    rerank_score: float | None = None


class _TSQuery(UserDefinedType):
    """PostgreSQL `tsquery` türü (CAST için)."""

    cache_ok = True

    def get_col_spec(self, **kw) -> str:
        return "TSQUERY"


# Sabit bir literal olarak gömülür (bind parametresi DEĞİL): asyncpg `regconfig` türünü
# str'den kodlayamaz; literal ise PostgreSQL tarafında doğrudan regconfig'e çözülür.
_TS_CONFIG = literal_column("'turkish'")


def _or_tsquery(query: str):
    """Doğal dil sorusunu VEYA (|) mantığıyla birleştirilmiş tsquery'ye çevirir.

    `plainto_tsquery` terimleri VE (&) ile bağlar; uzun bir soruda bu neredeyse hiçbir parçayla
    eşleşmez. Burada & -> | yapılır, sıralamayı (ts_rank_cd) çok terimle eşleşen parçalar kazanır.
    Türkçe sözlük durak sözcükleri (ve, bir, bu ...) ve ekleri (stemming) zaten ayıklar.
    """
    strict = func.plainto_tsquery(_TS_CONFIG, query)
    return cast(func.replace(cast(strict, Text), " & ", " | "), _TSQuery())


def _row_to_chunk(row, *, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.id,
        document_id=row.document_id,
        document_name=row.filename,
        chunk_index=row.chunk_index,
        page_number=row.page_number,
        content=row.content,
        score=score,
    )


def _base_select(*extra):
    return (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            Document.filename,
            DocumentChunk.chunk_index,
            DocumentChunk.page_number,
            DocumentChunk.content,
            *extra,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.status == DocumentStatus.COMPLETED.value)
    )


def _scope(stmt, user_id: int, document_ids: Sequence[int] | None):
    """Kullanıcının belgeleriyle sınırlar; `document_ids` doluysa yalnızca o belgelerde arar."""
    stmt = stmt.where(Document.user_id == user_id)
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(list(document_ids)))
    return stmt


async def _vector_search(
    db: AsyncSession,
    query_vector: list[float],
    *,
    user_id: int,
    document_ids: Sequence[int] | None,
    limit: int,
) -> list[RetrievedChunk]:
    """En yakın `limit` parçayı kosinüs mesafesine göre döner (eşik UYGULANMAZ)."""
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = _scope(_base_select(distance), user_id, document_ids).order_by(distance).limit(limit)

    # HNSW indeksinin bu sorgu için baktığı aday sayısı (yalnızca bu transaction için).
    await db.execute(text(f"SET LOCAL hnsw.ef_search = {int(HNSW_EF_SEARCH)}"))
    rows = (await db.execute(stmt)).all()
    return [_row_to_chunk(r, score=round(1.0 - float(r.distance), 4)) for r in rows]


async def _fts_search(
    db: AsyncSession,
    query: str,
    query_vector: list[float],
    *,
    user_id: int,
    document_ids: Sequence[int] | None,
    limit: int,
) -> list[RetrievedChunk]:
    """Türkçe tam metin aramasıyla en alakalı `limit` parçayı döner (ts_rank_cd sırasıyla).

    Bulunan parçaların `score` alanı yine kosinüs benzerliğidir (arayüzde tutarlı skor için).
    """
    ts_query = _or_tsquery(query)
    fts_rank = func.ts_rank_cd(DocumentChunk.content_tsv, ts_query, type_=Float).label("fts_rank")
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        _scope(_base_select(distance, fts_rank), user_id, document_ids)
        .where(DocumentChunk.content_tsv.op("@@", is_comparison=True)(ts_query))
        .order_by(fts_rank.desc(), DocumentChunk.id)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [_row_to_chunk(r, score=round(1.0 - float(r.distance), 4)) for r in rows]


async def search_similar_chunks(
    db: AsyncSession,
    *,
    user_id: int,
    query: str,
    document_ids: Sequence[int] | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
) -> list[RetrievedChunk]:
    """Sorguyu embed eder ve kullanıcının belgelerindeki en yakın `top_k` parçayı döner.

    - Yalnızca `completed` durumundaki belgeler aranır.
    - `document_ids` verilirse arama yalnızca o belgelerle sınırlanır (boş/None = tüm belgeler).
    - `min_score` altındaki parçalar elenir.
    """
    [query_vector] = await embed_texts_async([query])
    results = await _vector_search(
        db, query_vector, user_id=user_id, document_ids=document_ids, limit=top_k
    )
    kept = [r for r in results if r.score >= min_score]
    logger.info(
        "Vektör arama: %d sonuç, %d tanesi eşiği (%.2f) geçti. Skorlar: %s",
        len(results),
        len(kept),
        min_score,
        [r.score for r in results],
    )
    return kept


async def hybrid_search_chunks(
    db: AsyncSession,
    *,
    user_id: int,
    query: str,
    document_ids: Sequence[int] | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
) -> list[RetrievedChunk]:
    """Vektör + Türkçe tam metin aramasını RRF ile birleştirerek en iyi `top_k` parçayı döner.

    1. Her iki koldan `max(top_k * 3, HYBRID_CANDIDATES)` aday alınır.
    2. Adaylar sıralarına göre RRF ile birleştirilir (skor = ağırlık / (k + sıra)).
    3. Eşikler uygulanır (bkz. `fuse_ranked_chunks`) ve ilk `top_k` döner.

    Anahtar kelime kolu başarısız olursa (örn. `scripts/migrate_v2.py` çalıştırılmamışsa
    `content_tsv` sütunu yoktur) uyarı günlüğe yazılır ve arama yalnızca vektörle devam eder;
    SAVEPOINT kullanıldığı için ana transaction bozulmaz.
    """
    candidates = max(top_k * 3, HYBRID_CANDIDATES)
    [query_vector] = await embed_texts_async([query])

    vector_hits = await _vector_search(
        db, query_vector, user_id=user_id, document_ids=document_ids, limit=candidates
    )

    fts_hits: list[RetrievedChunk] = []
    try:
        async with db.begin_nested():
            fts_hits = await _fts_search(
                db,
                query,
                query_vector,
                user_id=user_id,
                document_ids=document_ids,
                limit=candidates,
            )
    except DBAPIError:
        logger.warning(
            "Anahtar kelime araması başarısız oldu, yalnızca vektör sonuçları kullanılacak. "
            "scripts/migrate_v2.py çalıştırıldı mı?",
            exc_info=True,
        )

    results = fuse_ranked_chunks(
        vector_hits,
        fts_hits,
        top_k=top_k,
        min_score=min_score,
        fts_min_score=HYBRID_FTS_MIN_SCORE,
        k=HYBRID_RRF_K,
        vector_weight=HYBRID_VECTOR_WEIGHT,
        fts_weight=HYBRID_FTS_WEIGHT,
    )
    logger.info(
        "Hibrit arama: vektör=%d, anahtar kelime=%d aday -> %d sonuç. (kosinüs, rrf, kaynak): %s",
        len(vector_hits),
        len(fts_hits),
        len(results),
        [(r.score, r.rrf_score, r.matched_by) for r in results],
    )
    return results
