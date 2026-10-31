"""RAG hattı: soru -> benzer parçalar -> prompt -> LLM -> cevap + kaynaklar."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import LLM_MODEL, RETRIEVAL_TOP_K
from app.schemas.chat import ChatMessage
from app.services.llm_service import generate
from app.services.prompts import (
    NO_ANSWER,
    build_context,
    build_messages,
    build_retrieval_query,
)
from app.services.retrieval_service import RetrievedChunk, search_similar_chunks

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RagResult:
    answer: str
    sources: list[RetrievedChunk]
    model: str


async def answer_question(
    db: AsyncSession,
    *,
    user_id: int,
    question: str,
    document_id: int | None = None,
    history: Sequence[ChatMessage] = (),
    top_k: int | None = None,
) -> RagResult:
    """Soruyu kullanıcının belgelerine dayanarak yanıtlar."""
    chunks = await search_similar_chunks(
        db,
        user_id=user_id,
        query=build_retrieval_query(question, history),
        document_id=document_id,
        top_k=top_k or RETRIEVAL_TOP_K,
    )

    # LLM yanıtı saniyeler sürebilir; bu sırada DB bağlantısını havuza geri bırak.
    await db.rollback()

    if not chunks:
        logger.info("İlgili parça bulunamadı, LLM çağrılmadan standart yanıt dönülüyor.")
        return RagResult(answer=NO_ANSWER, sources=[], model=LLM_MODEL)

    context, used = build_context(chunks)
    answer = await generate(build_messages(question, context, history))
    return RagResult(answer=answer, sources=chunks[:used], model=LLM_MODEL)
