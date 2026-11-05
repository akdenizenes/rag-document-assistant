"""Kaynak gösterimi (citations): arama sonucu parçaları API yanıtındaki `sources` listesine çevirir."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from app.config import SOURCE_SNIPPET_CHARS
from app.schemas.chat import SourceChunk

if TYPE_CHECKING:  # yalnızca tür bilgisi: çalışma zamanında sqlalchemy/pgvector yüklemez
    from app.services.retrieval_service import RetrievedChunk


def make_snippet(text: str, max_chars: int = SOURCE_SNIPPET_CHARS) -> str:
    """Metni tek satıra indirip en fazla `max_chars` karaktere kısaltır (kelime sınırında, "…" ile)."""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars]
    # Kelimenin ortasında kesmemek için son boşluğa kadar geri git (çok kısalmıyorsa).
    space = cut.rfind(" ")
    if space >= max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "…"


def build_sources(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """`sources` listesini üretir.

    `ref`, LLM'e giden bağlamdaki [n] etiketiyle (prompts.build_context ve özet/tam metin
    bağlamları) aynı sırayı kullanır: n. kaynak = `ref: n`. Bu yüzden `chunks`, bağlama verilen
    sırayla ve YALNIZCA bağlama giren parçalarla çağrılmalıdır (rag_service bunu sağlar).
    """
    sources: list[dict[str, Any]] = []
    for number, chunk in enumerate(chunks, start=1):
        data = SourceChunk.model_validate(chunk, from_attributes=True).model_dump(mode="json")
        data["ref"] = number
        data["snippet"] = make_snippet(chunk.content)
        sources.append(data)
    return sources
