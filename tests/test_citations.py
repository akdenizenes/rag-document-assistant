"""Kaynak gösterimi: `ref` numarası, `snippet` kısaltma ve prompt bağlamıyla tutarlılık."""

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

from app.schemas.chat import SourceChunk  # noqa: E402
from app.services.citations import build_sources, make_snippet  # noqa: E402
from app.services.prompts import build_context  # noqa: E402


def chunk(i: int, content: str = "Teslim süresi 30 gündür.", page: int | None = 3) -> SimpleNamespace:
    # RetrievedChunk ile aynı alanlar (retrieval_service'i import etmek sqlalchemy gerektirir).
    return SimpleNamespace(
        chunk_id=100 + i,
        document_id=7,
        document_name="sozlesme.pdf",
        chunk_index=i,
        page_number=page,
        content=content,
        score=0.8123,
        rrf_score=0.03,
        matched_by="both",
    )


def test_snippet_short_text_unchanged_and_whitespace_flattened():
    assert make_snippet("a  b\n\n c\t d") == "a b c d"


def test_snippet_truncates_on_word_boundary_with_ellipsis():
    text = "kelime " * 100
    snippet = make_snippet(text, max_chars=50)
    assert snippet.endswith("…") and len(snippet) <= 51
    assert not snippet[:-1].endswith("kelim")  # kelime ortasında kesilmez


def test_snippet_single_long_word_hard_cut():
    assert make_snippet("x" * 500, max_chars=20) == "x" * 20 + "…"


def test_build_sources_fields_and_ref_numbering():
    sources = build_sources([chunk(0), chunk(1, content="y " * 400, page=None)])
    assert [s["ref"] for s in sources] == [1, 2]
    first = sources[0]
    assert first["document_name"] == "sozlesme.pdf" and first["chunk_id"] == 100
    assert first["page_number"] == 3 and first["score"] == 0.8123
    assert first["snippet"] == first["content"] == "Teslim süresi 30 gündür."
    assert sources[1]["page_number"] is None
    assert len(sources[1]["snippet"]) < len(sources[1]["content"])
    SourceChunk.model_validate(first)  # API yanıt şemasına uyuyor


def test_ref_matches_context_labels_sent_to_llm():
    chunks = [chunk(i) for i in range(3)]
    context, used = build_context(chunks)
    sources = build_sources(chunks[:used])
    for s in sources:
        assert f"[{s['ref']}] Kaynak: {s['document_name']}" in context


def test_old_stored_sources_without_ref_still_validate():
    old = {"chunk_id": 1, "document_id": 1, "document_name": "a.pdf", "chunk_index": 0,
           "page_number": 1, "score": 0.5, "content": "x"}
    parsed = SourceChunk.model_validate(old)
    assert parsed.ref is None and parsed.snippet is None
