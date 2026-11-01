"""prepare_query yönlendirme ve geri dönüş (fallback) mantığı testleri.

Veritabanı gerekmez: veri yükleyici ve arama fonksiyonları sahte olanlarla değiştirilir.
Gerçek bağımlılıklar (sqlalchemy, pgvector ...) kurulu değilse test atlanır.
"""

import asyncio

import pytest

pytest.importorskip("sqlalchemy")

from app.services import rag_service  # noqa: E402
from app.services.prompts import NO_ANSWER, DocContext  # noqa: E402
from app.services.retrieval_service import RetrievedChunk  # noqa: E402


class FakeDB:
    def __init__(self, ready: int = 1) -> None:
        self.ready = ready
        self.rollbacks = 0

    async def scalar(self, stmt):  # hazır belge sayısı
        return self.ready

    async def rollback(self) -> None:
        self.rollbacks += 1


def make_doc(**kw) -> DocContext:
    base = dict(document_id=1, filename="ders.pdf", page_count=12)
    base.update(kw)
    return DocContext(**base)


def chunk(i: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i, document_id=1, document_name="ders.pdf", chunk_index=i,
        page_number=2, content=f"parça {i}", score=0.8,
    )


@pytest.fixture
def env(monkeypatch):
    """Sahte veri yükleyici + arama; hangi aramanın çağrıldığını kaydeder."""
    state = {"summary_docs": [], "text_docs": [], "search_result": [chunk()], "calls": []}

    async def fake_load(db, *, user_id, document_ids, need_summary=False, need_full_text=False, limit=10):
        return state["summary_docs"] if need_summary else state["text_docs"]

    async def fake_hybrid(db, **kw):
        state["calls"].append("hybrid")
        return state["search_result"]

    async def fake_vector(db, **kw):
        state["calls"].append("vector")
        return state["search_result"]

    monkeypatch.setattr(rag_service, "_load_docs", fake_load)
    monkeypatch.setattr(rag_service, "hybrid_search_chunks", fake_hybrid)
    monkeypatch.setattr(rag_service, "search_similar_chunks", fake_vector)
    monkeypatch.setattr(rag_service, "HYBRID_SEARCH_ENABLED", True)
    return state


def prepare(question, *, db=None, document_ids=None):
    db = db or FakeDB()
    prepared = asyncio.run(
        rag_service.prepare_query(db, user_id=1, question=question, document_ids=document_ids)
    )
    return prepared, db


def system_prompt(prepared) -> str:
    return prepared.messages[0]["content"]


def test_summary_route_uses_summary_without_search(env):
    env["summary_docs"] = [make_doc(summary="Kısa özet.", doc_metadata={"title": "Ağ"})]
    prepared, db = prepare("Bu belgeyi özetle", document_ids=[1])
    assert prepared.route == "summary"
    assert "ÖZET modu" in system_prompt(prepared)
    assert "Kısa özet." in prepared.messages[-1]["content"]
    assert prepared.sources[0].matched_by == "summary"
    assert env["calls"] == []  # arama yapılmadı
    assert db.rollbacks == 1  # bağlantı LLM öncesi bırakıldı


def test_summary_missing_falls_back_to_full_text_when_scoped(env):
    env["text_docs"] = [make_doc(full_text="--- Sayfa 1 ---\nSoru 1) a")]
    prepared, _ = prepare("Belgeyi özetle", document_ids=[1])
    assert prepared.route == "full_text"
    assert env["calls"] == []


def test_summary_missing_unscoped_falls_back_to_hybrid(env):
    prepared, _ = prepare("Belgeyi özetle")
    assert prepared.route == "hybrid"
    assert env["calls"] == ["hybrid"]


def test_full_text_route_puts_text_in_context(env):
    env["text_docs"] = [make_doc(full_text="--- Sayfa 1 ---\nSoru 1) a\n--- Sayfa 2 ---\nSoru 2) b")]
    prepared, _ = prepare("Kaç soru var?", document_ids=[1])
    assert prepared.route == "full_text"
    assert "numaralandırarak listele" in system_prompt(prepared)
    assert "Soru 2) b" in prepared.messages[-1]["content"]
    assert prepared.sources[0].matched_by == "full_text"
    assert env["calls"] == []


def test_full_text_unscoped_multiple_docs_asks_to_pick_one(env):
    prepared, _ = prepare("Kaç soru var?", db=FakeDB(ready=5))
    assert prepared.route == "full_text"
    assert prepared.messages is None
    assert "daraltıp" in prepared.direct_answer and "5" in prepared.direct_answer
    assert env["calls"] == []


def test_full_text_unscoped_single_doc_is_used(env):
    env["text_docs"] = [make_doc(full_text="--- Sayfa 1 ---\nSoru 1) a")]
    prepared, _ = prepare("Kaç soru var?", db=FakeDB(ready=1))
    assert prepared.route == "full_text" and prepared.messages is not None


def test_full_text_unscoped_no_ready_docs_goes_to_hybrid(env):
    env["search_result"] = []
    prepared, _ = prepare("Kaç soru var?", db=FakeDB(ready=0))
    assert prepared.route == "hybrid"
    assert prepared.messages is None and prepared.direct_answer is None  # -> NO_ANSWER


def test_full_text_missing_for_legacy_doc_falls_back_to_hybrid(env):
    prepared, _ = prepare("Kaç soru var?", document_ids=[1])  # text_docs boş: backfill yapılmamış
    assert prepared.route == "hybrid"
    assert env["calls"] == ["hybrid"]


def test_specific_question_uses_hybrid_search(env):
    prepared, _ = prepare("Teslim süresi kaç gündür?", document_ids=[1])
    assert prepared.route == "hybrid"
    assert env["calls"] == ["hybrid"]
    assert "Ek kurallar" not in system_prompt(prepared)
    assert prepared.sources == [chunk()]


def test_hybrid_can_be_disabled(env, monkeypatch):
    monkeypatch.setattr(rag_service, "HYBRID_SEARCH_ENABLED", False)
    prepare("Teslim süresi kaç gündür?")
    assert env["calls"] == ["vector"]


def test_no_chunks_means_no_llm(env):
    env["search_result"] = []
    prepared, _ = prepare("Teslim süresi kaç gündür?")
    assert prepared.messages is None and prepared.route == "hybrid"


def test_answer_question_direct_answer_skips_llm(env, monkeypatch):
    async def boom(messages):
        raise AssertionError("LLM çağrılmamalıydı")

    monkeypatch.setattr(rag_service, "generate", boom)
    result = asyncio.run(
        rag_service.answer_question(FakeDB(ready=9), user_id=1, question="Kaç soru var?")
    )
    assert "daraltıp" in result.answer and result.route == "full_text"

    env["search_result"] = []
    result = asyncio.run(
        rag_service.answer_question(FakeDB(), user_id=1, question="Teslim süresi kaç gündür?")
    )
    assert result.answer == NO_ANSWER and result.route == "hybrid"
