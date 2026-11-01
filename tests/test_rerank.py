"""Reranker mantığı testleri (model/veritabanı gerekmez)."""

import asyncio
from dataclasses import dataclass

import pytest

from app.services import rerank_service as rr


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    content: str
    score: float  # kosinüs
    matched_by: str = "vector"
    rerank_score: float | None = None


def chunks(*scores, matched=None):
    matched = matched or {}
    return [
        Chunk(i, f"parça {i}", s, matched.get(i, "vector")) for i, s in enumerate(scores, start=1)
    ]


def run(coro):
    return asyncio.run(coro)


def test_smart_filter_drops_weak_chunks_keeps_order():
    out = rr.smart_filter(chunks(0.80, 0.70, 0.30, 0.20), ratio=0.5, min_keep=1)
    assert [c.chunk_id for c in out] == [1, 2]  # kesim = 0.40


def test_smart_filter_min_keep_and_both():
    out = rr.smart_filter(chunks(0.80, 0.10, 0.09, 0.08), ratio=0.9, min_keep=2)
    assert [c.chunk_id for c in out] == [1, 2]  # ilk 2 korunur
    out = rr.smart_filter(chunks(0.80, 0.10, 0.09), ratio=0.9, min_keep=1, )
    assert [c.chunk_id for c in out] == [1]
    out = rr.smart_filter(chunks(0.80, 0.10, 0.09, matched={3: "both"}), ratio=0.9, min_keep=1)
    assert [c.chunk_id for c in out] == [1, 3]  # hem vektör hem FTS -> korunur


def test_smart_filter_empty():
    assert rr.smart_filter([], ratio=0.5, min_keep=2) == []


def test_apply_rerank_scores_orders_and_limits():
    cs = chunks(0.9, 0.8, 0.7)
    out = rr.apply_rerank_scores(cs, [0.1, 5.0, 2.0], top_k=2)
    assert [c.chunk_id for c in out] == [2, 3]
    assert out[0].rerank_score == 5.0
    assert out[0].score == 0.8  # kosinüs skoru korunur


def test_apply_rerank_scores_min_score_and_validation():
    cs = chunks(0.9, 0.8)
    assert rr.apply_rerank_scores(cs, [-3.0, -4.0], top_k=5, min_score=-3.5)[0].chunk_id == 1
    assert len(rr.apply_rerank_scores(cs, [-3.0, -4.0], top_k=5, min_score=-3.5)) == 1
    assert rr.apply_rerank_scores(cs, [-9.0, -9.0], top_k=5, min_score=0.0) == []
    with pytest.raises(ValueError):
        rr.apply_rerank_scores(cs, [1.0], top_k=1)


def test_to_floats_handles_scalars_and_arrays():
    assert rr._to_floats([0.5, 2]) == [0.5, 2.0]
    assert rr._to_floats([[0.1, 0.9], [0.7, 0.3]]) == [0.9, 0.3]


def test_candidates_needed():
    assert rr.candidates_needed(5, "off") == 5
    assert rr.candidates_needed(5, "score") == 5
    assert rr.candidates_needed(5, "cross_encoder") == max(5, rr.RERANK_CANDIDATES)
    assert rr.candidates_needed(50, "cross_encoder") == 50


def test_rerank_modes():
    cs = chunks(0.8, 0.7, 0.2, 0.1)
    assert [c.chunk_id for c in run(rr.rerank_chunks("q", cs, top_k=3, mode="off"))] == [1, 2, 3]
    out = run(rr.rerank_chunks("q", cs, top_k=5, mode="score"))
    assert [c.chunk_id for c in out] == [1, 2]
    assert run(rr.rerank_chunks("q", [], top_k=3, mode="score")) == []


def test_cross_encoder_reorders_with_predictor():
    calls = []

    def fake(query, texts):
        calls.append((query, texts))
        return [0.1, 0.9, 0.5]

    out = run(rr.rerank_chunks("soru", chunks(0.9, 0.8, 0.7), top_k=2, mode="cross_encoder", predictor=fake))
    assert [c.chunk_id for c in out] == [2, 3]
    assert calls == [("soru", ["parça 1", "parça 2", "parça 3"])]


def test_cross_encoder_failure_falls_back_to_score_filter(monkeypatch):
    def boom(query, texts):
        raise RuntimeError("model yok")

    monkeypatch.setattr(rr, "_disabled_until", 0.0)
    out = run(rr.rerank_chunks("q", chunks(0.8, 0.7, 0.1), top_k=5, mode="cross_encoder", predictor=boom))
    assert [c.chunk_id for c in out] == [1, 2]  # düştü: skor filtresi
    assert rr._disabled_until > 0  # bir süre tekrar denenmeyecek
    monkeypatch.setattr(rr, "_disabled_until", 0.0)
