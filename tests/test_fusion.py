"""RRF birleştirme testleri (veritabanı gerekmez)."""

from dataclasses import dataclass

import pytest

from app.services.fusion import fuse_ranked_chunks, reciprocal_rank_fusion


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    score: float  # kosinüs
    rrf_score: float | None = None
    matched_by: str = "vector"


def test_rrf_formula_and_sum():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)  # iki listede de var -> toplanır
    assert scores["d"] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == "b"


def test_rrf_weights_and_duplicates():
    scores = reciprocal_rank_fusion([["a", "a", "b"], ["b"]], k=0, weights=[2.0, 1.0])
    assert scores["a"] == pytest.approx(2.0)  # tekrar yok sayılır, sıra 1
    assert scores["b"] == pytest.approx(2.0 / 2 + 1.0)  # sıra 2 (tekrar atlandı) + 1. listede 1.


def test_rrf_validation():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def _fuse(vector, fts, **kw):
    params = dict(top_k=5, min_score=0.1, fts_min_score=0.05, k=60)
    params.update(kw)
    return fuse_ranked_chunks(vector, fts, **params)


def test_chunk_in_both_lists_wins_and_is_tagged():
    vec = [Chunk(1, 0.80), Chunk(2, 0.70), Chunk(3, 0.60)]
    fts = [Chunk(3, 0.60), Chunk(9, 0.30)]
    out = _fuse(vec, fts)
    assert out[0].chunk_id == 3  # iki kolda da var: 1/63 + 1/61 > 1/61
    by_id = {c.chunk_id: c for c in out}
    assert by_id[3].matched_by == "both"
    assert by_id[1].matched_by == "vector"
    assert by_id[9].matched_by == "fts"
    # skor alanı kosinüs olarak korunur
    assert by_id[9].score == 0.30
    assert all(c.rrf_score and c.rrf_score > 0 for c in out)


def test_keyword_hit_rescues_low_cosine_chunk():
    vec = [Chunk(1, 0.80)]
    fts = [Chunk(7, 0.07)]  # kosinüs 0.07 < min_score(0.1) ama FTS eşiği (0.05) geçildi
    out = _fuse(vec, fts)
    assert {c.chunk_id for c in out} == {1, 7}


def test_thresholds_filter():
    vec = [Chunk(1, 0.05)]  # yalnızca vektör, 0.05 < 0.1 -> elenir
    fts = [Chunk(2, 0.02)]  # fts ama 0.02 < 0.05 -> elenir
    assert _fuse(vec, fts) == []


def test_top_k_and_order_by_rrf():
    vec = [Chunk(i, 0.9 - i * 0.05) for i in range(1, 8)]
    out = _fuse(vec, [], top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]


def test_fts_weight_changes_order():
    vec = [Chunk(1, 0.9), Chunk(2, 0.8)]
    fts = [Chunk(2, 0.8), Chunk(1, 0.9)]
    assert _fuse(vec, fts, vector_weight=1.0, fts_weight=1.0)[0].chunk_id == 1  # eşitlikte kosinüs
    assert _fuse(vec, fts, vector_weight=1.0, fts_weight=3.0)[0].chunk_id == 2


def test_empty_inputs():
    assert _fuse([], []) == []
