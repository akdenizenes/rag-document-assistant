"""Reciprocal Rank Fusion (RRF): birden fazla sıralı listeyi tek sıralamada birleştirir.

Saf fonksiyondur (veritabanı yok), bu yüzden kolayca test edilir.

Her liste için bir öğenin skoru `ağırlık / (k + sıra)` olur (sıra 1'den başlar); bir öğe
birden çok listede geçiyorsa skorları toplanır. Ham skorlar (kosinüs, ts_rank) farklı
ölçeklerde olduğundan yalnızca SIRA kullanılır; bu da skor normalizasyonu gerektirmez.

Referans: Cormack, Clarke, Büttcher (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[Hashable, float]:
    """Sıralı listeleri (en iyi başta) RRF ile birleştirir.

    - `rankings`: her biri öğe kimliklerinden oluşan, en alakalıdan başlayan listeler.
    - `k`: sıra etkisini yumuşatan sabit (standart değer 60).
    - `weights`: liste başına çarpan; verilmezse hepsi 1.0.

    Döner: {öğe: RRF skoru}. Bir listede aynı öğe tekrar ederse yalnızca ilk (en iyi) sırası sayılır.
    """
    if k < 0:
        raise ValueError("k negatif olamaz.")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights uzunluğu rankings ile aynı olmalı.")

    scores: dict[Hashable, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[Hashable] = set()
        position = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            position += 1
            scores[item] = scores.get(item, 0.0) + weight / (k + position)
    return scores


def fuse_ranked_chunks(
    vector_hits: Sequence,
    fts_hits: Sequence,
    *,
    top_k: int,
    min_score: float,
    fts_min_score: float,
    k: int = 60,
    vector_weight: float = 1.0,
    fts_weight: float = 1.0,
) -> list:
    """Vektör ve anahtar kelime sonuçlarını RRF ile birleştirir, eşikleri uygular, `top_k` döner.

    Girdi öğeleri `RetrievedChunk` benzeri dataclass'lardır (alanlar: chunk_id, score, rrf_score,
    matched_by). `score` HER İKİ koldan gelen parça için de gerçek kosinüs benzerliğidir; böylece
    arayüzde gösterilen skor tutarlı kalır, sıralama ise RRF skoruyla yapılır.

    Eşik kuralı:
      - yalnızca vektör kolundan gelen parça: kosinüs >= `min_score`
      - anahtar kelime eşleşmesi olan parça (tek başına ya da vektörle birlikte):
        kosinüs >= `fts_min_score` (tam eşleşen terim/kod anlamsal olarak zayıf olsa da elenmesin)
    """
    scores = reciprocal_rank_fusion(
        [[h.chunk_id for h in vector_hits], [h.chunk_id for h in fts_hits]],
        k=k,
        weights=[vector_weight, fts_weight],
    )
    in_vector = {h.chunk_id for h in vector_hits}
    in_fts = {h.chunk_id for h in fts_hits}

    merged: dict[int, object] = {}
    for hit in (*vector_hits, *fts_hits):
        merged.setdefault(hit.chunk_id, hit)

    ranked = sorted(
        merged.values(),
        key=lambda c: (-scores[c.chunk_id], -c.score, c.chunk_id),
    )

    results = []
    for chunk in ranked:
        found_vector = chunk.chunk_id in in_vector
        found_fts = chunk.chunk_id in in_fts
        matched_by = "both" if found_vector and found_fts else ("fts" if found_fts else "vector")
        floor = min_score if matched_by == "vector" else fts_min_score
        if chunk.score < floor:
            continue
        results.append(
            dataclasses.replace(
                chunk, rrf_score=round(scores[chunk.chunk_id], 6), matched_by=matched_by
            )
        )
        if len(results) >= top_k:
            break
    return results
