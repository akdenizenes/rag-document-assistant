"""Hibrit arama sonuçlarını LLM'e gitmeden önce daraltan / yeniden sıralayan katman.

Üç mod (RERANK_MODE):
  off           : sonuç olduğu gibi (RRF sırası), yalnızca top_k kesilir
  score         : modelsiz akıllı filtre (varsayılan). En iyi kosinüs skoruna göre belirgin şekilde
                  zayıf kalan parçalar elenir; hem vektör hem anahtar kelime aramasında çıkan
                  ("both") parçalar korunur. Ek maliyet yok, token tasarrufu sağlar.
  cross_encoder : sentence-transformers CrossEncoder (sorgu, parça) çiftlerini birlikte okuyup
                  alaka skoru üretir; aday listesi bu skora göre yeniden sıralanır. Bi-encoder
                  (embedding) aramasından belirgin şekilde daha isabetlidir ama CPU'da yavaştır
                  ve ilk kullanımda model indirilir.

cross_encoder modunda model yüklenemez ya da tahmin başarısız olursa hata fırlatılmaz:
uyarı yazılır, `score` filtresine düşülür ve model bir süre (COOLDOWN) tekrar denenmez.

Girdi öğeleri `RetrievedChunk` benzeri dataclass'lardır (alanlar: content, score, matched_by,
rerank_score); bu modül veritabanına bağımlı değildir.
"""

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Callable, Sequence

from app.config import (
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATES,
    RERANK_MAX_LENGTH,
    RERANK_MIN_KEEP,
    RERANK_MIN_SCORE,
    RERANK_MODE,
    RERANK_MODEL,
    RERANK_RELATIVE_CUTOFF,
)

logger = logging.getLogger(__name__)

_COOLDOWN_SECONDS = 300.0
_model = None
_load_lock = threading.Lock()
_predict_lock = threading.Lock()  # CPU yoğun tahminleri sıraya sokar
_disabled_until = 0.0


def candidates_needed(top_k: int, mode: str | None = None) -> int:
    """Aramadan kaç aday istenmeli: cross_encoder daha geniş havuzdan seçer."""
    return max(top_k, RERANK_CANDIDATES) if (mode or RERANK_MODE) == "cross_encoder" else top_k


# --------------------------------------------------------------------------- #
# Saf mantık
# --------------------------------------------------------------------------- #
def smart_filter(chunks: Sequence, *, ratio: float, min_keep: int) -> list:
    """Modelsiz daraltma: en iyi kosinüs skorunun `ratio` katının altındaki parçaları eler.

    - İlk `min_keep` parça (RRF sırasıyla) her zaman kalır.
    - Hem vektör hem anahtar kelime aramasında çıkan ("both") parçalar kalır: iki bağımsız
      yöntemin hemfikir olması güçlü bir alaka işaretidir.
    - Sıra değişmez.
    """
    if not chunks:
        return []
    cutoff = max(c.score for c in chunks) * ratio
    return [
        c
        for index, c in enumerate(chunks)
        if index < min_keep or c.score >= cutoff or getattr(c, "matched_by", None) == "both"
    ]


def apply_rerank_scores(
    chunks: Sequence,
    scores: Sequence[float],
    *,
    top_k: int,
    min_score: float | None = None,
) -> list:
    """Parçaları verilen skorlara göre (yüksekten düşüğe) sıralar, eşiği uygular, `top_k` döner."""
    if len(chunks) != len(scores):
        raise ValueError("chunks ve scores aynı uzunlukta olmalı.")
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: -pair[1])  # kararlı sıralama
    results = []
    for chunk, score in ranked:
        if min_score is not None and score < min_score:
            continue
        results.append(dataclasses.replace(chunk, rerank_score=round(float(score), 4)))
        if len(results) >= top_k:
            break
    return results


def _to_floats(raw) -> list[float]:
    """CrossEncoder çıktısını (tek skorlu ya da çok sınıflı) düz float listesine çevirir."""
    return [float(x[-1]) if hasattr(x, "__len__") else float(x) for x in raw]


# --------------------------------------------------------------------------- #
# Cross-encoder
# --------------------------------------------------------------------------- #
def get_reranker():
    """CrossEncoder'ı ilk çağrıda yükler (lazy singleton). İlk çalıştırmada model indirilir."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                logger.info("Reranker modeli yükleniyor: %s", RERANK_MODEL)
                _model = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
                logger.info("Reranker modeli hazır.")
    return _model


def _predict_sync(query: str, texts: list[str]) -> list[float]:
    model = get_reranker()
    pairs = [(query, text) for text in texts]
    with _predict_lock:
        raw = model.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)
    return _to_floats(raw)


async def warmup() -> None:
    """Açılışta modeli ısıtır (cross_encoder modunda). Başarısızlık açılışı engellemez."""
    if RERANK_MODE != "cross_encoder":
        return
    try:
        await asyncio.to_thread(get_reranker)
    except Exception:
        logger.warning(
            "Reranker modeli açılışta yüklenemedi; ilk istekte tekrar denenecek.", exc_info=True
        )


# --------------------------------------------------------------------------- #
# Giriş noktası
# --------------------------------------------------------------------------- #
async def rerank_chunks(
    query: str,
    chunks: Sequence,
    *,
    top_k: int,
    mode: str | None = None,
    predictor: Callable[[str, list[str]], list[float]] | None = None,
) -> list:
    """Arama sonuçlarını moda göre daraltır / yeniden sıralar ve en fazla `top_k` parça döner.

    `predictor` yalnızca testler içindir (CrossEncoder yerine sahte skorlayıcı).
    """
    global _disabled_until
    mode = mode or RERANK_MODE
    if not chunks:
        return []
    if mode == "off":
        return list(chunks[:top_k])

    def by_score() -> list:
        return smart_filter(chunks, ratio=RERANK_RELATIVE_CUTOFF, min_keep=RERANK_MIN_KEEP)[:top_k]

    if mode == "score":
        kept = by_score()
        logger.info("Skor filtresi: %d -> %d parça.", len(chunks), len(kept))
        return kept

    # cross_encoder
    if predictor is None and time.monotonic() < _disabled_until:
        return by_score()
    try:
        scores = await asyncio.to_thread(predictor or _predict_sync, query, [c.content for c in chunks])
    except Exception:
        logger.warning(
            "Cross-encoder kullanılamadı; skor filtresine düşülüyor (%.0f sn tekrar denenmeyecek).",
            _COOLDOWN_SECONDS,
            exc_info=True,
        )
        _disabled_until = time.monotonic() + _COOLDOWN_SECONDS
        return by_score()

    results = apply_rerank_scores(chunks, scores, top_k=top_k, min_score=RERANK_MIN_SCORE)
    logger.info(
        "Cross-encoder: %d aday -> %d parça. (kosinüs, rerank): %s",
        len(chunks),
        len(results),
        [(r.score, r.rerank_score) for r in results],
    )
    return results
