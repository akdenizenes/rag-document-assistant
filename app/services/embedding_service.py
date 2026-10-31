"""sentence-transformers ile 384 boyutlu embedding üretimi."""

import asyncio
import logging
import threading

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL,
)
from app.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)

_model = None
_load_lock = threading.Lock()
_encode_lock = threading.Lock()  # CPU yoğun encode işlemlerini sıraya sokar


def get_embedding_model():
    """Modeli ilk çağrıda yükler (lazy singleton). İlk çalıştırmada model indirilir."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Embedding modeli yükleniyor: %s", EMBEDDING_MODEL)
                model = SentenceTransformer(EMBEDDING_MODEL)
                model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

                dim = model.get_sentence_embedding_dimension()
                if dim != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"Model {dim} boyutlu vektör üretiyor, ancak veritabanı "
                        f"Vector({EMBEDDING_DIM}) bekliyor. EMBEDDING_MODEL ayarını kontrol et."
                    )
                _model = model
                logger.info("Embedding modeli hazır (%d boyut).", dim)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Metinleri L2-normalize edilmiş vektörlere çevirir (senkron, CPU yoğun)."""
    if not texts:
        return []
    model = get_embedding_model()
    with _encode_lock:
        vectors = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return vectors.tolist()


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    """Event loop'u bloklamadan embedding üretir."""
    return await asyncio.to_thread(embed_texts, texts)
