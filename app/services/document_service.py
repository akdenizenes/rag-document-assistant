"""Belge işleme hattı: PDF -> metin -> chunk -> embedding -> veritabanı.

v2: tam metin (`documents.full_text`) saklanır; belge hazır olduktan SONRA arka planda LLM ile
kısa özet ve metaveri (`documents.summary`, `documents.doc_metadata`) üretilir. Özet üretimi
belge işlemeden bağımsızdır: başarısız olsa bile belge sorgulanabilir kalır.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import undefer

from app.config import (
    FULL_TEXT_CONTEXT_MAX_CHARS,
    FULL_TEXT_MAX_CHARS,
    LLM_MODEL,
    SUMMARY_ENABLED,
    UPLOAD_DIR,
)
from app.database import AsyncSessionLocal
from app.models import Document, DocumentChunk, DocumentStatus, SummaryStatus
from app.services.chunking import build_chunks
from app.services.embedding_service import embed_texts_async
from app.services.llm_service import LLMError, generate
from app.services.pdf_service import build_full_text, extract_pages
from app.services.prompts import build_summary_messages, parse_summary_response

logger = logging.getLogger(__name__)

# Aynı anda yalnızca bir özet isteği: birden çok belge birlikte yüklenince LLM kotası aşılmasın.
_summary_semaphore = asyncio.Semaphore(1)
_SUMMARY_MAX_ATTEMPTS = 2  # yalnızca 429 (kota) için yeniden denenir
_SUMMARY_RETRY_DELAY_SECONDS = 20.0

# Özet üretiminde korunan, LLM'den bağımsız (programatik) metaveri anahtarları.
_BASIC_META_KEYS = ("page_count", "char_count", "word_count", "text_truncated")


def upload_path(document_id: int) -> Path:
    """Diskteki dosya adı belge id'sine bağlıdır; kullanıcı girdisi dosya yoluna karışmaz."""
    return UPLOAD_DIR / f"{document_id}.pdf"


def save_file(document_id: int, data: bytes) -> None:
    upload_path(document_id).write_bytes(data)


def delete_file(document_id: int) -> None:
    upload_path(document_id).unlink(missing_ok=True)


def compute_basic_metadata(page_count: int, full_text: str, truncated: bool) -> dict[str, Any]:
    """LLM gerektirmeyen, güvenilir metaveri: sayfa/karakter/kelime sayısı."""
    return {
        "page_count": page_count,
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
        "text_truncated": truncated,
    }


async def process_document(document_id: int) -> None:
    """Yüklenmiş bir PDF'i işler. Arka plan görevi olarak çalışır, kendi DB oturumunu açar.

    Sıra: metin çıkar -> chunk + embedding + tam metin kaydet (belge `completed`) -> özet üret.
    """
    completed = False
    async with AsyncSessionLocal() as session:
        try:
            doc = await session.get(Document, document_id)
            if doc is None:
                return

            doc.status = DocumentStatus.PROCESSING.value
            await session.commit()

            pages = await asyncio.to_thread(extract_pages, upload_path(document_id))
            chunks = build_chunks(pages)
            if not chunks:
                raise ValueError(
                    "PDF'den metin çıkarılamadı (taranmış/görsel bir PDF olabilir)."
                )

            full_text, truncated = build_full_text(pages, FULL_TEXT_MAX_CHARS)
            if truncated:
                logger.warning(
                    "Belge %d tam metni %d karakterde kesildi.", document_id, FULL_TEXT_MAX_CHARS
                )

            embeddings = await embed_texts_async([c.content for c in chunks])

            session.add_all(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    content=c.content,
                    embedding=vector,
                )
                for c, vector in zip(chunks, embeddings, strict=True)
            )
            doc.page_count = len(pages)
            doc.full_text = full_text
            doc.doc_metadata = compute_basic_metadata(len(pages), full_text, truncated)
            doc.summary = None
            doc.summary_status = (
                SummaryStatus.PENDING.value if SUMMARY_ENABLED else SummaryStatus.SKIPPED.value
            )
            doc.status = DocumentStatus.COMPLETED.value
            doc.error_message = None
            await session.commit()
            completed = True
            logger.info(
                "Belge %d işlendi: %d sayfa, %d parça, %d karakter metin.",
                document_id,
                len(pages),
                len(chunks),
                len(full_text),
            )

        except Exception as exc:
            logger.exception("Belge %d işlenirken hata oluştu.", document_id)
            await session.rollback()
            # Belge bu sırada silinmiş olabilir; o durumda sessizce çık.
            doc = await session.get(Document, document_id)
            if doc is not None:
                doc.status = DocumentStatus.FAILED.value
                doc.error_message = str(exc)[:1000]
                await session.commit()

    # Belge artık sorgulanabilir. Özet, DB oturumu kapandıktan sonra ayrıca üretilir
    # (LLM beklenirken bağlantı tutulmaz). Kendi hatalarını yakalar, buraya fırlatmaz.
    if completed and SUMMARY_ENABLED:
        await generate_document_summary(document_id)


async def _set_summary_status(document_id: int, status: SummaryStatus) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(summary_status=status.value)
        )
        await session.commit()


async def generate_document_summary(document_id: int) -> bool:
    """Belgenin tam metninden LLM ile kısa özet + metaveri üretip kaydeder.

    - `summary_status`: processing -> completed | failed (SUMMARY_ENABLED=false ise çağrılmaz).
    - DB oturumu LLM çağrısı sırasında AÇIK TUTULMAZ.
    - Hata fırlatmaz; başarıyı True/False döner (backfill betiği de bunu kullanır).
    """
    async with AsyncSessionLocal() as session:
        doc = await session.scalar(
            select(Document)
            .options(undefer(Document.full_text))
            .where(Document.id == document_id)
        )
        if doc is None or not doc.full_text:
            logger.warning("Belge %d için özet üretilemedi: tam metin yok.", document_id)
            return False
        filename, full_text = doc.filename, doc.full_text
        doc.summary_status = SummaryStatus.PROCESSING.value
        await session.commit()

    messages = build_summary_messages(filename, full_text[:FULL_TEXT_CONTEXT_MAX_CHARS])
    try:
        async with _summary_semaphore:
            for attempt in range(1, _SUMMARY_MAX_ATTEMPTS + 1):
                try:
                    raw = await generate(messages)
                    break
                except LLMError as exc:
                    if exc.status_code != 429 or attempt == _SUMMARY_MAX_ATTEMPTS:
                        raise
                    logger.warning(
                        "Belge %d özeti: kota aşıldı, %.0f sn sonra tekrar denenecek.",
                        document_id,
                        _SUMMARY_RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(_SUMMARY_RETRY_DELAY_SECONDS)
        summary, llm_meta = parse_summary_response(raw)
    except (LLMError, ValueError) as exc:
        logger.warning("Belge %d için özet üretilemedi: %s", document_id, exc)
        await _set_summary_status(document_id, SummaryStatus.FAILED)
        return False
    except Exception:
        logger.exception("Belge %d özeti üretilirken beklenmeyen hata.", document_id)
        await _set_summary_status(document_id, SummaryStatus.FAILED)
        return False

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:  # özet üretilirken belge silinmiş olabilir
            return False
        existing = doc.doc_metadata or {}
        metadata = {k: existing[k] for k in _BASIC_META_KEYS if k in existing}
        metadata.update({k: v for k, v in llm_meta.items() if v is not None})
        metadata["summary_model"] = LLM_MODEL
        metadata["summary_generated_at"] = datetime.now(timezone.utc).isoformat()

        doc.summary = summary
        doc.doc_metadata = metadata
        doc.summary_status = SummaryStatus.COMPLETED.value
        await session.commit()
    logger.info("Belge %d için özet ve metaveri kaydedildi.", document_id)
    return True


async def recover_stale_documents() -> None:
    """Sunucu yeniden başlarken yarıda kalmış işleri temizler.

    - 'pending/processing' belgeler -> 'failed'
    - 'processing' özetler -> 'failed' (scripts/backfill_metadata.py ile yeniden üretilebilir)
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Document)
            .where(Document.summary_status == SummaryStatus.PROCESSING.value)
            .values(summary_status=SummaryStatus.FAILED.value)
        )
        await session.execute(
            update(Document)
            .where(
                Document.status.in_(
                    [DocumentStatus.PENDING.value, DocumentStatus.PROCESSING.value]
                )
            )
            .values(
                status=DocumentStatus.FAILED.value,
                error_message="İşlem sunucu yeniden başlatıldığı için yarıda kaldı. Dosyayı tekrar yükleyin.",
            )
        )
        await session.commit()
