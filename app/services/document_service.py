"""Belge işleme hattı: PDF -> tam metin + chunk -> embedding -> veritabanı -> arka plan özet/metaveri."""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import update

from app.config import UPLOAD_DIR
from app.database import AsyncSessionLocal
from app.models import Document, DocumentChunk, DocumentStatus, SummaryStatus
from app.services.chunking import build_chunks, normalize_text
from app.services.embedding_service import embed_texts_async
from app.services.llm_service import generate_document_summary
from app.services.pdf_service import PageText, extract_pages

logger = logging.getLogger(__name__)


def upload_path(document_id: int) -> Path:
    """Diskteki dosya adı belge id'sine bağlıdır; kullanıcı girdisi dosya yoluna karışmaz."""
    return UPLOAD_DIR / f"{document_id}.pdf"


def save_file(document_id: int, data: bytes) -> None:
    upload_path(document_id).write_bytes(data)


def delete_file(document_id: int) -> None:
    upload_path(document_id).unlink(missing_ok=True)


def build_full_text(pages: list[PageText]) -> str:
    """Sayfa metinlerini sayfa başlıklarıyla birleştirerek tam metin (full_text) oluşturur."""
    blocks: list[str] = []
    for p in pages:
        cleaned = normalize_text(p.text)
        if cleaned:
            blocks.append(f"--- Sayfa {p.page_number} ---\n{cleaned}")
    return "\n\n".join(blocks)


async def generate_summary_for_document(document_id: int) -> None:
    """Belge için Gemini ile kısa özet ve JSONB metaveri üretip kaydeder (Adım 1)."""
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None or not doc.full_text:
            return

        doc.summary_status = SummaryStatus.PROCESSING.value
        await session.commit()

        try:
            summary, metadata = await generate_document_summary(
                full_text=doc.full_text,
                filename=doc.filename,
                page_count=doc.page_count,
            )
            doc = await session.get(Document, document_id)
            if doc is None:
                return
            doc.summary = summary
            doc.doc_metadata = metadata
            doc.summary_status = SummaryStatus.COMPLETED.value
            await session.commit()
            logger.info(
                "Belge %d özeti ve metaverisi hazırlandı (tür=%s, konu=%d).",
                document_id,
                metadata.get("document_type"),
                len(metadata.get("main_topics", [])),
            )
        except Exception:
            logger.exception("Belge %d için özet/metaveri üretilirken hata oluştu.", document_id)
            await session.rollback()
            doc = await session.get(Document, document_id)
            if doc is not None:
                doc.summary_status = SummaryStatus.FAILED.value
                await session.commit()


async def process_document(document_id: int) -> None:
    """Yüklenmiş bir PDF'i işler. Arka plan görevi olarak çalışır, kendi DB oturumunu açar."""
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

            full_text = build_full_text(pages)
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
            doc.status = DocumentStatus.COMPLETED.value
            doc.summary_status = SummaryStatus.PENDING.value
            doc.error_message = None
            await session.commit()
            logger.info(
                "Belge %d işlendi: %d sayfa, %d parça, %d karakter tam metin.",
                document_id,
                len(pages),
                len(chunks),
                len(full_text),
            )

        except Exception as exc:
            logger.exception("Belge %d işlenirken hata oluştu.", document_id)
            await session.rollback()
            doc = await session.get(Document, document_id)
            if doc is not None:
                doc.status = DocumentStatus.FAILED.value
                doc.error_message = str(exc)[:1000]
                await session.commit()
            return

    # Ana belge işleme tamamlandıktan hemen sonra arka planda özet ve metaveri üret
    await generate_summary_for_document(document_id)


async def recover_stale_documents() -> None:
    """Sunucu yeniden başlarken 'pending/processing' kalmış belgeleri 'failed' yapar."""
    async with AsyncSessionLocal() as session:
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
