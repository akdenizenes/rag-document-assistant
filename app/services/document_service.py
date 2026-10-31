"""Belge işleme hattı: PDF -> metin -> chunk -> embedding -> veritabanı."""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import update

from app.config import UPLOAD_DIR
from app.database import AsyncSessionLocal
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.chunking import build_chunks
from app.services.embedding_service import embed_texts_async
from app.services.pdf_service import extract_pages

logger = logging.getLogger(__name__)


def upload_path(document_id: int) -> Path:
    """Diskteki dosya adı belge id'sine bağlıdır; kullanıcı girdisi dosya yoluna karışmaz."""
    return UPLOAD_DIR / f"{document_id}.pdf"


def save_file(document_id: int, data: bytes) -> None:
    upload_path(document_id).write_bytes(data)


def delete_file(document_id: int) -> None:
    upload_path(document_id).unlink(missing_ok=True)


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
            doc.status = DocumentStatus.COMPLETED.value
            doc.error_message = None
            await session.commit()
            logger.info(
                "Belge %d işlendi: %d sayfa, %d parça.", document_id, len(pages), len(chunks)
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
