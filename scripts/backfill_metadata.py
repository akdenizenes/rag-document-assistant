"""Adım 1: Mevcut PDF'ler için Tam Metin, Özet ve Metaveri Geri Doldurma (Backfill) Betiği.

Kullanım (proje kökünden):
    python scripts/backfill_metadata.py

Yaptığı işlemler:
  1. Veritabanındaki `completed` durumundaki tüm belgeleri tarar.
  2. `uploads/{doc.id}.pdf` dosyasını okuyarak (bulunamazsa mevcut `document_chunks` kayıtlarını birleştirerek) `full_text` sütununu doldurur.
  3. Özeti veya metaverisi eksik olan belgeler için Gemini (`generate_document_summary`) çağrısı yaparak `summary`, `doc_metadata` ve `summary_status='completed'` alanlarını günceller.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from app.database import AsyncSessionLocal, close_db
from app.models import Document, DocumentChunk, DocumentStatus, SummaryStatus
from app.services.document_service import build_full_text, upload_path
from app.services.llm_service import generate_document_summary
from app.services.pdf_service import extract_pages


async def _extract_or_reconstruct_full_text(session, doc: Document) -> tuple[str, int]:
    pdf_file = upload_path(doc.id)
    if pdf_file.exists() and pdf_file.is_file():
        print(f"  📄 Diskteki PDF okunuyor: {pdf_file}")
        pages = await asyncio.to_thread(extract_pages, pdf_file)
        return build_full_text(pages), len(pages)

    print(f"  ⚠️ Diskte {pdf_file.name} bulunamadı; veritabanındaki parçalardan (chunks) birleştiriliyor...")
    chunks = (
        await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc.id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
    ).scalars().all()
    text_joined = "\n\n".join(c.content for c in chunks)
    return text_joined, doc.page_count or 1


async def run_backfill() -> None:
    async with AsyncSessionLocal() as session:
        docs = (
            await session.execute(
                select(Document)
                .where(Document.status == DocumentStatus.COMPLETED.value)
                .order_by(Document.id.asc())
            )
        ).scalars().all()

        print(f"🔍 Toplam {len(docs)} tamamlanmış belge kontrol ediliyor...")

        for doc in docs:
            needs_full_text = not doc.full_text or not doc.full_text.strip()
            needs_summary = (
                not doc.summary
                or doc.summary_status != SummaryStatus.COMPLETED.value
            )

            if not needs_full_text and not needs_summary:
                print(f"⏭️  [ID={doc.id}] {doc.filename}: Tam metin ve özet zaten mevcut, atlanıyor.")
                continue

            print(f"⚙️  [ID={doc.id}] {doc.filename} işleniyor...")

            if needs_full_text:
                full_text, page_count = await _extract_or_reconstruct_full_text(session, doc)
                doc.full_text = full_text
                if not doc.page_count:
                    doc.page_count = page_count
                await session.commit()
                print(f"  ✅ full_text kaydedildi ({len(full_text)} karakter).")

            if doc.full_text and needs_summary:
                try:
                    doc.summary_status = SummaryStatus.PROCESSING.value
                    await session.commit()

                    summary, metadata = await generate_document_summary(
                        full_text=doc.full_text,
                        filename=doc.filename,
                        page_count=doc.page_count,
                    )
                    doc.summary = summary
                    doc.doc_metadata = metadata
                    doc.summary_status = SummaryStatus.COMPLETED.value
                    await session.commit()
                    print(
                        f"  ✅ Özet ve metaveri üretildi -> Tür: {metadata.get('document_type')}, "
                        f"Soru Sayısı: {metadata.get('question_count')}, "
                        f"Bölüm Sayısı: {metadata.get('section_count')}"
                    )
                except Exception as exc:
                    await session.rollback()
                    doc.summary_status = SummaryStatus.FAILED.value
                    await session.commit()
                    print(f"  ❌ Özet/metaveri hatası ([ID={doc.id}]): {exc}")

    await close_db()
    print("🎉 Backfill işlemi tamamlandı!")


if __name__ == "__main__":
    asyncio.run(run_backfill())
