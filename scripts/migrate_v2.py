"""Adım 1 & Adım 3 Veritabanı Migrasyon Betiği (Alembic kullanmadan).

Kullanım (proje kökünden):
    python scripts/migrate_v2.py

Yaptığı işlemler:
  1. `documents` tablosuna `full_text`, `summary`, `doc_metadata` (JSONB) ve `summary_status` sütunlarını ekler.
  2. `document_chunks` tablosuna Türkçe ('turkish' sözlüğü ile) otomatik üretilen `content_tsv` (tsvector) sütununu ekler.
  3. `document_chunks(content_tsv)` üzerinde GIN indeksini (`ix_document_chunks_content_tsv`) oluşturur.
"""

import asyncio
import sys
from pathlib import Path

# Proje kök dizinini import yoluna ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from app.database import close_db, engine

Statements = [
    # Adım 1: documents tablosu yeni sütunlar
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS full_text TEXT;",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT;",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_metadata JSONB DEFAULT '{}'::jsonb;",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_status VARCHAR(20) DEFAULT 'pending';",
    "CREATE INDEX IF NOT EXISTS ix_documents_summary_status ON documents (summary_status);",
    # Adım 3: document_chunks tablosuna Türkçe tsvector GENERATED ALWAYS sütunu
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'document_chunks'
              AND column_name = 'content_tsv'
        ) THEN
            ALTER TABLE document_chunks
            ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('turkish', coalesce(content, ''))) STORED;
        END IF;
    END
    $$;
    """,
    # Adım 3: GIN indeksi
    """
    CREATE INDEX IF NOT EXISTS ix_document_chunks_content_tsv
    ON document_chunks USING gin (content_tsv);
    """,
]


async def run_migration() -> None:
    print("🚀 Migrasyon v2 başlatılıyor...")
    async with engine.begin() as conn:
        for stmt in Statements:
            await conn.execute(text(stmt))

    async with engine.connect() as conn:
        doc_cols = (
            await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'documents' ORDER BY column_name"
                )
            )
        ).scalars().all()
        chunk_cols = (
            await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'document_chunks' ORDER BY column_name"
                )
            )
        ).scalars().all()
        indexes = (
            await conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'document_chunks' ORDER BY indexname"
                )
            )
        ).scalars().all()

    print(f"✅ documents sütunları      : {', '.join(doc_cols)}")
    print(f"✅ document_chunks sütunları: {', '.join(chunk_cols)}")
    print(f"✅ document_chunks indeksler: {', '.join(indexes)}")
    print("🎉 Migrasyon v2 başarıyla tamamlandı!")
    await close_db()


if __name__ == "__main__":
    asyncio.run(run_migration())
