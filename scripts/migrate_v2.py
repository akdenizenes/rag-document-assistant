"""v2 şema migrasyonu (Alembic kullanılmadığı için bağımsız betik).

Kullanım (proje kökünden, API'yi durdurmak şart değil ama önerilir):
    python scripts/migrate_v2.py              # migrasyonu uygular
    python scripts/migrate_v2.py --dry-run    # yalnızca çalıştıracağı SQL'i yazdırır
    python scripts/migrate_v2.py --print-sql  # psql'e verilebilecek saf SQL çıktısı

Ekledikleri:
  documents.summary         TEXT      kısa belge özeti
  documents.doc_metadata    JSONB     başlık, ana konular, soru/bölüm sayısı, sayfa/kelime sayısı
  documents.summary_status  VARCHAR   pending | processing | completed | failed | skipped
  documents.full_text       TEXT      sayfa işaretli tam metin
  document_chunks.content_tsv         TSVECTOR (otomatik üretilen, 'turkish' sözlüğü) + GIN indeksi

Betik TEKRAR ÇALIŞTIRILABİLİR (idempotent): her adım IF NOT EXISTS kullanır, veri silmez.
Tüm değişiklikler tek bir transaction'da yapılır; hata olursa hiçbiri uygulanmaz.

Notlar:
  - STORED üretilmiş sütun eklemek document_chunks tablosunu yeniden yazar (küçük/orta
    tablolarda saniyeler sürer) ve kısa süre kilit alır. Kilit 30 sn içinde alınamazsa betik
    hata verip çıkar (açık bir transaction tutan bir bağlantı olabilir; API'yi durdurup tekrar dene).
  - Yeni (boş) bir veritabanında bu betiğe gerek yoktur: `python init_db.py` v2 sütunlarını
    zaten oluşturur.
  - Mevcut belgelerin full_text/özet alanlarını doldurmak için ardından
    `python scripts/backfill_metadata.py` çalıştır.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.database import close_db, engine, verify_schema  # noqa: E402

LOCK_TIMEOUT = "30s"

# (açıklama, SQL) — sırayla çalışır.
STEPS: list[tuple[str, str]] = [
    (
        "documents.summary",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT",
    ),
    (
        "documents.doc_metadata",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_metadata JSONB",
    ),
    (
        "documents.summary_status",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_status VARCHAR(20) "
        "NOT NULL DEFAULT 'pending'",
    ),
    (
        "documents.full_text",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS full_text TEXT",
    ),
    (
        "document_chunks.content_tsv (üretilmiş tsvector, turkish)",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('turkish', content)) STORED",
    ),
    (
        "ix_document_chunks_content_tsv (GIN)",
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_content_tsv "
        "ON document_chunks USING gin (content_tsv)",
    ),
]


def render_sql() -> str:
    lines = ["BEGIN;", f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}';"]
    lines += [f"{sql};" for _, sql in STEPS]
    lines.append("COMMIT;")
    return "\n".join(lines)


async def run() -> None:
    async with engine.begin() as conn:
        tables = set(
            (
                await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = current_schema()"
                    )
                )
            ).scalars()
        )
        missing_tables = {"documents", "document_chunks"} - tables
        if missing_tables:
            raise SystemExit(
                f"Hata: tablolar bulunamadı ({', '.join(sorted(missing_tables))}). "
                "Boş bir veritabanı için `python init_db.py` çalıştır; v2 sütunlarını zaten oluşturur."
            )

        has_turkish = await conn.scalar(
            text("SELECT 1 FROM pg_ts_config WHERE cfgname = 'turkish'")
        )
        if not has_turkish:
            raise SystemExit(
                "Hata: PostgreSQL'de 'turkish' metin arama yapılandırması yok. "
                "Standart PostgreSQL / pgvector imajlarında bulunur; özel bir derleme kullanıyorsan "
                "snowball sözlüklerinin kurulu olduğundan emin ol."
            )

        await conn.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
        for description, sql in STEPS:
            await conn.execute(text(sql))
            print(f"  ✓ {description}")

    await verify_schema()

    async with engine.connect() as conn:
        total, no_text, no_summary = (
            await conn.execute(
                text(
                    "SELECT count(*), count(*) FILTER (WHERE full_text IS NULL), "
                    "count(*) FILTER (WHERE summary IS NULL) FROM documents"
                )
            )
        ).one()
        chunks = await conn.scalar(text("SELECT count(*) FROM document_chunks"))
        indexed = await conn.scalar(
            text("SELECT count(*) FROM document_chunks WHERE content_tsv IS NOT NULL")
        )

    print("\nMigrasyon tamamlandı.")
    print(f"  Belgeler           : {total} (tam metni boş: {no_text}, özeti boş: {no_summary})")
    print(f"  Parçalar           : {chunks} (tsvector dolu: {indexed})")
    if no_text or no_summary:
        print("\nSıradaki adım: mevcut belgeleri doldur ->  python scripts/backfill_metadata.py")


async def main() -> None:
    parser = argparse.ArgumentParser(description="v2 şema migrasyonu")
    parser.add_argument("--dry-run", action="store_true", help="SQL'i yazdır, uygulama")
    parser.add_argument("--print-sql", action="store_true", help="yalnızca saf SQL yazdır")
    args = parser.parse_args()

    if args.print_sql:
        print(render_sql())
        return
    if args.dry_run:
        print("-- Uygulanacak SQL (dry-run):")
        print(render_sql())
        return

    print("v2 migrasyonu uygulanıyor...")
    try:
        await run()
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
