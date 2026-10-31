"""Tek seferlik veritabanı başlatma betiği.

Kullanım (proje kökünden):
    python init_db.py

Yaptığı işler:
  1. pgvector eklentisini etkinleştirir (CREATE EXTENSION IF NOT EXISTS vector)
  2. models.py içindeki tüm tabloları ve HNSW indeksini oluşturur
  3. Sonucu doğrulayıp ekrana yazar

Betik tekrar çalıştırılabilir (idempotent); var olan tabloları silmez.
"""

import asyncio

from sqlalchemy import text

from app.database import Base, close_db, engine, init_db


async def main() -> None:
    await init_db()

    async with engine.connect() as conn:
        ext = await conn.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        tables = (
            await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' ORDER BY tablename"
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

    print(f"pgvector eklentisi : {'v' + ext if ext else 'YOK!'}")
    print(f"Beklenen tablolar  : {', '.join(sorted(Base.metadata.tables))}")
    print(f"Veritabanındakiler : {', '.join(tables)}")
    print(f"document_chunks indeksleri: {', '.join(indexes)}")

    missing = set(Base.metadata.tables) - set(tables)
    if ext is None or missing:
        raise SystemExit(f"Hata: eksik öğeler var -> {missing or 'pgvector'}")
    print("Veritabanı hazır.")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
