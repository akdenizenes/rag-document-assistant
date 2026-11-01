"""Asenkron SQLAlchemy 2.0 bağlantı katmanı."""

import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

DATABASE_URL: str | None = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL tanımlı değil. .env dosyasını oluşturduğundan emin ol."
    )

SQL_ECHO: bool = os.getenv("SQL_ECHO", "false").lower() == "true"

engine = create_async_engine(
    DATABASE_URL,
    echo=SQL_ECHO,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # kopmuş bağlantıları otomatik yeniler
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # commit sonrası nesne alanlarına async erişim için
    autoflush=False,
)


class Base(DeclarativeBase):
    """Tüm ORM modellerinin ortak tabanı."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: her istek için bir oturum açar, hata olursa rollback yapar."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# v2 ile eklenen sütunlar. `create_all` mevcut tablolara sütun EKLEMEZ; eski bir veritabanında
# bunlar yoksa kullanıcıya açık bir hata verilir (scripts/migrate_v2.py çalıştırılmalı).
_V2_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "documents": {"summary", "doc_metadata", "summary_status", "full_text"},
    "document_chunks": {"content_tsv"},
}


async def verify_schema() -> None:
    """Şemanın v2 sütunlarını içerdiğini doğrular; eksik varsa RuntimeError fırlatır."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name IN ('documents', 'document_chunks')"
            )
        )
        present: dict[str, set[str]] = {}
        for table_name, column_name in rows:
            present.setdefault(table_name, set()).add(column_name)

    missing = [
        f"{table}.{column}"
        for table, columns in _V2_REQUIRED_COLUMNS.items()
        for column in sorted(columns - present.get(table, set()))
    ]
    if missing:
        raise RuntimeError(
            "Veritabanı şeması güncel değil, eksik sütunlar: "
            + ", ".join(missing)
            + ". Önce şunu çalıştır: python scripts/migrate_v2.py"
        )


async def init_db() -> None:
    """pgvector eklentisini garantiler, tabloları oluşturur ve şemayı doğrular.

    Geliştirme aşaması içindir; ileride Alembic migration'a geçilebilir.
    Yeni kurulumda `create_all` v2 sütunlarını zaten oluşturur; mevcut bir veritabanı için
    `scripts/migrate_v2.py` gerekir.
    """
    from app import models  # noqa: F401  (modellerin Base'e kaydolması için)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    await verify_schema()


async def close_db() -> None:
    """Uygulama kapanırken bağlantı havuzunu temizler."""
    await engine.dispose()
