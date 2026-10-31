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


async def init_db() -> None:
    """pgvector eklentisini garantiler ve tabloları oluşturur.

    Geliştirme aşaması içindir; ileride Alembic migration'a geçilebilir.
    """
    from app import models  # noqa: F401  (modellerin Base'e kaydolması için)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Uygulama kapanırken bağlantı havuzunu temizler."""
    await engine.dispose()
