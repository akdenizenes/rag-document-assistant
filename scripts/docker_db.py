"""Docker başlangıç yardımcıları (entrypoint.sh tarafından çağrılır).

    python scripts/docker_db.py url        # POSTGRES_* değişkenlerinden DATABASE_URL üretir
    python scripts/docker_db.py wait       # PostgreSQL bağlantı kabul edene kadar bekler
    python scripts/docker_db.py bootstrap  # pgvector eklentisi + (eksik) tablolar

`bootstrap` bilerek `verify_schema()` çalıştırmaz: eski şemalı bir veritabanında v2 sütunları
henüz yoktur ve onları hemen ardından çalışan migrate_v2.py ekler. Boş veritabanında tüm
tablolar (v2/v3 dahil) burada oluşur; migrasyonlar o durumda zararsız bir "değişiklik yok"tur.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_database_url() -> str:
    """DATABASE_URL tanımlıysa onu, değilse POSTGRES_* değişkenlerinden (URL-encode ederek) üretir."""
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit
    try:
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        database = os.environ["POSTGRES_DB"]
    except KeyError as exc:
        raise SystemExit(f"Hata: {exc.args[0]} tanımlı değil (.env.docker dosyasını kontrol et).") from exc
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")
    # Parolada @ : / gibi karakterler olsa bile URL bozulmasın.
    return (
        f"postgresql+asyncpg://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(database, safe='')}"
    )


async def wait_for_db(timeout: float, interval: float = 2.0) -> None:
    import asyncpg

    dsn = build_database_url().replace("postgresql+asyncpg://", "postgresql://", 1)
    deadline = time.monotonic() + timeout
    attempt = 0
    last_error: Exception | None = None
    while True:
        attempt += 1
        try:
            conn = await asyncpg.connect(dsn, timeout=5)
            try:
                await conn.fetchval("SELECT 1")
            finally:
                await conn.close()
            print(f"[wait] Veritabanı hazır (deneme {attempt}).", flush=True)
            return
        except Exception as exc:  # bağlantı reddi, kimlik doğrulama, "starting up" vb.
            last_error = exc
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"Hata: veritabanına {timeout:.0f} sn içinde bağlanılamadı: "
                    f"{type(last_error).__name__}: {last_error}"
                ) from last_error
            print(f"[wait] Veritabanı henüz hazır değil ({type(exc).__name__}); {interval:.0f} sn sonra tekrar...", flush=True)
            await asyncio.sleep(interval)


async def bootstrap() -> None:
    from sqlalchemy import text

    from app import models  # noqa: F401  (modellerin Base'e kaydolması için)
    from app.database import Base, close_db, engine

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        print("[bootstrap] pgvector eklentisi ve tablolar hazır.", flush=True)
    finally:
        await close_db()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "url":
        print(build_database_url())
    elif command == "wait":
        # DATABASE_URL'nin tamamlanmış olması için entrypoint önce `url` çıktısını export eder.
        asyncio.run(wait_for_db(float(os.getenv("WAIT_FOR_DB_TIMEOUT", "90"))))
    elif command == "bootstrap":
        asyncio.run(bootstrap())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
