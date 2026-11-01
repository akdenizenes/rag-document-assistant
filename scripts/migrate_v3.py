"""v3 şema migrasyonu: sohbet geçmişi tabloları (Alembic kullanılmadığı için bağımsız betik).

Kullanım (proje kökünden):
    python scripts/migrate_v3.py              # migrasyonu uygular
    python scripts/migrate_v3.py --dry-run    # yalnızca çalıştıracağı SQL'i yazdırır
    python scripts/migrate_v3.py --print-sql  # psql'e verilebilecek saf SQL çıktısı

Oluşturdukları (app/models.py içindeki ChatSession / ChatMessage ile birebir aynı şema):
  chat_sessions   id, user_id (FK users, ON DELETE CASCADE), title, document_ids JSONB,
                  created_at, updated_at
                  + ix_chat_sessions_user_id, ix_chat_sessions_user_updated (user_id, updated_at)
  chat_messages   id, session_id (FK chat_sessions, ON DELETE CASCADE), role, content,
                  sources JSONB, model, route, created_at
                  + ix_chat_messages_session_id

Betik TEKRAR ÇALIŞTIRILABİLİR (idempotent): her adım IF NOT EXISTS kullanır, hiçbir veriyi
silmez ya da değiştirmez. Tüm değişiklikler tek bir transaction'da yapılır; hata olursa hiçbiri
uygulanmaz. Sonunda şema, beklenen sütun ve indekslerle karşılaştırılıp doğrulanır.

Notlar:
  - `Base.metadata.create_all` (init_db) var olmayan TABLOLARI zaten oluşturur; bu betik aynı işi
    uygulamayı başlatmadan, açık SQL ile ve doğrulamalı yapmak içindir. Hangisi önce çalışırsa
    çalışsın sonuç aynıdır.
  - `create_all` var olan bir tabloya sonradan sütun/indeks EKLEMEZ. Tablolardan biri elle ya da
    eski bir sürümle başka şekilde oluşturulmuşsa doğrulama adımı eksikleri raporlar ve çıkış
    kodu 1 ile biter.
  - Önce `users` tablosu bulunmalıdır (yabancı anahtar için). Boş bir veritabanı için
    `python init_db.py` çalıştır.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.database import close_db, engine  # noqa: E402

LOCK_TIMEOUT = "30s"

# (açıklama, SQL) — sırayla çalışır. Şema app/models.py ile aynı tutulmalıdır.
STEPS: list[tuple[str, str]] = [
    (
        "chat_sessions tablosu",
        "CREATE TABLE IF NOT EXISTS chat_sessions ("
        "id SERIAL NOT NULL, "
        "user_id INTEGER NOT NULL, "
        "title VARCHAR(200) NOT NULL, "
        "document_ids JSONB, "
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, "
        "updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, "
        "PRIMARY KEY (id), "
        "FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)",
    ),
    (
        "ix_chat_sessions_user_id",
        "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions (user_id)",
    ),
    (
        "ix_chat_sessions_user_updated (user_id, updated_at)",
        "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_updated "
        "ON chat_sessions (user_id, updated_at)",
    ),
    (
        "chat_messages tablosu",
        "CREATE TABLE IF NOT EXISTS chat_messages ("
        "id SERIAL NOT NULL, "
        "session_id INTEGER NOT NULL, "
        "role VARCHAR(20) NOT NULL, "
        "content TEXT NOT NULL, "
        "sources JSONB, "
        "model VARCHAR(100), "
        "route VARCHAR(20), "
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, "
        "PRIMARY KEY (id), "
        "FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE)",
    ),
    (
        "ix_chat_messages_session_id",
        "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id ON chat_messages (session_id)",
    ),
]

EXPECTED_COLUMNS: dict[str, set[str]] = {
    "chat_sessions": {"id", "user_id", "title", "document_ids", "created_at", "updated_at"},
    "chat_messages": {
        "id", "session_id", "role", "content", "sources", "model", "route", "created_at",
    },
}
EXPECTED_INDEXES: dict[str, set[str]] = {
    "chat_sessions": {"ix_chat_sessions_user_id", "ix_chat_sessions_user_updated"},
    "chat_messages": {"ix_chat_messages_session_id"},
}


def render_sql() -> str:
    lines = ["BEGIN;", f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}';"]
    lines += [f"{sql};" for _, sql in STEPS]
    lines.append("COMMIT;")
    return "\n".join(lines)


def find_problems(
    columns: dict[str, set[str]], indexes: dict[str, set[str]]
) -> list[str]:
    """Beklenen şema ile veritabanındakini karşılaştırır; eksikleri okunabilir metinle döner."""
    problems: list[str] = []
    for table, expected in EXPECTED_COLUMNS.items():
        if table not in columns:
            problems.append(f"tablo yok: {table}")
            continue
        problems += [f"eksik sütun: {table}.{c}" for c in sorted(expected - columns[table])]
    for table, expected in EXPECTED_INDEXES.items():
        present = indexes.get(table, set())
        problems += [f"eksik indeks: {name} ({table})" for name in sorted(expected - present)]
    return problems


async def read_schema(conn) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Sohbet tablolarının güncel sütunlarını ve indekslerini okur."""
    # Tablo adları bu dosyadaki sabitlerdir (kullanıcı girdisi değil), bu yüzden doğrudan SQL'e yazılır.
    table_list = ", ".join(f"'{name}'" for name in EXPECTED_COLUMNS)

    columns: dict[str, set[str]] = {}
    rows = await conn.execute(
        text(
            "SELECT table_name, column_name FROM information_schema.columns "
            f"WHERE table_schema = current_schema() AND table_name IN ({table_list})"
        )
    )
    for table_name, column_name in rows:
        columns.setdefault(table_name, set()).add(column_name)

    indexes: dict[str, set[str]] = {}
    rows = await conn.execute(
        text(
            "SELECT tablename, indexname FROM pg_indexes "
            f"WHERE schemaname = current_schema() AND tablename IN ({table_list})"
        )
    )
    for table_name, index_name in rows:
        indexes.setdefault(table_name, set()).add(index_name)
    return columns, indexes


async def run() -> int:
    async with engine.begin() as conn:
        has_users = await conn.scalar(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = 'users'"
            )
        )
        if not has_users:
            raise SystemExit(
                "Hata: 'users' tablosu bulunamadı. Boş bir veritabanı için `python init_db.py` "
                "çalıştır; sohbet tablolarını da oluşturur."
            )

        await conn.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
        for description, sql in STEPS:
            await conn.execute(text(sql))
            print(f"  ✓ {description}")

    # Transaction kesinleşti; şimdi gerçek durumu doğrula.
    async with engine.connect() as conn:
        columns, indexes = await read_schema(conn)
        sessions = await conn.scalar(text("SELECT count(*) FROM chat_sessions"))
        messages = await conn.scalar(text("SELECT count(*) FROM chat_messages"))

    problems = find_problems(columns, indexes)
    if problems:
        print("\nMigrasyon uygulandı ama şema beklenenle uyuşmuyor:")
        for problem in problems:
            print(f"  ✗ {problem}")
        print(
            "\nBu tablolar daha önce farklı bir şemayla oluşturulmuş olabilir. Veriler silinmedi; "
            "eksikleri elle tamamla ya da tablo boşsa DROP TABLE ile silip betiği yeniden çalıştır."
        )
        return 1

    print("\nMigrasyon tamamlandı, şema doğrulandı.")
    print(f"  Sohbet oturumları : {sessions}")
    print(f"  Mesajlar          : {messages}")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="v3 şema migrasyonu (sohbet geçmişi tabloları)")
    parser.add_argument("--dry-run", action="store_true", help="SQL'i yazdır, uygulama")
    parser.add_argument("--print-sql", action="store_true", help="yalnızca saf SQL yazdır")
    args = parser.parse_args()

    if args.print_sql:
        print(render_sql())
        return 0
    if args.dry_run:
        print("-- Uygulanacak SQL (dry-run):")
        print(render_sql())
        return 0

    print("v3 migrasyonu uygulanıyor...")
    try:
        return await run()
    finally:
        await close_db()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
