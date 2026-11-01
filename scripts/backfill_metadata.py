"""Mevcut belgeler için tam metin, metaveri ve özeti geri doldurur (backfill).

Önce `python scripts/migrate_v2.py` çalıştırılmış olmalı.

Kullanım (proje kökünden):
    python scripts/backfill_metadata.py                 # eksik olan her şeyi doldurur
    python scripts/backfill_metadata.py --no-summary    # yalnızca tam metin (LLM çağrısı YOK)
    python scripts/backfill_metadata.py --document-id 7 # tek belge
    python scripts/backfill_metadata.py --limit 5       # en fazla 5 belge
    python scripts/backfill_metadata.py --retry-failed  # özeti 'failed' olanları da dene
    python scripts/backfill_metadata.py --force         # dolu olanları da baştan üret
    python scripts/backfill_metadata.py --dry-run       # ne yapılacağını göster

Ne yapar? `uploads/<belge_id>.(pdf|docx|txt)` dosyalarını okuyup, veritabanında kaydı olan ve
`completed` durumundaki belgeler için:
  1. `full_text` boşsa PDF'ten sayfa işaretli tam metni çıkarıp kaydeder,
     temel metaveriyi (sayfa/karakter/kelime sayısı) yazar.
  2. Özeti yoksa LLM ile kısa özet + metaveri (başlık, ana konular, soru/bölüm sayısı) üretir.

Tekrar çalıştırılabilir (idempotent): yalnızca eksik olanları işler. Token kotasını korumak için
belgeler tek tek ve aralarında bekleyerek işlenir; art arda 3 özet başarısız olursa (kota/anahtar
sorunu olabilir) betik durur, sorun giderilince aynı komutla devam edilir.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import FULL_TEXT_MAX_CHARS, SUMMARY_ENABLED, UPLOAD_DIR  # noqa: E402
from app.database import AsyncSessionLocal, close_db, verify_schema  # noqa: E402
from app.models import Document, DocumentStatus, SummaryStatus  # noqa: E402
from app.services.document_service import (  # noqa: E402
    compute_basic_metadata,
    find_upload,
    generate_document_summary,
)
from app.services.document_parser import parse_document  # noqa: E402
from app.services.pdf_service import build_full_text  # noqa: E402

MAX_CONSECUTIVE_SUMMARY_FAILURES = 3


async def load_candidates(args: argparse.Namespace) -> tuple[list, int, set[int]]:
    """(işlenecek belgeler, hazır olmadığı için atlanan sayısı, veritabanındaki tüm belge id'leri)."""
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    Document.id,
                    Document.filename,
                    Document.status,
                    Document.full_text.is_(None).label("no_text"),
                    Document.summary.is_(None).label("no_summary"),
                    Document.summary_status,
                ).order_by(Document.id)
            )
        ).all()

    selected = []
    not_ready = 0
    retry_statuses = {SummaryStatus.PENDING.value, SummaryStatus.PROCESSING.value}
    if args.retry_failed:
        retry_statuses.add(SummaryStatus.FAILED.value)

    for row in rows:
        if args.document_id is not None and row.id != args.document_id:
            continue
        if row.status != DocumentStatus.COMPLETED.value:
            not_ready += 1
            continue
        need_text = args.force or row.no_text
        need_summary = (
            not args.no_summary
            and SUMMARY_ENABLED
            and (args.force or (row.no_summary and row.summary_status in retry_statuses))
        )
        if need_text or need_summary:
            selected.append((row, need_text, need_summary))

    if args.limit:
        selected = selected[: args.limit]
    return selected, not_ready, {r.id for r in rows}


async def backfill_text(document_id: int) -> str | None:
    """PDF'ten tam metni çıkarıp kaydeder. Hata varsa mesajı, başarıda None döner."""
    path = find_upload(document_id)
    if path is None:
        return f"dosya yok (uploads/{document_id}.*)"
    try:
        parsed = await asyncio.to_thread(parse_document, path)
    except ValueError as exc:
        return str(exc)
    pages = parsed.pages

    full_text, truncated = build_full_text(pages, FULL_TEXT_MAX_CHARS)
    if not full_text:
        return "belgeden metin çıkarılamadı (taranmış/görsel olabilir)"

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            return "belge bu sırada silinmiş"
        existing = doc.doc_metadata or {}
        doc.full_text = full_text
        doc.doc_metadata = {**existing, **compute_basic_metadata(
            len(pages), full_text, truncated,
            file_format=parsed.file_format, ocr_pages=parsed.ocr_pages,
        )}
        await session.commit()
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Mevcut belgeler için tam metin/özet geri doldurma")
    parser.add_argument("--document-id", type=int, help="yalnızca bu belgeyi işle")
    parser.add_argument("--limit", type=int, help="en fazla bu kadar belge işle")
    parser.add_argument("--no-summary", action="store_true", help="LLM özeti üretme, yalnızca tam metin")
    parser.add_argument("--retry-failed", action="store_true", help="özeti 'failed' olanları da dene")
    parser.add_argument("--force", action="store_true", help="dolu alanları da yeniden üret")
    parser.add_argument("--sleep", type=float, default=1.5, help="özetler arası bekleme (sn)")
    parser.add_argument("--dry-run", action="store_true", help="yalnızca ne yapılacağını göster")
    args = parser.parse_args()

    try:
        await verify_schema()
    except RuntimeError as exc:
        raise SystemExit(f"Hata: {exc}") from exc

    if not SUMMARY_ENABLED and not args.no_summary:
        print("Uyarı: SUMMARY_ENABLED=false, özet üretimi atlanacak (yalnızca tam metin).")

    selected, not_ready, known = await load_candidates(args)
    total = len(known)

    # uploads/ içinde veritabanında karşılığı olmayan PDF'ler (bilgi amaçlı)
    orphans = sorted(
        p.name for p in UPLOAD_DIR.glob("*.*") if not (p.stem.isdigit() and int(p.stem) in known)
    )

    print(f"Toplam belge: {total} | işlenecek: {len(selected)} | hazır olmadığı için atlanan: {not_ready}")
    if orphans:
        print(f"Bilgi: uploads/ içinde veritabanında kaydı olmayan {len(orphans)} dosya var "
              f"(atlanır): {', '.join(orphans[:5])}{' ...' if len(orphans) > 5 else ''}")
    if not selected:
        print("Yapılacak iş yok.")
        await close_db()
        return

    if args.dry_run:
        for row, need_text, need_summary in selected:
            todo = [n for n, flag in (("tam metin", need_text), ("özet", need_summary)) if flag]
            print(f"  [{row.id}] {row.filename}: {' + '.join(todo)}")
        await close_db()
        return

    text_ok = summary_ok = failures = 0
    consecutive_summary_failures = 0
    aborted = False

    for index, (row, need_text, need_summary) in enumerate(selected, start=1):
        prefix = f"[{index}/{len(selected)}] #{row.id} {row.filename}"

        if need_text:
            error = await backfill_text(row.id)
            if error:
                failures += 1
                print(f"{prefix}: tam metin YAZILAMADI -> {error}")
                continue
            text_ok += 1
            print(f"{prefix}: tam metin kaydedildi")

        if need_summary:
            if await generate_document_summary(row.id):
                summary_ok += 1
                consecutive_summary_failures = 0
                print(f"{prefix}: özet ve metaveri kaydedildi")
            else:
                failures += 1
                consecutive_summary_failures += 1
                print(f"{prefix}: özet ÜRETİLEMEDİ (ayrıntı için sunucu günlüklerine bak)")
                if consecutive_summary_failures >= MAX_CONSECUTIVE_SUMMARY_FAILURES:
                    print(
                        f"\nArt arda {MAX_CONSECUTIVE_SUMMARY_FAILURES} özet başarısız oldu "
                        "(kota ya da LLM_API_KEY sorunu olabilir). Durduruluyor; sorunu giderip "
                        "aynı komutu `--retry-failed` ile tekrar çalıştır."
                    )
                    aborted = True
                    break
            await asyncio.sleep(args.sleep)

    print(f"\nBitti: tam metin {text_ok}, özet {summary_ok}, hata {failures}.")
    await close_db()
    if aborted or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
