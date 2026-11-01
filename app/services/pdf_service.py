"""PDF'ten sayfa sayfa metin çıkarma (pypdf) ve tam metin birleştirme."""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError


@dataclass(slots=True)
class PageText:
    page_number: int  # 1'den başlar
    text: str


def extract_pages(path: Path) -> list[PageText]:
    """PDF'in her sayfasından metni çıkarır.

    Senkron ve CPU yoğun bir işlemdir; async kodda `asyncio.to_thread` ile çağır.
    Okunamayan/şifreli dosyalarda ValueError fırlatır.
    """
    try:
        reader = PdfReader(str(path))

        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF parola korumalı, okunamadı.")

        pages: list[PageText] = []
        for number, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            # PostgreSQL TEXT alanları NUL (\x00) karakterini kabul etmez.
            pages.append(PageText(page_number=number, text=raw.replace("\x00", "")))
    except PyPdfError as exc:
        raise ValueError(f"PDF okunamadı: {exc}") from exc

    return pages


def page_marker(page_number: int) -> str:
    """Tam metindeki sayfa ayracı. Prompt kuralları ve LLM bu biçime güvenir."""
    return f"--- Sayfa {page_number} ---"


def _clean_page_text(text: str) -> str:
    """Satır yapısını KORUYARAK gürültüyü temizler.

    chunking.normalize_text tek satır sonlarını boşluğa çevirir; sayma/listeleme için tam
    metinde satırlar ("Soru 1", "2)" gibi) kritik olduğundan burada satırlar saklanır.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_full_text(pages: list[PageText], max_chars: int) -> tuple[str, bool]:
    """Sayfaları `--- Sayfa N ---` ayraçlarıyla tek metne birleştirir.

    Boş sayfalar atlanır. Toplam uzunluk `max_chars`'ı aşacaksa sayfa sınırında kesilir.
    Döner: (tam metin, kısaltıldı mı)
    """
    parts: list[str] = []
    used = 0
    for page in pages:
        body = _clean_page_text(page.text)
        if not body:
            continue
        block = f"{page_marker(page.page_number)}\n{body}"
        if parts and used + len(block) > max_chars:
            parts.append(
                f"[... Metin uzunluk sınırı nedeniyle burada kesildi; "
                f"{page.page_number}. sayfadan itibaren olan kısım saklanmadı ...]"
            )
            return "\n\n".join(parts), True
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts), False
