"""PDF'ten sayfa sayfa metin çıkarma (pypdf)."""

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
