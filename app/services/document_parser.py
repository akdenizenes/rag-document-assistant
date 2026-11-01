"""Çoklu biçim belge okuma katmanı: PDF (+OCR yedeği), DOCX ve TXT -> sayfa/bölüm metinleri.

Tüm biçimler aynı çıktıyı üretir (`ParsedDocument.pages`: `PageText` listesi), bu yüzden
chunking / embedding / tam metin hattı biçimden bağımsız çalışır.

- PDF : pypdf ile metin katmanı; bir sayfanın metni boş veya çok kısaysa (taranmış/görsel sayfa)
        yalnızca o sayfa pdf2image + pytesseract ile OCR'lanır.
- DOCX: python-docx ile paragraflar ve tablolar belge sırasıyla metne çevrilir.
- TXT : UTF-8 (BOM'lu/BOM'suz) -> UTF-16 (BOM varsa) -> cp1254 -> ISO-8859-9 sırasıyla çözülür.

DOCX ve TXT'de gerçek sayfa yoktur; metin ~VIRTUAL_PAGE_CHARS karakterlik "bölümlere" ayrılır.

Hepsi senkron ve CPU yoğundur; async kodda `asyncio.to_thread` ile çağır.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.config import (
    MAX_DOCX_UNCOMPRESSED_MB,
    OCR_DPI,
    OCR_ENABLED,
    OCR_LANGUAGES,
    OCR_MAX_PAGES,
    OCR_MIN_CHARS_PER_PAGE,
    OCR_PAGE_TIMEOUT_SECONDS,
    SUPPORTED_FORMATS,
    VIRTUAL_PAGE_CHARS,
)
from app.services.pdf_service import PageText, extract_pages

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParsedDocument:
    pages: list[PageText]
    file_format: str  # "pdf" | "docx" | "txt"
    has_real_pages: bool  # False: sayfalar yapay "bölümlerdir" (docx/txt)
    ocr_pages: list[int] = field(default_factory=list)  # OCR uygulanan sayfa numaraları
    warnings: list[str] = field(default_factory=list)


def _sanitize(text: str) -> str:
    # PostgreSQL TEXT alanları NUL (\x00) karakterini kabul etmez.
    return text.replace("\x00", "")


# --------------------------------------------------------------------------- #
# Yapay sayfalama (DOCX / TXT)
# --------------------------------------------------------------------------- #
def paginate_text(text: str, page_chars: int = VIRTUAL_PAGE_CHARS) -> list[PageText]:
    """Uzun metni paragraf sınırlarında ~page_chars karakterlik bölümlere ayırır."""
    text = _sanitize(text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text:
        return []

    pages: list[PageText] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            pages.append(PageText(page_number=len(pages) + 1, text="\n".join(buffer).strip()))
            buffer, size = [], 0

    for line in text.split("\n"):
        # Tek bir satır sınırdan uzunsa (satır sonu içermeyen dev metin) sert parçala.
        pieces = [line[i : i + page_chars] for i in range(0, len(line), page_chars)] or [""]
        for piece in pieces:
            if size + len(piece) > page_chars and buffer:
                flush()
            buffer.append(piece)
            size += len(piece) + 1
    flush()
    return pages


# --------------------------------------------------------------------------- #
# TXT
# --------------------------------------------------------------------------- #
def decode_text_bytes(raw: bytes) -> str:
    """Bayt dizisini Türkçe karakterleri koruyarak çözer.

    Sıra: UTF-8 (BOM'u atar) -> UTF-16 (yalnızca BOM varsa) -> cp1254 (Windows-1254) ->
    ISO-8859-9 (Latin-5). Son ikisi eski Türkçe dosyalar içindir; ISO-8859-9 her baytı
    çözebildiğinden zincir asla hata vermez.
    """
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8-sig", "cp1254", "iso-8859-9"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")  # erişilmez; güvenlik ağı


def parse_txt(path: Path) -> ParsedDocument:
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError("Metin dosyası boş.")
    if b"\x00" in raw and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ValueError("Dosya düz metin değil (ikili içerik algılandı).")
    pages = paginate_text(decode_text_bytes(raw))
    return ParsedDocument(pages=pages, file_format="txt", has_real_pages=False)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def _table_to_lines(table) -> list[str]:
    """Tabloyu satır satır metne çevirir: hücreler ' | ' ile ayrılır, birleşik hücreler tekrarlanmaz."""
    lines: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        seen: set[int] = set()
        for cell in row.cells:
            if id(cell._tc) in seen:  # yatay birleştirilmiş hücre aynı nesneyi tekrar döndürür
                continue
            seen.add(id(cell._tc))
            parts = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
            # İç içe tablolar
            for nested in cell.tables:
                parts.extend(_table_to_lines(nested))
            cells.append(" ".join(parts))
        if any(cells):
            lines.append(" | ".join(cells))
    return lines


def _iter_docx_blocks(document) -> list[str]:
    """Paragraf ve tabloları belgedeki GERÇEK sırayla döndürür (python-docx bunu kendiliğinden vermez)."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    lines: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, document).text.strip()
            if text:
                lines.append(text)
        elif tag == "tbl":
            lines.extend(_table_to_lines(Table(child, document)))
    return lines


def _check_docx_zip(path: Path) -> None:
    limit = MAX_DOCX_UNCOMPRESSED_MB * 1024 * 1024
    try:
        with zipfile.ZipFile(path) as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError("Geçerli bir DOCX dosyası değil (word/document.xml yok).")
            if sum(info.file_size for info in zf.infolist()) > limit:
                raise ValueError(
                    f"DOCX açıldığında {MAX_DOCX_UNCOMPRESSED_MB} MB'ı aşıyor; işlenmedi."
                )
    except zipfile.BadZipFile as exc:
        raise ValueError("DOCX dosyası bozuk veya geçerli bir DOCX değil.") from exc


def parse_docx(path: Path) -> ParsedDocument:
    _check_docx_zip(path)
    try:
        from docx import Document as DocxDocument
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as exc:  # pragma: no cover
        raise ValueError("python-docx kurulu değil: pip install python-docx") from exc

    try:
        document = DocxDocument(str(path))
        lines = _iter_docx_blocks(document)
    except (PackageNotFoundError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"DOCX okunamadı: {exc}") from exc

    pages = paginate_text("\n".join(lines))
    return ParsedDocument(pages=pages, file_format="docx", has_real_pages=False)


# --------------------------------------------------------------------------- #
# OCR (yalnızca metin katmanı yetersiz PDF sayfaları için)
# --------------------------------------------------------------------------- #
class OcrUnavailableError(RuntimeError):
    """Tesseract/Poppler veya gerekli Python paketleri eksik."""


_resolved_languages: str | None = None


def ocr_dependencies_error() -> str | None:
    """OCR çalışabilir durumdaysa None, değilse nedenini açıklayan Türkçe mesaj döner."""
    try:
        import pdf2image  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError as exc:
        return f"OCR paketleri eksik ({exc.name}): pip install pytesseract pdf2image"
    if shutil.which("tesseract") is None:
        return "Tesseract kurulu değil (Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-tur)."
    if shutil.which("pdftoppm") is None:
        return "Poppler kurulu değil (Ubuntu: sudo apt install poppler-utils)."
    return None


def _ocr_languages() -> str:
    """İstenen dillerden Tesseract'ta kurulu olanları seçer (yoksa 'eng'e düşer)."""
    global _resolved_languages
    if _resolved_languages is not None:
        return _resolved_languages

    import pytesseract

    installed = set(pytesseract.get_languages(config=""))
    wanted = [lang for lang in OCR_LANGUAGES.split("+") if lang]
    available = [lang for lang in wanted if lang in installed]
    missing = [lang for lang in wanted if lang not in installed]
    if missing:
        logger.warning(
            "Tesseract dil paketi eksik: %s (Ubuntu: sudo apt install %s).",
            ", ".join(missing),
            " ".join(f"tesseract-ocr-{m}" for m in missing),
        )
    if not available:
        if "eng" not in installed:
            raise OcrUnavailableError("Tesseract'ta hiçbir OCR dil paketi kurulu değil.")
        available = ["eng"]
    _resolved_languages = "+".join(available)
    return _resolved_languages


def ocr_pdf_page(path: Path, page_number: int) -> str:
    """Tek bir PDF sayfasını görüntüye çevirip OCR uygular (bellek için sayfa sayfa)."""
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(path), dpi=OCR_DPI, first_page=page_number, last_page=page_number, grayscale=True
    )
    if not images:
        return ""
    try:
        text = pytesseract.image_to_string(
            images[0], lang=_ocr_languages(), timeout=OCR_PAGE_TIMEOUT_SECONDS
        )
    finally:
        for image in images:
            image.close()
    return _sanitize(text)


# --------------------------------------------------------------------------- #
# PDF (metin katmanı + OCR yedeği)
# --------------------------------------------------------------------------- #
def _needs_ocr(text: str) -> bool:
    return len("".join(text.split())) < OCR_MIN_CHARS_PER_PAGE


def parse_pdf(path: Path) -> ParsedDocument:
    pages = extract_pages(path)
    result = ParsedDocument(pages=pages, file_format="pdf", has_real_pages=True)

    weak = [p for p in pages if _needs_ocr(p.text)]
    if not weak:
        return result

    if not OCR_ENABLED:
        result.warnings.append(f"{len(weak)} sayfada metin yok; OCR kapalı (OCR_ENABLED=false).")
        return result

    problem = ocr_dependencies_error()
    if problem:
        logger.warning("OCR atlandı: %s", problem)
        result.warnings.append(f"{len(weak)} sayfada metin yok ve OCR kullanılamıyor: {problem}")
        return result

    logger.info("%s: %d/%d sayfa OCR'lanacak.", path.name, len(weak), len(pages))
    for index, page in enumerate(weak):
        if index >= OCR_MAX_PAGES:
            result.warnings.append(
                f"OCR sayfa sınırı (OCR_MAX_PAGES={OCR_MAX_PAGES}) aşıldı; "
                f"{page.page_number}. sayfadan sonrası OCR'lanmadı."
            )
            break
        try:
            text = ocr_pdf_page(path, page.page_number)
        except OcrUnavailableError as exc:
            result.warnings.append(str(exc))
            break
        except Exception as exc:  # tek sayfanın OCR hatası tüm belgeyi düşürmesin
            logger.warning("%s sayfa %d OCR hatası: %s", path.name, page.page_number, exc)
            result.warnings.append(f"{page.page_number}. sayfa OCR hatası: {exc}")
            continue
        if not _needs_ocr(text) or len(text.strip()) > len(page.text.strip()):
            page.text = text
            if text.strip():
                result.ocr_pages.append(page.page_number)
    return result


# --------------------------------------------------------------------------- #
# Giriş noktası
# --------------------------------------------------------------------------- #
_PARSERS = {".pdf": parse_pdf, ".docx": parse_docx, ".txt": parse_txt}


def parse_document(path: Path) -> ParsedDocument:
    """Dosyayı uzantısına göre okur. Desteklenmeyen/okunamayan dosyada ValueError fırlatır."""
    extension = path.suffix.lower()
    parser = _PARSERS.get(extension)
    if parser is None:
        supported = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(f"Desteklenmeyen dosya biçimi: {extension or '(uzantısız)'} ({supported})")
    return parser(path)
