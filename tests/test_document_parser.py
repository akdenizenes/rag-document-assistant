"""document_parser: TXT kodlamaları, DOCX paragraf/tablo, PDF + OCR yedeği."""

import shutil

import pytest

from app.services.document_parser import (
    decode_text_bytes,
    paginate_text,
    parse_document,
)

HAS_OCR = shutil.which("tesseract") and shutil.which("pdftoppm")


# ---------------- TXT ----------------
@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1254", "iso-8859-9"]
)
def test_txt_turkish_encodings(tmp_path, encoding):
    text = "Çağrı şişeyi öğrenci ğüzel İstanbul ıspanak"
    path = tmp_path / "a.txt"
    path.write_bytes(text.encode(encoding))
    parsed = parse_document(path)
    assert parsed.file_format == "txt" and not parsed.has_real_pages
    assert parsed.pages[0].text == text


def test_txt_binary_and_empty_rejected(tmp_path):
    (tmp_path / "b.txt").write_bytes(b"abc\x00\x01\x02")
    with pytest.raises(ValueError):
        parse_document(tmp_path / "b.txt")
    (tmp_path / "e.txt").write_bytes(b"  \n ")
    with pytest.raises(ValueError):
        parse_document(tmp_path / "e.txt")


def test_decode_never_raises():
    assert isinstance(decode_text_bytes(bytes(range(256))), str)


def test_paginate_splits_and_numbers():
    text = "\n".join(f"satır {i} " + "x" * 50 for i in range(200))
    pages = paginate_text(text, page_chars=1000)
    assert len(pages) > 5
    assert [p.page_number for p in pages] == list(range(1, len(pages) + 1))
    assert all(len(p.text) <= 1000 for p in pages)


def test_paginate_giant_single_line():
    pages = paginate_text("a" * 2500, page_chars=1000)
    assert [len(p.text) for p in pages] == [1000, 1000, 500]


# ---------------- DOCX ----------------
def _make_docx(path):
    from docx import Document

    d = Document()
    d.add_paragraph("Giriş paragrafı")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Ad", "Not"
    t.cell(1, 0).text, t.cell(1, 1).text = "Ayşe", "90"
    d.add_paragraph("Tablodan sonra")
    d.save(str(path))


def test_docx_paragraphs_and_tables_in_order(tmp_path):
    path = tmp_path / "a.docx"
    _make_docx(path)
    text = parse_document(path).pages[0].text
    assert text.splitlines() == ["Giriş paragrafı", "Ad | Not", "Ayşe | 90", "Tablodan sonra"]


def test_docx_merged_cells_not_duplicated(tmp_path):
    from docx import Document

    d = Document()
    t = d.add_table(rows=1, cols=3)
    t.cell(0, 0).merge(t.cell(0, 1)).text = "Birleşik"
    t.cell(0, 2).text = "Son"
    path = tmp_path / "m.docx"
    d.save(str(path))
    assert parse_document(path).pages[0].text == "Birleşik | Son"


def test_docx_invalid(tmp_path):
    p = tmp_path / "x.docx"
    p.write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        parse_document(p)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("a")
    with pytest.raises(ValueError):
        parse_document(p)


# ---------------- PDF + OCR ----------------
def _text_pdf(path, text):
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    c.drawString(72, 750, text)
    c.save()


def _scanned_pdf(path, lines, pages=1):
    """Metin katmanı OLMAYAN (yalnızca görüntü) PDF üretir."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=48)
    imgs = []
    for n in range(pages):
        img = Image.new("RGB", (1700, 600), "white")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((60, 60 + i * 90), f"{line} {n + 1}", fill="black", font=font)
        imgs.append(img)
    imgs[0].save(str(path), "PDF", resolution=200, save_all=True, append_images=imgs[1:])


def test_pdf_with_text_layer_skips_ocr(tmp_path):
    p = tmp_path / "t.pdf"
    _text_pdf(p, "Bu bir metin katmanli PDF belgesidir")
    parsed = parse_document(p)
    assert parsed.ocr_pages == [] and parsed.has_real_pages
    assert "metin katmanli" in parsed.pages[0].text


@pytest.mark.skipif(not HAS_OCR, reason="tesseract/poppler kurulu değil")
def test_scanned_pdf_falls_back_to_ocr(tmp_path):
    p = tmp_path / "s.pdf"
    _scanned_pdf(p, ["Invoice total amount", "Quarterly report summary"], pages=2)
    parsed = parse_document(p)
    assert parsed.ocr_pages == [1, 2]
    assert "invoice" in parsed.pages[0].text.lower()
    assert "quarterly" in parsed.pages[1].text.lower()


def test_ocr_disabled_returns_empty_with_warning(tmp_path, monkeypatch):
    import app.services.document_parser as dp

    monkeypatch.setattr(dp, "OCR_ENABLED", False)
    p = tmp_path / "s.pdf"
    _scanned_pdf(p, ["Hello world"])
    parsed = dp.parse_pdf(p)
    assert parsed.ocr_pages == [] and parsed.warnings
