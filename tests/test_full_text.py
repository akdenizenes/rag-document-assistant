"""Tam metin birleştirme testleri (pypdf gerekir, PDF dosyası gerekmez)."""

from app.services.pdf_service import PageText, build_full_text, page_marker


def test_markers_and_line_structure_preserved():
    pages = [PageText(1, "Soru 1)  Nedir?\r\nCevap: a"), PageText(2, "  \n "), PageText(3, "Soru 2) b\x00")]
    text, truncated = build_full_text(pages, 10_000)
    assert not truncated
    assert text == (
        f"{page_marker(1)}\nSoru 1) Nedir?\nCevap: a\n\n"  # satır sonu korunur (sayma için kritik)
        f"{page_marker(3)}\nSoru 2) b"  # boş sayfa atlanır, NUL temizlenir, numara gerçek kalır
    )


def test_truncates_on_page_boundary():
    pages = [PageText(i, "x" * 100) for i in range(1, 6)]
    text, truncated = build_full_text(pages, 250)
    assert truncated
    assert page_marker(2) in text and page_marker(4) not in text
    assert "kesildi" in text


def test_first_page_is_always_kept_even_if_over_limit():
    text, truncated = build_full_text([PageText(1, "y" * 500)], 100)
    assert not truncated and page_marker(1) in text


def test_empty_document():
    assert build_full_text([PageText(1, "  ")], 100) == ("", False)
