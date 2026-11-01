"""Prompt yardımcıları ve özet ayrıştırıcı testleri (saf fonksiyonlar)."""

import pytest

from app.services.prompts import (
    FULL_TEXT_ROUTE_RULES,
    SUMMARY_ROUTE_RULES,
    SYSTEM_PROMPT,
    DocContext,
    build_full_text_context,
    build_messages,
    build_summary_context,
    build_summary_messages,
    parse_summary_response,
)


def test_parse_clean_json():
    raw = (
        '{"title": "Ağ Ödevi", "document_type": "sınav çözümü", "summary": "TCP ve UDP anlatılıyor.", '
        '"main_topics": ["TCP", "UDP", ""], "question_count": 12, "section_count": "3"}'
    )
    summary, meta = parse_summary_response(raw)
    assert summary == "TCP ve UDP anlatılıyor."
    assert meta["title"] == "Ağ Ödevi"
    assert meta["main_topics"] == ["TCP", "UDP"]
    assert meta["question_count"] == 12
    assert meta["section_count"] == 3  # "3" -> 3
    assert "parse_warning" not in meta


def test_parse_fenced_json_with_noise():
    raw = 'İşte sonuç:\n```json\n{"summary": "Kısa özet", "question_count": null}\n```'
    summary, meta = parse_summary_response(raw)
    assert summary == "Kısa özet"
    assert meta["question_count"] is None
    assert meta["main_topics"] == []


def test_parse_invalid_counts():
    _, meta = parse_summary_response('{"summary": "x", "question_count": -4, "section_count": true}')
    assert meta["question_count"] is None and meta["section_count"] is None


def test_parse_falls_back_to_raw_text():
    summary, meta = parse_summary_response("Bu belge TCP'yi anlatır, JSON değil.")
    assert summary.startswith("Bu belge TCP")
    assert meta == {"parse_warning": True}

    summary, meta = parse_summary_response('{"title": "özet alanı yok"}')
    assert meta.get("parse_warning") is True


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_summary_response("   ")


def test_summary_messages_contain_text_and_schema():
    msgs = build_summary_messages("ders.pdf", "--- Sayfa 1 ---\nmerhaba")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "merhaba" in msgs[1]["content"] and "question_count" in msgs[1]["content"]


def test_summary_context():
    doc = DocContext(
        document_id=1, filename="a.pdf", page_count=3, summary="Özet metni",
        doc_metadata={"title": "Başlık", "main_topics": ["A", "B"], "document_type": "ders notu"},
    )
    ctx = build_summary_context([doc])
    assert "[1] Kaynak: a.pdf (3 sayfa)" in ctx
    assert "Ana konular: A; B" in ctx and "Özet metni" in ctx


def test_full_text_context_truncation():
    text = "\n".join(f"satır {i}" for i in range(1000))
    doc = DocContext(document_id=1, filename="a.pdf", page_count=2, full_text=text)
    ctx, truncated = build_full_text_context(doc, max_chars=500)
    assert truncated and "kesildi" in ctx and len(ctx) < 700
    ctx, truncated = build_full_text_context(doc, max_chars=10**6)
    assert not truncated and "satır 999" in ctx


def test_build_messages_extra_rules():
    base = build_messages("soru?", "ctx", [])
    assert base[0]["content"] == SYSTEM_PROMPT
    full = build_messages("soru?", "ctx", [], extra_rules=FULL_TEXT_ROUTE_RULES)
    assert "numaralandırarak listele" in full[0]["content"]
    assert SUMMARY_ROUTE_RULES.startswith("Ek kurallar")
    assert full[-1]["content"].endswith("SORU: soru?")
