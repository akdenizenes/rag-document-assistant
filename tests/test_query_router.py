"""Kural tabanlı soru yönlendiricisi testleri (veritabanı/ağ gerekmez)."""

import pytest

from app.services.query_router import classify_question, normalize_question

SUMMARY = [
    "Bu belgeyi özetle",
    "özetle",
    "Belgenin özetini çıkarır mısın?",
    "ozetle lutfen",
    "Bu doküman ne anlatıyor?",
    "Bu pdf neyi anlatıyor",
    "Bu dosya ne hakkında?",
    "Bu notlar neyle ilgili?",
    "Belgenin konusu nedir?",
    "Genel olarak nedir bu belge?",
    "Belgenin ana fikri nedir",
    "Kısaca anlatır mısın?",
    "Bana genel bir bakış sun",
]

FULL_TEXT = [
    "Kaç soru var?",
    "kac soru var",
    "Bu sınavda kaç tane soru var?",
    "Kaç farklı konu işlenmiş?",
    "Toplam kaç bölüm var",
    "Soru sayısı kaç?",
    "Sayfa sayısı nedir",
    "Bölümleri listele",
    "Tüm konuları listele",
    "Bütün soruları sırala",
    "tüm konular neler",
    "Sorular neler?",
    "Belgede hangi konular var?",
    "Hangi bölümler ele alınıyor?",
    "Belgede neler var?",
    "İçindekiler",
    "Soruları say",
    "Tüm konuları özetle",  # özet + tüm kapsam -> tam metin
    "Kaç deney yapılmış ve hangileri?",
]

HYBRID = [
    "Teslim süresi kaç gündür?",
    "TCP üç yollu el sıkışma nasıl çalışır?",
    "Ohm kanunu nedir?",
    "3. sorunun cevabı nedir?",
    "Soru 5'i çöz",
    "3. soruyu özetle",
    "Bölüm 2 ne anlatıyor?",
    "Sayfa 4'te ne yazıyor",
    "Deney 3'te kaç adım var?",
    "Laboratuvarda hangi cihaz kullanılıyor?",
    "Vize tarihi ne zaman?",
    "Bu formülün birimi nedir?",
    "Peki ya sonrası?",
    "Kaç volt gerekir?",  # 'kaç' var ama öğe yok
    "Kaç puan alırım?",
    "Kaç adımda çözülür?",
    "Birinci bölümü özetle",
    "Son soruyu çöz",
    "Sayfa 2'yi özetle",
]


@pytest.mark.parametrize("question", SUMMARY)
def test_summary_route(question):
    assert classify_question(question) == "summary", question


@pytest.mark.parametrize("question", FULL_TEXT)
def test_full_text_route(question):
    assert classify_question(question) == "full_text", question


@pytest.mark.parametrize("question", HYBRID)
def test_hybrid_route(question):
    assert classify_question(question) == "hybrid", question


def test_turkish_normalization():
    assert normalize_question("İÇİNDEKİLER  Işık") == "icindekiler isik"
    assert normalize_question("Özetle'") == "ozetle"
