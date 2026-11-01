"""Kural tabanlı (regex / anahtar kelime) Türkçe soru yönlendirici.

Bir soruyu üç rotadan birine atar:

  summary   : "özetle", "ne anlatıyor", "genel olarak nedir" ...
              -> arama yok, doğrudan `documents.summary` bağlama verilir.
  full_text : "kaç soru var", "bölümleri listele", "tüm konular" ...
              -> vektör araması yerine belgenin `full_text` alanı bağlama verilir.
  hybrid    : diğer her şey (spesifik sorular) -> hibrit arama (vektör + FTS).

Saf fonksiyonlardır (veritabanı/ağ yok), bu yüzden kolayca test edilir. LLM çağrısı yapılmaz,
yani yönlendirmenin token maliyeti sıfırdır.

Eşleştirme Türkçe karakterleri ASCII'ye katlanmış metin üzerinde yapılır; böylece
"özetle" / "ozetle", "kaç" / "kac", "İçindekiler" / "icindekiler" aynı kuralla yakalanır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Route = Literal["summary", "full_text", "hybrid"]

ROUTE_SUMMARY: Route = "summary"
ROUTE_FULL_TEXT: Route = "full_text"
ROUTE_HYBRID: Route = "hybrid"

_TR_FOLD = str.maketrans(
    {
        "İ": "i", "I": "i", "ı": "i",
        "Ç": "c", "ç": "c",
        "Ğ": "g", "ğ": "g",
        "Ö": "o", "ö": "o",
        "Ş": "s", "ş": "s",
        "Ü": "u", "ü": "u",
        "’": "", "'": "", "`": "", "´": "",
    }
)


def normalize_question(text: str) -> str:
    """Küçük harfe çevirir, Türkçe karakterleri ASCII'ye katlar, boşlukları sadeleştirir."""
    return re.sub(r"\s+", " ", text.translate(_TR_FOLD).lower()).strip()


# Sayılabilir / listelenebilir "öğe" kökleri (ek almış hâlleri \w* ile yakalanır).
_ITEM = (
    r"(?:soru|bolum|madde|konu|baslik|kisim|unite|deney|alistirma|problem|"
    r"ornek|kavram|tablo|sekil|grafik|formul)\w*"
)
_ITEM_OR_PAGE = rf"(?:{_ITEM}|sayfa\w*)"
_ANY_WORDS = r"(?:\w+\s+){0,2}?"  # araya girebilecek en fazla 2 kelime

# --- Dar kapsam: belirli bir soru/bölüm/sayfa hakkında ise genel tarama DEĞİLDİR -------------
# "3. soruyu özetle", "bölüm 2'de kaç adım var", "sayfa 5 ne anlatıyor" -> spesifik soru.
_ORDINALS = (
    r"(?:birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|"
    r"onuncu|ilk|son|sonraki|onceki)"
)
_NARROW_SCOPE = re.compile(
    rf"\b\d+\s*(?:\.|inci|nci|uncu|ncu)\s*{_ITEM_OR_PAGE}"  # 3. soru, 2nci bolum
    rf"|\b(?:soru|bolum|madde|kisim|unite|deney|adim|alistirma|problem|ornek|sayfa)\w*\s*(?:no\s*)?\d+"
    rf"|\b{_ORDINALS}\s+(?:\w+\s+)?{_ITEM_OR_PAGE}"  # birinci bolum, son soru, ilk 5 soru
)

# --- Sayma / genel tarama ------------------------------------------------------------------
_FULL_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        # "kaç soru var", "kaç tane soru", "kaç farklı konu", "kaç sorudan oluşuyor"
        ("kac-oge", rf"\bkac\s+(?:(?:adet|tane|farkli|ayri|toplam)\s+)*{_ITEM_OR_PAGE}"),
        # "soru sayısı", "toplam bölüm sayısı", "sayfa sayısı"
        ("oge-sayisi", rf"\b{_ITEM_OR_PAGE}\s+sayisi"),
        ("toplam-kac", r"\btoplam\s+kac\b"),
        # "tüm konular", "bütün sorular", "tüm bölümleri"
        ("tum-oge", rf"\b(?:tum|butun|tamami)\s+{_ANY_WORDS}{_ITEM}"),
        # "hangi konular var", "hangi bölümler ele alınıyor"
        (
            "hangi-oge-var",
            rf"\bhangi\s+{_ANY_WORDS}{_ITEM}\s+{_ANY_WORDS}"
            r"(?:var|yer\s+al|ele\s+al|islen|bulun|kapsa|icer)\w*",
        ),
        ("oge-neler", rf"\b{_ITEM}\s+neler\b"),
        ("neler-var", r"\bneler\s+(?:var|yer\s+al|islen|ele\s+al|anlat|kapsa)\w*"),
        ("icindekiler", r"\bicindekiler\b"),
    )
)
# "listele", "sırala", "say" + bir öğe adı ("bölümleri listele", "soruları say")
_LIST_VERB = re.compile(r"\b(?:liste\w*|sirala\w*|say|sayar\w*|saysana|sayin)\b")
_HAS_ITEM = re.compile(rf"\b{_ITEM}")

# --- Özet istekleri --------------------------------------------------------------------------
_SUMMARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("ozet", r"\bozet\w*"),  # özetle, özeti, özet çıkar, özetler misin
        ("ne-anlatiyor", r"\bne(?:yi)?\s+anlat\w*"),  # ne anlatıyor, neyi anlatıyor
        ("ne-hakkinda", r"\bne(?:yle)?\s+(?:hakkinda|ilgili)\b"),  # ne hakkında, neyle ilgili
        ("konusu-ne", r"\bkonusu\s+(?:ne|nedir)\b"),
        ("genel-olarak", r"\bgenel\s+olarak\s+(?:nedir|ne\b|anlat\w*|neyi|nelerden)"),
        ("genel-bakis", r"\bgenel\s+(?:bir\s+)?(?:bakis|hatlari\w*|cerceve\w*)"),
        ("ana-fikir", r"\bana\s+(?:fikir|fikri|fikrini|fikirleri)\b"),
        ("kisaca-anlat", r"\bkisaca\s+(?:anlat|acikla|bahset)\w*"),
    )
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    reason: str  # günlükte hangi kuralın tetiklendiğini görmek için


def decide_route(question: str) -> RouteDecision:
    """Soruyu yönlendirir ve gerekçesini döner."""
    text = normalize_question(question)

    # 1) Belirli bir soru/bölüm/sayfa soruluyorsa genel tarama değil, spesifik sorudur.
    if _NARROW_SCOPE.search(text):
        return RouteDecision(ROUTE_HYBRID, "dar-kapsam")

    # 2) Sayma / listeleme / genel tarama. Özetten önce bakılır: "tüm konuları özetle" gibi
    #    isteklerde kısa özet yetmez, tam kapsam gerekir.
    for name, pattern in _FULL_TEXT_PATTERNS:
        if pattern.search(text):
            return RouteDecision(ROUTE_FULL_TEXT, name)
    if _LIST_VERB.search(text) and _HAS_ITEM.search(text):
        return RouteDecision(ROUTE_FULL_TEXT, "listele-oge")

    # 3) Özet istekleri.
    for name, pattern in _SUMMARY_PATTERNS:
        if pattern.search(text):
            return RouteDecision(ROUTE_SUMMARY, name)

    # 4) Varsayılan: spesifik soru -> arama motoru.
    return RouteDecision(ROUTE_HYBRID, "varsayilan")


def classify_question(question: str) -> Route:
    """Soruyu `summary`, `full_text` veya `hybrid` rotasına atar."""
    return decide_route(question).route
