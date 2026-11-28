"""Kayıt sırasında girilen ad soyad için temizleme ve doğrulama kuralları."""

from __future__ import annotations

import unicodedata

_ALLOWED_PUNCTUATION = frozenset(" -'’.")
_MIN_LETTERS = 2


def clean_full_name(value: str) -> str:
    """Boşlukları sadeleştirip adı döner; kurallara uymuyorsa ValueError fırlatır."""
    name = " ".join(unicodedata.normalize("NFC", value).split())
    if sum(ch.isalpha() for ch in name) < _MIN_LETTERS:
        raise ValueError("Ad Soyad en az 2 harf içermelidir.")
    if not name[0].isalpha() or any(
        not (ch.isalpha() or ch in _ALLOWED_PUNCTUATION) for ch in name
    ):
        raise ValueError("Ad Soyad yalnızca harf, boşluk, tire, kesme işareti ve nokta içerebilir.")
    return name
