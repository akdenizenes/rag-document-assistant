"""Metni, anlamı mümkün olduğunca bozmadan örtüşmeli parçalara böler."""

import re
from dataclasses import dataclass

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.services.pdf_service import PageText

# Bölme noktası tercih sırası: paragraf > satır > cümle > virgül > boşluk
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass(slots=True)
class Chunk:
    chunk_index: int
    page_number: int
    content: str


def normalize_text(text: str) -> str:
    """PDF'ten gelen metindeki satır sonu/boşluk gürültüsünü temizler."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # satır sonunda bölünmüş kelimeleri birleştir
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # tek satır sonu = aynı paragraf
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Metni en fazla `chunk_size` karakterlik, `overlap` kadar örtüşen parçalara böler."""
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Pencerenin ikinci yarısındaki en iyi doğal bölme noktasını ara.
            window_start = start + chunk_size // 2
            for sep in _SEPARATORS:
                idx = text.rfind(sep, window_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        # Bir sonraki parça kelimenin ortasından başlamasın.
        space = text.find(" ", next_start, end)
        if space != -1:
            next_start = space + 1
        start = next_start

    return chunks


def build_chunks(pages: list[PageText]) -> list[Chunk]:
    """Sayfaları parçalara böler; her parça kaynak sayfa numarasını taşır."""
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for piece in split_text(page.text):
            chunks.append(Chunk(chunk_index=index, page_number=page.page_number, content=piece))
            index += 1
    return chunks
