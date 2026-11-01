"""İstek sınırlama çekirdeği: kayan pencere (sliding window) sayacı.

Saf Python'dur (FastAPI/veritabanı yok), bu yüzden kolayca test edilir. FastAPI bağlantısı
`app/rate_limit.py` içindedir.

Sayaç SÜREÇ İÇİ (bellek) tutulur: tek bir uvicorn süreci için doğrudur. `--workers N` ile
birden çok süreç çalıştırırsan her süreç kendi sayacını tutar (efektif sınır ~N katı olur);
o durumda ortak bir depo (Redis) kullanan `slowapi` / `limits` gibi bir çözüme geçilmelidir.
Sınırlama mantığı `SlidingWindowLimiter.hit` arkasında olduğu için yalnızca bu sınıf değişir.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

_UNIT_SECONDS = {
    "second": 1, "sec": 1, "s": 1,
    "minute": 60, "min": 60, "m": 60,
    "hour": 3600, "h": 3600,
    "day": 86400, "d": 86400,
}
_RATE_RE = re.compile(
    r"^\s*(\d+)\s*/\s*(?:(\d+)\s*)?(second|sec|s|minute|min|m|hour|h|day|d)s?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Rate:
    count: int  # pencere başına izin verilen istek sayısı
    window: int  # pencere uzunluğu (saniye)

    def __str__(self) -> str:
        return f"{self.count}/{self.window}s"


def parse_rate(spec: str) -> Rate:
    """"20/minute", "5/second", "100/hour", "30/10seconds" biçimlerini çözer."""
    match = _RATE_RE.match(spec or "")
    if not match:
        raise ValueError(
            f"Geçersiz istek sınırı: {spec!r}. Örnekler: '20/minute', '5/second', '100/hour'."
        )
    count = int(match.group(1))
    multiplier = int(match.group(2) or 1)
    window = multiplier * _UNIT_SECONDS[match.group(3).lower()]
    if count < 1 or window < 1:
        raise ValueError(f"İstek sınırı en az 1 olmalı: {spec!r}")
    return Rate(count=count, window=window)


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # engellendiyse kaç saniye sonra tekrar denenebilir (>= 1), aksi halde 0


class SlidingWindowLimiter:
    """Anahtar başına son `window` saniyedeki istek zamanlarını tutar."""

    def __init__(
        self, *, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000
    ) -> None:
        self._clock = clock
        self._max_keys = max_keys
        # anahtar -> (pencere, istek zamanları)
        self._hits: dict[str, tuple[int, deque[float]]] = {}

    def hit(self, key: str, rate: Rate) -> Decision:
        """Bir isteği kaydetmeyi dener. Engellenen istekler sayaca EKLENMEZ."""
        now = self._clock()
        entry = self._hits.get(key)
        if entry is None:
            if len(self._hits) >= self._max_keys:
                self._prune(now)
            entry = self._hits[key] = (rate.window, deque())
        _, hits = entry

        cutoff = now - rate.window
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= rate.count:
            retry_after = max(1, math.ceil(hits[0] + rate.window - now))
            return Decision(False, rate.count, 0, retry_after)

        hits.append(now)
        return Decision(True, rate.count, rate.count - len(hits), 0)

    def _prune(self, now: float) -> None:
        """Süresi dolmuş anahtarları siler; bellek sınırsız büyümesin."""
        expired = [
            key
            for key, (window, hits) in self._hits.items()
            if not hits or hits[-1] <= now - window
        ]
        for key in expired:
            del self._hits[key]

    def reset(self) -> None:
        self._hits.clear()
