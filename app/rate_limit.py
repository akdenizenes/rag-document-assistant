"""FastAPI için istek sınırlama bağımlılıkları (kullanıcı başına, süreç içi).

Kullanım:
    @router.post("/chat", dependencies=[Depends(limit_chat)])

Sınır aşılırsa 429 döner ve `Retry-After` başlığı eklenir. Sayaç anahtarı kullanıcı kimliğidir
(şu an tek "demo" kullanıcı; JWT eklenince otomatik olarak kullanıcı başına çalışır).
Sınırlar `.env` içindeki RATE_LIMIT_* ayarlarından gelir (bkz. .env.example).
"""

import logging

from fastapi import Depends, HTTPException, Response, status

from app.config import (
    RATE_LIMIT_CHAT,
    RATE_LIMIT_DEFAULT,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_UPLOAD,
)
from app.dependencies import get_current_user
from app.models import User
from app.services.rate_limiter import SlidingWindowLimiter, parse_rate

logger = logging.getLogger(__name__)

limiter = SlidingWindowLimiter()

# Geçersiz ayar uygulama açılışında (import sırasında) net bir hatayla yakalanır.
_RATES = {
    "chat": parse_rate(RATE_LIMIT_CHAT),
    "upload": parse_rate(RATE_LIMIT_UPLOAD),
    "default": parse_rate(RATE_LIMIT_DEFAULT),
}


def rate_limit(scope: str):
    """`scope` ("chat" | "upload" | "default") için bir FastAPI bağımlılığı üretir."""
    rate = _RATES[scope]

    async def dependency(
        response: Response, user: User = Depends(get_current_user)
    ) -> None:
        if not RATE_LIMIT_ENABLED:
            return
        decision = limiter.hit(f"{scope}:{user.id}", rate)
        if not decision.allowed:
            logger.warning("İstek sınırı aşıldı: kapsam=%s kullanıcı=%s", scope, user.id)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Çok fazla istek gönderdin. Lütfen {decision.retry_after} saniye sonra tekrar dene.",
                headers={
                    "Retry-After": str(decision.retry_after),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)

    return dependency


limit_chat = rate_limit("chat")
limit_upload = rate_limit("upload")
limit_default = rate_limit("default")
