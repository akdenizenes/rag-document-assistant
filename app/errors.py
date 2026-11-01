"""Merkezi hata yönetimi: tutarlı JSON hata yanıtları, istek kimliği (request id), günlükleme.

Amaç: veritabanı ya da LLM sorunları ham yığın izi (stack trace) yerine anlaşılır bir mesaj ve
doğru HTTP durum koduyla dönsün; arayüz (Streamlit) bu yüzden çökmesin. Tüm hata yanıtları
`{"detail": "...", "request_id": "..."}` biçimindedir (FastAPI'nin varsayılan `detail` alanıyla uyumlu).
"""

import logging
import re
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.services.llm_service import LLMError

logger = logging.getLogger("app.errors")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
REQUEST_ID_HEADER = "X-Request-ID"

DB_UNAVAILABLE_MESSAGE = "Veritabanına şu anda ulaşılamıyor. Lütfen biraz sonra tekrar dene."
DB_ERROR_MESSAGE = "Veritabanı işlemi sırasında bir hata oluştu."
UNEXPECTED_MESSAGE = "Beklenmeyen bir sunucu hatası oluştu."


def sanitize_request_id(value: str | None) -> str:
    """Gelen X-Request-ID güvenliyse kullanılır, değilse yenisi üretilir (log enjeksiyonunu önler)."""
    if value and _REQUEST_ID_RE.match(value):
        return value
    return uuid.uuid4().hex[:12]


class RequestIdMiddleware:
    """Her isteğe bir kimlik atar ve yanıta `X-Request-ID` başlığı olarak ekler.

    Saf ASGI middleware'dir (BaseHTTPMiddleware değil): akışlı (SSE) yanıtları tamponlamaz.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = next(
            (v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"x-request-id"),
            None,
        )
        request_id = sanitize_request_id(incoming)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_id)


def _request_id(request: Request) -> str | None:
    return request.scope.get("state", {}).get("request_id")


def _error(status_code: int, detail: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": _request_id(request)},
    )


async def _llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    logger.warning("LLM hatası [%s]: %s", _request_id(request), exc)
    return _error(exc.status_code, str(exc), request)


async def _db_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Veritabanı erişilemiyor [%s]: %r", _request_id(request), exc)
    return _error(503, DB_UNAVAILABLE_MESSAGE, request)


async def _db_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("Veritabanı hatası [%s]", _request_id(request), exc_info=exc)
    return _error(500, DB_ERROR_MESSAGE, request)


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    # İç ayrıntılar (yığın izi, SQL) istemciye SIZDIRILMAZ; yalnızca request_id ile günlükte aranır.
    logger.exception("Yakalanmamış hata [%s] %s %s", _request_id(request), request.method, request.url.path, exc_info=exc)
    return _error(500, UNEXPECTED_MESSAGE, request)


def install_error_handlers(app: FastAPI) -> None:
    """İstek kimliği middleware'ini ve hata yakalayıcıları uygulamaya ekler."""
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(LLMError, _llm_error_handler)
    # Bağlantı/havuz sorunları -> 503 (geçici), diğer SQLAlchemy hataları -> 500.
    for exc_type in (OperationalError, InterfaceError, PoolTimeoutError, ConnectionError):
        app.add_exception_handler(exc_type, _db_unavailable_handler)
    app.add_exception_handler(SQLAlchemyError, _db_error_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
