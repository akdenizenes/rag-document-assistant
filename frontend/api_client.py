"""FastAPI backend'i için ince HTTP istemcisi (Streamlit'ten bağımsız, test edilebilir)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

BACKEND_HISTORY_LIMIT = 20  # /chat history alanının kabul ettiği en fazla mesaj
BACKEND_MESSAGE_LIMIT = 4000  # tek bir geçmiş mesajının en fazla karakter sayısı


class ApiError(Exception):
    """Kullanıcıya gösterilebilir, Türkçe hata mesajı taşır."""


def _extract_detail(resp: httpx.Response) -> str:
    """FastAPI hata gövdesinden ({"detail": ...}) okunabilir mesaj çıkarır."""
    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = None
    if isinstance(detail, list):  # 422 doğrulama hataları
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail or resp.text[:300] or f"HTTP {resp.status_code}")


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        upload_timeout: float = 120.0,
        chat_timeout: float = 200.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._upload_timeout = upload_timeout
        self._chat_timeout = chat_timeout
        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    def _request(
        self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any
    ) -> httpx.Response:
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"API'ye bağlanılamadı ({self.base_url}). Sunucu çalışıyor mu?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError("API zamanında yanıt vermedi (zaman aşımı).") from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"İstek başarısız oldu: {exc}") from exc

        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp))
        return resp

    # --- sistem ---
    def health(self) -> bool:
        try:
            self._request("GET", "/health", timeout=5.0)
            return True
        except ApiError:
            return False

    # --- belgeler ---
    def list_documents(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._request("GET", "/documents", params={"limit": limit}).json()["items"]

    def upload_document(self, filename: str, data: bytes) -> dict[str, Any]:
        resp = self._request(
            "POST",
            "/upload",
            files={"file": (filename, data, "application/pdf")},
            timeout=self._upload_timeout,
        )
        return resp.json()

    def delete_document(self, document_id: int) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    # --- sohbet ---
    def chat(
        self,
        question: str,
        *,
        document_id: int | None = None,
        history: Sequence[dict[str, Any]] = (),
        top_k: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "history": [
                {"role": m["role"], "content": m["content"][:BACKEND_MESSAGE_LIMIT]}
                for m in list(history)[-BACKEND_HISTORY_LIMIT:]
            ],
        }
        if document_id is not None:
            payload["document_id"] = document_id
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request(
            "POST", "/chat", json=payload, timeout=self._chat_timeout
        ).json()
