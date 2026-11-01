"""FastAPI backend'i için ince HTTP istemcisi (Streamlit'ten bağımsız, test edilebilir)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import httpx

BACKEND_HISTORY_LIMIT = 20  # /chat history alanının kabul ettiği en fazla mesaj
BACKEND_MESSAGE_LIMIT = 4000  # tek bir geçmiş mesajının en fazla karakter sayısı


class ApiError(Exception):
    """Kullanıcıya gösterilebilir, Türkçe hata mesajı taşır."""


_GENERIC_STATUS_MESSAGES = {
    401: "Oturum açmanız gerekiyor veya oturum süreniz doldu.",
    403: "Bu işlem için yetkiniz yok.",
    429: "Çok fazla istek gönderildi. Lütfen biraz bekleyip tekrar dene.",
    502: "Sunucu şu anda yanıt vermiyor (502). Biraz sonra tekrar dene.",
    503: "Servis şu anda kullanılamıyor (503). Biraz sonra tekrar dene.",
    504: "Sunucu zamanında yanıt vermedi (504). Biraz sonra tekrar dene.",
}


def _extract_detail(resp: httpx.Response) -> str:
    """FastAPI hata gövdesinden ({"detail": ...}) okunabilir mesaj çıkarır.

    Gövde JSON değilse (örn. bir ters vekilin HTML hata sayfası) ham metin yerine durum koduna
    uygun kısa bir Türkçe mesaj döner.
    """
    try:
        body = resp.json()
        detail = body.get("detail") if isinstance(body, dict) else None
    except Exception:
        detail = None
    if isinstance(detail, list):  # 422 doğrulama hataları
        detail = "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        )
    if detail:
        return str(detail)
    if resp.status_code in _GENERIC_STATUS_MESSAGES:
        return _GENERIC_STATUS_MESSAGES[resp.status_code]
    if resp.status_code >= 500:
        return f"Sunucu hatası (HTTP {resp.status_code})."
    return f"İstek reddedildi (HTTP {resp.status_code})."


def _json(resp: httpx.Response) -> Any:
    """Başarılı yanıtın gövdesini JSON olarak okur; bozuksa ApiError fırlatır."""
    try:
        return resp.json()
    except ValueError as exc:
        raise ApiError("Sunucudan beklenmeyen bir yanıt geldi (geçersiz JSON).") from exc


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Satır satır gelen Server-Sent Events akışını (olay_adı, json_verisi) çiftlerine çevirir.

    - Boş satır bir olayı bitirir.
    - ':' ile başlayan satırlar yorumdur (keep-alive), yok sayılır.
    - JSON olmayan 'data' alanı boş sözlük olarak döner.
    """
    event = "message"
    data_lines: list[str] = []

    def flush() -> tuple[str, dict[str, Any]] | None:
        if not data_lines:
            return None
        try:
            payload = json.loads("\n".join(data_lines))
        except ValueError:
            payload = {}
        return event, payload if isinstance(payload, dict) else {}

    for line in lines:
        if line == "":
            item = flush()
            if item is not None:
                yield item
            event, data_lines = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            value = line[len("data:") :]
            data_lines.append(value[1:] if value.startswith(" ") else value)

    item = flush()  # son olay boş satırla bitmediyse
    if item is not None:
        yield item


class ChatStream:
    """/chat/stream yanıtını okur.

    Kullanım:
        stream = api.chat_stream("soru?")
        for text in stream.tokens():     # veya st.write_stream(stream.tokens())
            ...
        stream.sources, stream.model     # akış başlar başlamaz dolar (ilk token'dan önce)
        stream.route                     # "summary" | "full_text" | "hybrid" (sources olayıyla gelir)
        stream.session_id                # soru-cevabın kaydedildiği oturum (varsa)
        stream.error                     # akış hatayla bittiyse mesaj, aksi halde None
        stream.finished                  # 'done' olayı alındıysa True
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.sources: list[dict[str, Any]] = []
        self.model: str | None = None
        self.route: str | None = None
        self.session_id: int | None = None
        self.error: str | None = None
        self.finished: bool = False

    def tokens(self) -> Iterator[str]:
        """Cevap metni parçalarını üretir. Bağlantıyı iş bitince (veya hata/iptalde) kapatır."""
        try:
            for event, data in iter_sse(self._response.iter_lines()):
                if event == "sources":
                    self.sources = data.get("sources", [])
                    self.model = data.get("model")
                    self.route = data.get("route")
                    self.session_id = data.get("session_id")
                elif event == "token":
                    text = data.get("text", "")
                    if text:
                        yield text
                elif event == "error":
                    self.error = str(data.get("detail") or "Bilinmeyen akış hatası.")
                    break
                elif event == "done":
                    self.finished = True
                    break
        except httpx.TimeoutException:
            self.error = "Cevap akışı zaman aşımına uğradı."
        except httpx.HTTPError as exc:
            self.error = f"Bağlantı koptu: {exc}"
        except Exception as exc:  # beklenmeyen okuma/ayrıştırma hatası arayüzü çökertmesin
            self.error = f"Cevap akışı okunurken beklenmeyen bir hata oluştu: {exc}"
        finally:
            self._response.close()

        if not self.finished and self.error is None:
            self.error = "Cevap akışı beklenmedik şekilde yarıda kesildi."

    def close(self) -> None:
        self._response.close()


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        upload_timeout: float = 120.0,
        chat_timeout: float = 200.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._upload_timeout = upload_timeout
        self._chat_timeout = chat_timeout
        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    def set_token(self, token: str | None) -> None:
        """Kullanıcı giriş/çıkış yaptığında token'ı günceller."""
        self.token = token

    def _headers(self, custom_headers: dict[str, str] | None = None) -> dict[str, str]:
        """Authorization başlığını ve isteğe özel başlıkları birleştirir."""
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if custom_headers:
            headers.update(custom_headers)
        return headers

    def _connection_error(self, exc: Exception) -> ApiError:
        if isinstance(exc, httpx.ConnectError):
            return ApiError(f"API'ye bağlanılamadı ({self.base_url}). Sunucu çalışıyor mu?")
        if isinstance(exc, httpx.TimeoutException):
            return ApiError("API zamanında yanıt vermedi (zaman aşımı).")
        return ApiError(f"İstek başarısız oldu: {exc}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        retries: int = 0,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """İstek gönderir. `retries` yalnızca BAĞLANTI hatalarında (istek sunucuya ulaşmadıysa)
        tekrar dener; bu yüzden yalnızca idempotent (GET) çağrılarda kullanılmalıdır."""
        if timeout is not None:
            kwargs["timeout"] = timeout
        
        req_headers = self._headers(headers)
        attempt = 0
        while True:
            try:
                resp = self._client.request(method, path, headers=req_headers, **kwargs)
                break
            except httpx.ConnectError as exc:
                if attempt >= retries:
                    raise self._connection_error(exc) from exc
                attempt += 1
                time.sleep(0.4 * attempt)
            except httpx.HTTPError as exc:
                raise self._connection_error(exc) from exc

        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp))
        return resp

    # --- kimlik doğrulama (auth) ---
    def register(
        self, email: str, password: str, full_name: str | None = None
    ) -> dict[str, Any]:
        """Yeni kullanıcı kaydeder ve JWT token döner."""
        payload: dict[str, Any] = {"email": email, "password": password}
        if full_name:
            payload["full_name"] = full_name
        data = _json(self._request("POST", "/auth/register", json=payload))
        if "access_token" in data:
            self.set_token(data["access_token"])
        return data

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Giriş yapar ve JWT token döner."""
        payload = {"email": email, "password": password}
        data = _json(self._request("POST", "/auth/login", json=payload))
        if "access_token" in data:
            self.set_token(data["access_token"])
        return data

    # --- sistem ---
    def health(self) -> bool:
        try:
            self._request("GET", "/health", timeout=5.0)
            return True
        except ApiError:
            return False

    def ready(self) -> bool:
        """Veritabanı dahil tüm bağımlılıklar hazır mı? (GET /health/ready)

        Eski bir API sürümünde bu uç nokta yoktur (404); o durumda engel sayılmaz.
        """
        try:
            resp = self._client.get("/health/ready", timeout=5.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200 or resp.status_code == 404

    # --- belgeler ---
    def list_documents(self, limit: int = 200) -> list[dict[str, Any]]:
        data = _json(self._request("GET", "/documents", params={"limit": limit}, retries=2))
        return data["items"]

    def upload_document(self, filename: str, data: bytes) -> dict[str, Any]:
        resp = self._request(
            "POST",
            "/upload",
            files={"file": (filename, data, "application/pdf")},
            timeout=self._upload_timeout,
        )
        return _json(resp)

    def delete_document(self, document_id: int) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    # --- sohbet oturumları (geçmiş) ---
    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        data = _json(self._request("GET", "/sessions", params={"limit": limit}, retries=1))
        return data["items"]

    def create_session(
        self, *, title: str | None = None, document_ids: Sequence[int] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        if document_ids:
            payload["document_ids"] = list(document_ids)
        return _json(self._request("POST", "/sessions", json=payload))

    def get_session(self, session_id: int) -> dict[str, Any]:
        """Oturumu ve mesajlarını döner."""
        return _json(self._request("GET", f"/sessions/{session_id}", retries=1))

    def rename_session(self, session_id: int, title: str) -> dict[str, Any]:
        return _json(self._request("PATCH", f"/sessions/{session_id}", json={"title": title}))

    def delete_session(self, session_id: int) -> None:
        self._request("DELETE", f"/sessions/{session_id}")

    # --- sohbet ---
    def _chat_payload(
        self,
        question: str,
        document_ids: Sequence[int] | None,
        history: Sequence[dict[str, Any]],
        top_k: int | None,
        session_id: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question}
        if session_id is not None:
            # Geçmişi sunucu oturumdan okur; istemci geçmişi gönderilmez.
            payload["session_id"] = session_id
        else:
            payload["history"] = [
                {"role": m["role"], "content": m["content"][:BACKEND_MESSAGE_LIMIT]}
                for m in list(history)[-BACKEND_HISTORY_LIMIT:]
            ]
        if document_ids:
            payload["document_ids"] = list(document_ids)
        if top_k is not None:
            payload["top_k"] = top_k
        return payload

    def chat(
        self,
        question: str,
        *,
        document_ids: Sequence[int] | None = None,
        history: Sequence[dict[str, Any]] = (),
        top_k: int | None = None,
        session_id: int | None = None,
    ) -> dict[str, Any]:
        """Cevabı tek seferde döner (POST /chat). `document_ids` boşsa tüm belgelerde aranır."""
        payload = self._chat_payload(question, document_ids, history, top_k, session_id)
        return _json(self._request("POST", "/chat", json=payload, timeout=self._chat_timeout))

    def chat_stream(
        self,
        question: str,
        *,
        document_ids: Sequence[int] | None = None,
        history: Sequence[dict[str, Any]] = (),
        top_k: int | None = None,
        session_id: int | None = None,
    ) -> ChatStream:
        """Cevabı parça parça alır (POST /chat/stream).

        İstek gönderilir ve yanıt BAŞLIKLARI okunur; 404/409/422 gibi hatalar burada ApiError
        olarak fırlatılır. Gövde (cevap metni) `ChatStream.tokens()` ile tüketilir.
        """
        payload = self._chat_payload(question, document_ids, history, top_k, session_id)
        headers = self._headers({"Accept": "text/event-stream"})
        request = self._client.build_request(
            "POST",
            "/chat/stream",
            json=payload,
            headers=headers,
            # read: iki parça arasında beklenebilecek en uzun süre (ilk token dahil)
            timeout=httpx.Timeout(30.0, read=self._chat_timeout),
        )
        try:
            response = self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc

        if response.status_code >= 400:
            try:
                response.read()
                detail = _extract_detail(response)
            finally:
                response.close()
            raise ApiError(detail)
        return ChatStream(response)