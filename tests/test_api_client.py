"""Streamlit'in kullandığı API istemcisi testleri (ağ yok: httpx.MockTransport)."""

import json
import sys
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frontend"))
from api_client import ApiClient, ApiError, _extract_detail, iter_sse  # noqa: E402


def make_client(handler) -> ApiClient:
    return ApiClient("http://api.test", transport=httpx.MockTransport(handler))


def sse(*events: tuple[str, dict]) -> bytes:
    return "".join(
        f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n" for name, data in events
    ).encode()


def test_extract_detail_variants():
    assert _extract_detail(httpx.Response(404, json={"detail": "Belge bulunamadı."})) == "Belge bulunamadı."
    html = httpx.Response(502, text="<html><body>Bad Gateway</body></html>")
    assert "502" in _extract_detail(html) and "<html>" not in _extract_detail(html)
    assert "Çok fazla istek" in _extract_detail(httpx.Response(429, text=""))
    assert _extract_detail(httpx.Response(422, json={"detail": [{"msg": "alan eksik"}]})) == "alan eksik"
    assert "500" in _extract_detail(httpx.Response(500, text="boom"))


def test_rate_limit_error_is_readable():
    def handler(request):
        return httpx.Response(429, json={"detail": "Çok fazla istek. 12 saniye sonra tekrar dene."})

    with pytest.raises(ApiError, match="12 saniye"):
        make_client(handler).chat("x")


def test_get_retries_on_connect_error_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("bağlantı reddedildi")
        return httpx.Response(200, json={"items": [{"id": 1}]})

    assert make_client(handler).list_documents() == [{"id": 1}]
    assert calls["n"] == 2


def test_post_is_never_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("kapalı")

    with pytest.raises(ApiError, match="bağlanılamadı"):
        make_client(handler).chat("x")
    assert calls["n"] == 1


def test_invalid_json_on_success_becomes_api_error():
    client = make_client(lambda request: httpx.Response(200, text="<html>proxy</html>"))
    with pytest.raises(ApiError, match="geçersiz JSON"):
        client.list_documents()


def test_chat_payload_with_session_omits_history_and_sends_scope():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "a", "sources": [], "model": "m"})

    make_client(handler).chat(
        "soru", document_ids=[2, 5], history=[{"role": "user", "content": "eski"}], session_id=9, top_k=3
    )
    assert seen == {"question": "soru", "session_id": 9, "document_ids": [2, 5], "top_k": 3}


def test_chat_payload_without_session_sends_history():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={})

    make_client(handler).chat("soru", history=[{"role": "user", "content": "x" * 5000, "sources": []}])
    assert seen["history"] == [{"role": "user", "content": "x" * 4000}]
    assert "document_ids" not in seen and "session_id" not in seen


def test_stream_reads_route_session_and_tokens():
    body = sse(
        ("sources", {"sources": [{"document_name": "a.pdf"}], "model": "m", "route": "hybrid", "session_id": 4}),
        ("token", {"text": "Mer"}),
        ("token", {"text": "haba"}),
        ("done", {}),
    )
    client = make_client(lambda request: httpx.Response(200, content=body))
    stream = client.chat_stream("selam", session_id=4)
    assert "".join(stream.tokens()) == "Merhaba"
    assert stream.finished and stream.error is None
    assert (stream.route, stream.session_id, stream.model) == ("hybrid", 4, "m")


def test_stream_error_event_and_truncated_stream():
    body = sse(("sources", {"sources": []}), ("token", {"text": "yarım"}), ("error", {"detail": "kota doldu"}))
    stream = make_client(lambda request: httpx.Response(200, content=body)).chat_stream("x")
    assert "".join(stream.tokens()) == "yarım" and stream.error == "kota doldu"

    body = sse(("sources", {"sources": []}), ("token", {"text": "yarım"}))  # 'done' yok
    stream = make_client(lambda request: httpx.Response(200, content=body)).chat_stream("x")
    list(stream.tokens())
    assert "yarıda kesildi" in stream.error


def test_stream_http_error_before_streaming():
    client = make_client(lambda request: httpx.Response(409, json={"detail": "Belge henüz sorgulanamaz"}))
    with pytest.raises(ApiError, match="henüz sorgulanamaz"):
        client.chat_stream("x")


def test_session_endpoints():
    log = []

    def handler(request):
        log.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/sessions":
            return httpx.Response(200, json={"total": 1, "items": [{"id": 3, "title": "t"}]})
        if request.method == "POST":
            return httpx.Response(201, json={"id": 8, "title": "Yeni sohbet"})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"id": 3, "messages": []})

    client = make_client(handler)
    assert client.list_sessions() == [{"id": 3, "title": "t"}]
    assert client.create_session(document_ids=[1, 2])["id"] == 8
    assert client.get_session(3)["messages"] == []
    assert client.rename_session(3, "yeni")["id"] == 3
    client.delete_session(3)
    assert log == [
        ("GET", "/sessions"), ("POST", "/sessions"), ("GET", "/sessions/3"),
        ("PATCH", "/sessions/3"), ("DELETE", "/sessions/3"),
    ]


def test_iter_sse_ignores_comments():
    lines = [": keep-alive", "event: token", 'data: {"text": "a"}', "", "data: bozuk", ""]
    assert list(iter_sse(lines)) == [("token", {"text": "a"}), ("message", {})]
