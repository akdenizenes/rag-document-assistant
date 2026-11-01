"""LLM istemcisi.

Desteklenen sağlayıcılar (LLM_PROVIDER):
  - ollama   : yerel, ücretsiz           (httpx, POST /api/chat)
  - openai   : OpenAI ve uyumlu API'ler  (httpx, POST /chat/completions)
  - anthropic: Claude API                (httpx, POST /v1/messages)
  - gemini   : Google Gemini             (resmi `google-genai` SDK)

İki giriş noktası vardır:
  - generate(messages)        -> cevabın tamamını tek seferde döner
  - generate_stream(messages) -> cevabı parça parça (async generator) üretir

Gerçek akış (streaming) şu an yalnızca Gemini için vardır. Diğer sağlayıcılarda
generate_stream cevabı tek parça olarak verir; arayüz yine de sorunsuz çalışır.

`google-genai` yalnızca LLM_PROVIDER=gemini iken import edilir.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

import httpx

from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

Messages = list[dict[str, str]]


class LLMError(Exception):
    """LLM çağrısı başarısız olduğunda fırlatılır; `status_code` HTTP yanıtına yansıtılır."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# httpx tabanlı sağlayıcılar
# --------------------------------------------------------------------------- #
async def _call_ollama(client: httpx.AsyncClient, messages: Messages) -> str:
    resp = await client.post(
        f"{LLM_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
                "num_ctx": LLM_NUM_CTX,
            },
        },
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


async def _call_openai(client: httpx.AsyncClient, messages: Messages) -> str:
    if not LLM_API_KEY and "api.openai.com" in LLM_BASE_URL:
        raise LLMError("LLM_API_KEY tanımlı değil (.env dosyasını kontrol et).", 503)
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    resp = await client.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_anthropic(client: httpx.AsyncClient, messages: Messages) -> str:
    if not LLM_API_KEY:
        raise LLMError("LLM_API_KEY tanımlı değil (.env dosyasını kontrol et).", 503)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    conversation = [m for m in messages if m["role"] != "system"]
    resp = await client.post(
        f"{LLM_BASE_URL}/v1/messages",
        headers={"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01"},
        json={
            "model": LLM_MODEL,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "system": system,
            "messages": conversation,
        },
    )
    resp.raise_for_status()
    blocks = resp.json()["content"]
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


_HTTP_PROVIDERS = {
    "ollama": _call_ollama,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


# --------------------------------------------------------------------------- #
# Gemini (resmi google-genai SDK)
# --------------------------------------------------------------------------- #
_gemini_client = None


def _get_gemini_client():
    """İstemciyi ilk çağrıda oluşturur (lazy singleton)."""
    global _gemini_client
    if _gemini_client is None:
        if not LLM_API_KEY:
            raise LLMError("LLM_API_KEY tanımlı değil (.env dosyasını kontrol et).", 503)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "google-genai paketi kurulu değil. Çalıştır: pip install -U google-genai", 503
            ) from exc

        _gemini_client = genai.Client(
            api_key=LLM_API_KEY,
            # SDK zaman aşımını milisaniye cinsinden bekler.
            http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),
        )
    return _gemini_client


def _to_gemini_contents(messages: Messages, types) -> tuple[str | None, list]:
    """OpenAI tarzı mesajları Gemini biçimine çevirir.

    - 'system' mesajları -> system_instruction
    - 'assistant' rolü   -> 'model'
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
        if m["role"] != "system"
    ]
    return system, contents


def _gemini_request(messages: Messages) -> tuple:
    """generate_content ve generate_content_stream için ortak (client, errors, model, contents, config)."""
    client = _get_gemini_client()
    from google.genai import errors as genai_errors
    from google.genai import types

    system, contents = _to_gemini_contents(messages, types)
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
    )
    return client, genai_errors, contents, config


def _map_gemini_error(exc) -> LLMError:
    """google-genai APIError'unu kullanıcıya gösterilebilir bir LLMError'a çevirir."""
    code = getattr(exc, "code", None)
    message = str(getattr(exc, "message", None) or exc)[:300]
    logger.error("Gemini API hatası (%s): %s", code, message)

    if code in (401, 403) or (code == 400 and "api key" in message.lower()):
        return LLMError(
            f"Gemini kimlik doğrulama hatası: {message} "
            "(LLM_API_KEY değerini ve google-genai sürümünü kontrol et: pip install -U google-genai)",
            503,
        )
    if code == 429:
        return LLMError("Gemini kota/istek sınırı aşıldı. Biraz bekleyip tekrar dene.", 429)
    if code == 404:
        return LLMError(
            f"Gemini modeli bulunamadı: {LLM_MODEL!r}. LLM_MODEL değerini kontrol et.", 502
        )
    return LLMError(f"Gemini hata döndürdü ({code}): {message}")


def _empty_gemini_message(finish_reason, block_reason) -> str:
    return (
        f"Gemini metin döndürmedi (finish_reason={finish_reason}, block_reason={block_reason}). "
        "Güvenlik filtresi ya da LLM_MAX_TOKENS çok düşük olabilir."
    )


async def _call_gemini(messages: Messages) -> str:
    client, genai_errors, contents, config = _gemini_request(messages)

    try:
        response = await client.aio.models.generate_content(
            model=LLM_MODEL, contents=contents, config=config
        )
    except genai_errors.APIError as exc:
        raise _map_gemini_error(exc) from exc

    text = response.text
    if not text:
        finish = response.candidates[0].finish_reason if response.candidates else None
        feedback = getattr(response, "prompt_feedback", None)
        blocked = getattr(feedback, "block_reason", None) if feedback else None
        raise LLMError(_empty_gemini_message(finish, blocked))
    return text


async def _stream_gemini(messages: Messages) -> AsyncIterator[str]:
    """Gemini cevabını `generate_content_stream` ile parça parça üretir."""
    client, genai_errors, contents, config = _gemini_request(messages)

    finish_reason = None
    block_reason = None
    got_text = False
    try:
        stream = await client.aio.models.generate_content_stream(
            model=LLM_MODEL, contents=contents, config=config
        )
        async for chunk in stream:
            feedback = getattr(chunk, "prompt_feedback", None)
            if feedback is not None and getattr(feedback, "block_reason", None):
                block_reason = feedback.block_reason
            if chunk.candidates:
                finish_reason = chunk.candidates[0].finish_reason or finish_reason

            piece = chunk.text
            if piece:
                got_text = True
                yield piece
    except genai_errors.APIError as exc:
        raise _map_gemini_error(exc) from exc

    if not got_text:
        raise LLMError(_empty_gemini_message(finish_reason, block_reason))


# --------------------------------------------------------------------------- #
# Ortak giriş noktaları
# --------------------------------------------------------------------------- #
async def _dispatch(messages: Messages) -> str:
    if LLM_PROVIDER == "gemini":
        return await _call_gemini(messages)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=10.0)
    ) as client:
        return await _HTTP_PROVIDERS[LLM_PROVIDER](client, messages)


def _wrap_transport_error(exc: Exception) -> LLMError | None:
    """httpx / ayrıştırma hatalarını LLMError'a çevirir; tanınmayan hata için None döner."""
    if isinstance(exc, httpx.ConnectError):
        hint = " Ollama çalışıyor mu? (ollama serve)" if LLM_PROVIDER == "ollama" else ""
        return LLMError(f"LLM servisine bağlanılamadı ({LLM_BASE_URL}).{hint}", 503)
    if isinstance(exc, httpx.TimeoutException):
        return LLMError("LLM zamanında yanıt vermedi (zaman aşımı).", 504)
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:300]
        logger.error("LLM HTTP %s: %s", exc.response.status_code, body)
        return LLMError(f"LLM hata döndürdü ({exc.response.status_code}): {body}")
    if isinstance(exc, httpx.HTTPError):
        return LLMError(f"LLM isteği başarısız oldu: {exc}")
    if isinstance(exc, (KeyError, IndexError, TypeError, ValueError)):
        return LLMError("LLM yanıtı beklenmeyen bir biçimde geldi.")
    return None


async def generate(messages: Messages) -> str:
    """Mesaj listesini seçili LLM'e gönderir ve yanıt metnini döner."""
    try:
        text = await _dispatch(messages)
    except LLMError:
        raise
    except Exception as exc:
        wrapped = _wrap_transport_error(exc)
        if wrapped is None:
            raise
        raise wrapped from exc

    text = (text or "").strip()
    if not text:
        raise LLMError("LLM boş yanıt döndürdü.")
    return text


async def generate_stream(messages: Messages) -> AsyncIterator[str]:
    """Cevabı parça parça üretir (async generator).

    Hatalar LLMError olarak fırlatılır; akış başladıktan sonra da fırlatılabilir
    (çağıran taraf bunu istemciye 'error' olayı olarak iletmelidir).
    """
    if LLM_PROVIDER != "gemini":
        # Bu sağlayıcılar için gerçek akış henüz yok: cevabı tek parça olarak ver.
        yield await generate(messages)
        return

    try:
        async with aclosing(_stream_gemini(messages)) as pieces:
            async for piece in pieces:
                yield piece
    except LLMError:
        raise
    except Exception as exc:
        wrapped = _wrap_transport_error(exc)
        if wrapped is None:
            raise
        raise wrapped from exc
