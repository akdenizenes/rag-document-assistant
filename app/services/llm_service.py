"""LLM istemcisi (Normal + SSE Streaming + Belge Özet/Metaveri Çıkarımı).

Desteklenen sağlayıcılar (LLM_PROVIDER):
  - ollama   : yerel, ücretsiz           (httpx, POST /api/chat)
  - openai   : OpenAI ve uyumlu API'ler  (httpx, POST /chat/completions)
  - anthropic: Claude API                (httpx, POST /v1/messages)
  - gemini   : Google Gemini             (resmi `google-genai` SDK)

Sağlayıcı, model ve anahtar .env üzerinden seçilir (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY).
`google-genai` yalnızca LLM_PROVIDER=gemini iken import edilir.
"""

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import BaseModel

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
from app.services.prompts import DOCUMENT_METADATA_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

Messages = list[dict[str, str]]


class LLMError(Exception):
    """LLM çağrısı başarısız olduğunda fırlatılır; `status_code` HTTP yanıtına yansıtılır."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class _GeminiSummarySchema(BaseModel):
    summary: str
    title: str
    document_type: str
    main_topics: list[str]
    question_count: int
    section_count: int


# --------------------------------------------------------------------------- #
# httpx tabanlı sağlayıcılar (Senkron Tek Yanıt & Streaming)
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


async def _stream_ollama(messages: Messages) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=10.0)
    ) as client:
        async with client.stream(
            "POST",
            f"{LLM_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                    "num_ctx": LLM_NUM_CTX,
                },
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token


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


async def _stream_openai(messages: Messages) -> AsyncGenerator[str, None]:
    if not LLM_API_KEY and "api.openai.com" in LLM_BASE_URL:
        raise LLMError("LLM_API_KEY tanımlı değil (.env dosyasını kontrol et).", 503)
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=10.0)
    ) as client:
        async with client.stream(
            "POST",
            f"{LLM_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta


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


async def _stream_anthropic(messages: Messages) -> AsyncGenerator[str, None]:
    text = await _dispatch(messages)
    yield text


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
            http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),
        )
    return _gemini_client


def _to_gemini_contents(messages: Messages, types) -> tuple[str | None, list]:
    """OpenAI tarzı mesajları Gemini biçimine çevirir."""
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


def _handle_gemini_error(exc: Exception) -> None:
    from google.genai import errors as genai_errors

    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", None) or exc)
        logger.error("Gemini API hatası (%s): %s", code, message)

        if code in (401, 403) or (code == 400 and "api key" in message.lower()):
            raise LLMError(
                f"Gemini kimlik doğrulama hatası: {message} "
                "(LLM_API_KEY değerini ve google-genai sürümünü kontrol et: pip install -U google-genai)",
                503,
            ) from exc
        if code == 429:
            raise LLMError(
                "Gemini kota/istek sınırı aşıldı. Biraz bekleyip tekrar dene.", 429
            ) from exc
        raise LLMError(f"Gemini API Hatası ({code}): {message}", 502) from exc

    logger.exception("Beklenmeyen Gemini hatası")
    raise LLMError(f"Beklenmeyen Hata: {type(exc).__name__} - {str(exc)}", 502) from exc


async def _call_gemini(messages: Messages) -> str:
    client = _get_gemini_client()
    from google.genai import types

    system, contents = _to_gemini_contents(messages, types)

    try:
        response = await client.aio.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )
    except Exception as exc:
        _handle_gemini_error(exc)

    text = response.text
    if not text:
        finish = response.candidates[0].finish_reason if response.candidates else None
        feedback = getattr(response, "prompt_feedback", None)
        blocked = getattr(feedback, "block_reason", None) if feedback else None
        raise LLMError(
            f"Gemini metin döndürmedi (finish_reason={finish}, block_reason={blocked}). "
            "Güvenlik filtresi ya da LLM_MAX_TOKENS çok düşük olabilir."
        )
    return text


async def _stream_gemini(messages: Messages) -> AsyncGenerator[str, None]:
    client = _get_gemini_client()
    from google.genai import types

    system, contents = _to_gemini_contents(messages, types)

    try:
        stream = await client.aio.models.generate_content_stream(
            model=LLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        _handle_gemini_error(exc)


# --------------------------------------------------------------------------- #
# Ortak giriş noktaları: generate, generate_stream, generate_document_summary
# --------------------------------------------------------------------------- #
async def _dispatch(messages: Messages) -> str:
    if LLM_PROVIDER == "gemini":
        return await _call_gemini(messages)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=10.0)
    ) as client:
        return await _HTTP_PROVIDERS[LLM_PROVIDER](client, messages)


async def generate(messages: Messages) -> str:
    """Mesaj listesini seçili LLM'e gönderir ve yanıt metnini döner."""
    try:
        text = await _dispatch(messages)
    except LLMError:
        raise
    except httpx.ConnectError as exc:
        hint = (
            " Ollama çalışıyor mu? (ollama serve)" if LLM_PROVIDER == "ollama" else ""
        )
        raise LLMError(f"LLM servisine bağlanılamadı ({LLM_BASE_URL}).{hint}", 503) from exc
    except httpx.TimeoutException as exc:
        raise LLMError("LLM zamanında yanıt vermedi (zaman aşımı).", 504) from exc
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:300]
        logger.error("LLM HTTP %s: %s", exc.response.status_code, body)
        raise LLMError(f"LLM hata döndürdü ({exc.response.status_code}): {body}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM isteği başarısız oldu: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError("LLM yanıtı beklenmeyen bir biçimde geldi.") from exc

    text = (text or "").strip()
    if not text:
        raise LLMError("LLM boş yanıt döndürdü.")
    return text


async def generate_stream(messages: Messages) -> AsyncGenerator[str, None]:
    """Mesaj listesini seçili LLM'e gönderir ve yanıtı parça parça (stream) üretir."""
    try:
        if LLM_PROVIDER == "gemini":
            async for token in _stream_gemini(messages):
                yield token
        elif LLM_PROVIDER == "ollama":
            async for token in _stream_ollama(messages):
                yield token
        elif LLM_PROVIDER == "openai":
            async for token in _stream_openai(messages):
                yield token
        else:
            async for token in _stream_anthropic(messages):
                yield token
    except LLMError:
        raise
    except Exception as exc:
        logger.exception("Streaming sırasında LLM hatası")
        raise LLMError(f"Akış hatası: {exc}", 502) from exc


async def generate_document_summary(
    full_text: str,
    filename: str = "",
    page_count: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Belgenin tam metninden Türkçe kısa özet ve temel metaveriler üretir (Adım 1)."""
    prompt = DOCUMENT_METADATA_EXTRACTION_PROMPT.format(
        filename=filename,
        page_count=page_count,
        full_text=full_text,
    )

    if LLM_PROVIDER == "gemini":
        client = _get_gemini_client()
        from google.genai import types

        try:
            response = await client.aio.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=_GeminiSummarySchema,
                ),
            )
            data = json.loads(response.text or "{}")
            summary = str(data.pop("summary", "")).strip()
            return summary, data
        except Exception as exc:
            _handle_gemini_error(exc)

    # Diğer sağlayıcılar için JSON parse fallback
    raw = await generate([{"role": "user", "content": prompt}])
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    summary = str(data.pop("summary", "")).strip()
    return summary, data
