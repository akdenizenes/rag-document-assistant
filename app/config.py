"""Uygulama ayarları (.env üzerinden okunur)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Embedding ---
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
# Modelin varsayılanı 128 token; Türkçe için biraz daha geniş tutuyoruz.
EMBEDDING_MAX_SEQ_LENGTH: int = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "256"))

# --- Chunking (karakter bazlı) ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

if CHUNK_OVERLAP >= CHUNK_SIZE // 2:
    raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE'ın yarısından küçük olmalı.")

# --- Yükleme ---
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

_upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR: Path = _upload_dir if _upload_dir.is_absolute() else BASE_DIR / _upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Retrieval (3. aşama) ---
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
# Kosinüs benzerliği (1 - cosine distance) bu değerin altındaki parçalar elenir.
RETRIEVAL_MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.1"))
# HNSW aramasında bakılan aday sayısı (yüksek = daha isabetli, biraz daha yavaş).
HNSW_EF_SEARCH: int = int(os.getenv("HNSW_EF_SEARCH", "100"))
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))

# --- LLM ---
# ollama   : ücretsiz, yerel, açık kaynak (varsayılan)
# openai   : OpenAI ve OpenAI uyumlu tüm API'ler (Groq, LM Studio, vLLM, ...)
# anthropic: Anthropic Claude API
# gemini   : Google Gemini (resmi google-genai SDK; LLM_BASE_URL kullanılmaz)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

_LLM_DEFAULTS: dict[str, tuple[str, str]] = {
    # provider: (varsayılan base url, varsayılan model)
    "ollama": ("http://localhost:11434", "qwen2.5:7b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "anthropic": ("https://api.anthropic.com", "claude-sonnet-5-5"),
    "gemini": ("https://generativelanguage.googleapis.com", "gemini-3.5-flash-lite"),
}
if LLM_PROVIDER not in _LLM_DEFAULTS:
    raise ValueError(
        f"Geçersiz LLM_PROVIDER={LLM_PROVIDER!r}. Seçenekler: {', '.join(_LLM_DEFAULTS)}"
    )

LLM_BASE_URL: str = (os.getenv("LLM_BASE_URL") or _LLM_DEFAULTS[LLM_PROVIDER][0]).rstrip("/")
LLM_MODEL: str = os.getenv("LLM_MODEL") or _LLM_DEFAULTS[LLM_PROVIDER][1]
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "4096"))  # yalnızca Ollama
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
