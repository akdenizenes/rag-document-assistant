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

# --- Hibrit arama (v2): pgvector + PostgreSQL full-text search, RRF ile birleştirilir ---
HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
# Her koldan (vektör / anahtar kelime) en fazla kaç aday alınır (en az top_k * 3 kullanılır).
HYBRID_CANDIDATES: int = int(os.getenv("HYBRID_CANDIDATES", "20"))
# Reciprocal Rank Fusion sabiti: skor = ağırlık / (k + sıra). 60 literatürdeki standart değerdir.
HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))
HYBRID_VECTOR_WEIGHT: float = float(os.getenv("HYBRID_VECTOR_WEIGHT", "1.0"))
HYBRID_FTS_WEIGHT: float = float(os.getenv("HYBRID_FTS_WEIGHT", "1.0"))
# Yalnızca anahtar kelime aramasından gelen parçalar için daha gevşek kosinüs eşiği
# (tam eşleşen terim/kod, anlamsal olarak zayıf görünse bile elenmesin).
HYBRID_FTS_MIN_SCORE: float = float(os.getenv("HYBRID_FTS_MIN_SCORE", "0.05"))

# --- Belge özeti / tam metin / yönlendirme (v2) ---
# false ise yüklemede LLM ile özet üretilmez (summary_status = 'skipped'); tam metin yine saklanır.
SUMMARY_ENABLED: bool = os.getenv("SUMMARY_ENABLED", "true").lower() == "true"
# documents.full_text için üst sınır (karakter). 80 sayfa ≈ 150-250 bin karakter.
FULL_TEXT_MAX_CHARS: int = int(os.getenv("FULL_TEXT_MAX_CHARS", "600000"))
# LLM'e tek seferde verilecek en fazla tam metin (karakter, ≈ 3-3.5 karakter/token).
# Ollama gibi küçük bağlamlı sağlayıcılarda bunu düşür.
FULL_TEXT_CONTEXT_MAX_CHARS: int = int(os.getenv("FULL_TEXT_CONTEXT_MAX_CHARS", "400000"))
# "Tüm belgelerde" özet sorusunda en fazla kaç belgenin özeti bağlama eklenir.
ROUTE_SUMMARY_MAX_DOCS: int = int(os.getenv("ROUTE_SUMMARY_MAX_DOCS", "10"))

# --- Sohbet geçmişi (adım 4.1) ---
# session_id ile gelen sorularda prompt için veritabanından okunacak en fazla mesaj
# (prompta girecek son N mesajı ayrıca MAX_HISTORY_MESSAGES belirler).
CHAT_HISTORY_DB_LIMIT: int = int(os.getenv("CHAT_HISTORY_DB_LIMIT", "20"))
SESSION_TITLE_MAX_CHARS: int = int(os.getenv("SESSION_TITLE_MAX_CHARS", "60"))
SESSION_MESSAGES_MAX: int = int(os.getenv("SESSION_MESSAGES_MAX", "500"))  # bir oturumda geri yüklenecek en fazla mesaj

# --- Çoklu belge kapsamı (adım 4.2) ---
MAX_SCOPE_DOCUMENTS: int = int(os.getenv("MAX_SCOPE_DOCUMENTS", "20"))  # bir sorguda seçilebilecek en fazla belge
# Sayma/genel tarama (tam metin) rotasında aynı anda bağlama girebilecek en fazla belge.
FULL_TEXT_MAX_DOCS: int = int(os.getenv("FULL_TEXT_MAX_DOCS", "3"))

# --- Yeniden sıralama / sonuç daraltma (adım 4.3) ---
# off          : arama sonucu olduğu gibi (RRF sırası) kullanılır
# score        : modelsiz akıllı filtre; en iyi skora göre zayıf parçaları eler (varsayılan)
# cross_encoder: sentence-transformers CrossEncoder ile yeniden sıralar (ilk kullanımda model iner)
RERANK_MODE: str = os.getenv("RERANK_MODE", "score").strip().lower()
if RERANK_MODE not in {"off", "score", "cross_encoder"}:
    raise ValueError(f"Geçersiz RERANK_MODE={RERANK_MODE!r}. Seçenekler: off, score, cross_encoder")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "15"))  # cross_encoder: yeniden sıralanacak aday sayısı
RERANK_BATCH_SIZE: int = int(os.getenv("RERANK_BATCH_SIZE", "16"))
RERANK_MAX_LENGTH: int = int(os.getenv("RERANK_MAX_LENGTH", "512"))  # (sorgu + parça) token sınırı
# cross_encoder: bu ham skorun altındaki parçalar elenir (model skor ölçeğine bağlıdır; boş = kapalı)
RERANK_MIN_SCORE: float | None = (
    float(os.environ["RERANK_MIN_SCORE"]) if os.getenv("RERANK_MIN_SCORE") else None
)
# score modu: en iyi kosinüs skorunun bu oranının altındaki parçalar elenir ("hem" eşleşenler hariç)
RERANK_RELATIVE_CUTOFF: float = float(os.getenv("RERANK_RELATIVE_CUTOFF", "0.5"))
RERANK_MIN_KEEP: int = int(os.getenv("RERANK_MIN_KEEP", "2"))  # filtre en az bu kadar parça bırakır

# --- İstek sınırlama (adım 4.4) ---
# Biçim: "<sayı>/<second|minute|hour|day>"; kullanıcı başına, süreç içi (bellek) sayaç.
RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "20/minute")
RATE_LIMIT_UPLOAD: str = os.getenv("RATE_LIMIT_UPLOAD", "10/minute")
RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "120/minute")

# --- LLM ---
# ollama    : ücretsiz, yerel, açık kaynak (varsayılan)
# openai    : OpenAI ve OpenAI uyumlu tüm API'ler (Groq, LM Studio, vLLM, ...)
# anthropic : Anthropic Claude API
# gemini    : Google Gemini (resmi google-genai SDK; LLM_BASE_URL kullanılmaz)
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

# --- Kimlik Doğrulama / JWT (Adım 5) ---
JWT_SECRET_KEY: str = os.getenv(
    "JWT_SECRET_KEY",
    "super-secret-rag-jwt-token-key-change-in-production-2026",
)
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 7))  # Varsayılan: 7 gün
)