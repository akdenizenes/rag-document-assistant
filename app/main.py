"""FastAPI uygulaması.

Çalıştırma (proje kökünden):
    uvicorn app.main:app --reload
Swagger arayüzü: http://localhost:8000/docs
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from app.routers import auth, chat, documents, sessions

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import close_db, get_db, init_db
from app.errors import install_error_handlers
from app.routers import chat, documents, sessions
from app.services.document_service import recover_stale_documents
from app.services.embedding_service import get_embedding_model
from app.services.rerank_service import warmup as warmup_reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tablolar/eklenti yoksa oluşturur; varsa dokunmaz (init_db.py ile aynı işlev).
    await init_db()
    await recover_stale_documents()

    # İlk yüklemede kullanıcı beklemesin diye modeli açılışta ısıt.
    # (İlk çalıştırmada model internetten indirilir, birkaç dakika sürebilir.)
    try:
        await asyncio.to_thread(get_embedding_model)
    except Exception:
        logger.warning(
            "Embedding modeli açılışta yüklenemedi; ilk istekte tekrar denenecek.",
            exc_info=True,
        )

    await warmup_reranker()  # yalnızca RERANK_MODE=cross_encoder iken modeli yükler

    yield
    await close_db()


app = FastAPI(
    title="RAG Belge Asistanı",
    version="0.5.0",
    lifespan=lifespan,
)

install_error_handlers(app)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(auth.router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", tags=["system"])
async def ready(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Veritabanı bağlantısını da doğrular. Erişilemezse 503 döner (bkz. app/errors.py)."""
    await db.execute(text("SELECT 1"))
    return {"status": "ready"}
