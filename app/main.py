"""FastAPI uygulaması.

Çalıştırma (proje kökünden):
    uvicorn app.main:app --reload
Swagger arayüzü: http://localhost:8000/docs
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import close_db, init_db
from app.routers import chat, documents
from app.services.document_service import recover_stale_documents
from app.services.embedding_service import get_embedding_model

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

    yield
    await close_db()


app = FastAPI(
    title="RAG Belge Asistanı",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
