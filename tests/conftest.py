"""Test ortamı: ağır/dış bağımlılıklar olmadan saf mantığı test edebilmek için."""

import os
import sys
import tempfile
import types

# app.config import edilirken uploads/ klasörü oluşturulur; testler proje dizinini kirletmesin.
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp(prefix="rag-test-uploads-"))
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")

try:  # python-dotenv kurulu değilse (ör. minimal CI) boş bir yedek kullan
    import dotenv  # noqa: F401
except ImportError:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: False)
