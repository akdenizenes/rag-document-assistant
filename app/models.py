"""ORM modelleri: User, Document, DocumentChunk (pgvector + Türkçe tsvector), ChatSession, ChatMessage."""

import enum
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

EMBEDDING_DIM = 384


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryStatus(str, enum.Enum):
    """documents.summary_status değerleri (özet üretimi belge işlemeden bağımsızdır)."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # SUMMARY_ENABLED=false


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(
        String(100), default="application/pdf"
    )
    file_size: Mapped[int] = mapped_column(Integer, default=0)  # bayt
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(20), default=DocumentStatus.PENDING.value, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # --- v2: özet, metaveri ve tam metin (scripts/migrate_v2.py ile eklenir) ---
    # Kısa belge özeti (LLM üretir). Özet isteklerinde doğrudan bağlama verilir.
    summary: Mapped[str | None] = mapped_column(Text)
    # Başlık, belge türü, ana konular, soru/bölüm sayısı, sayfa/kelime sayısı vb.
    # (`metadata` adı SQLAlchemy'de rezerve olduğu için `doc_metadata`.)
    doc_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    summary_status: Mapped[str] = mapped_column(
        String(20),
        default=SummaryStatus.PENDING.value,
        server_default=SummaryStatus.PENDING.value,
    )
    # Sayfa işaretli tam metin. Büyük olduğu için tembel yüklenir: listelerde okunmaz,
    # gerektiğinde açıkça `undefer(Document.full_text)` veya sütun seçimiyle alınır.
    full_text: Mapped[str | None] = mapped_column(Text, deferred=True)

    owner: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Türkçe tam metin araması için otomatik üretilen tsvector (hibrit aramanın anahtar kelime kolu).
    # Veritabanı üretir; INSERT/UPDATE'e dahil edilmez. Yalnızca GIN indeksi üzerinden sorgulanır.
    content_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('turkish', content)", persisted=True),
        deferred=True,
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __mapper_args__ = {"eager_defaults": False}  # üretilen sütun INSERT sonrası geri okunmasın
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
        # Kosinüs benzerliği için HNSW indeksi (hızlı yaklaşık en yakın komşu araması)
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Anahtar kelime (full-text) araması için GIN indeksi
        Index("ix_document_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"


DEFAULT_SESSION_TITLE = "Yeni sohbet"


class ChatSession(Base):
    """Bir sohbet oturumu: kullanıcıya ait, başlıklı mesaj listesi.

    Yeni tablolar `init_db()` içindeki `create_all` ile otomatik oluşturulur (migrasyon gerekmez).
    """

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default=DEFAULT_SESSION_TITLE)
    # Oturumun son kullanılan belge kapsamı (belge id listesi); None = tüm belgeler.
    document_ids: Mapped[list[int] | None] = mapped_column(JSONB(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,  # silmeyi veritabanı (ON DELETE CASCADE) yapar
        order_by="ChatMessage.id",
    )

    __table_args__ = (Index("ix_chat_sessions_user_updated", "user_id", "updated_at"),)

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} user={self.user_id} title={self.title!r}>"


class ChatMessage(Base):
    """Oturumdaki tek bir mesaj (kullanıcı sorusu ya da asistan cevabı).

    NOT: `app.schemas.chat.ChatMessage` (istek şeması) ile karışmasın; ikisi farklı modüllerdedir.
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    # Yalnızca asistan mesajlarında: cevabın dayandığı kaynaklar, model ve rota
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB(none_as_null=True))
    model: Mapped[str | None] = mapped_column(String(100))
    route: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} session={self.session_id} role={self.role}>"
