"""Sohbet oturumu iş mantığı: oturum bulma/oluşturma, geçmiş okuma, soru-cevap kaydetme."""

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import SESSION_TITLE_MAX_CHARS
from app.database import AsyncSessionLocal
from app.models import DEFAULT_SESSION_TITLE, ChatMessage, ChatSession
from app.schemas.chat import ChatMessage as ChatMessageSchema

logger = logging.getLogger(__name__)

_PROMPT_MESSAGE_LIMIT = 4000  # schemas.chat.ChatMessage.content üst sınırı


def make_title(question: str) -> str:
    """İlk sorudan kısa bir oturum başlığı üretir."""
    title = " ".join(question.split())
    if len(title) > SESSION_TITLE_MAX_CHARS:
        title = title[: SESSION_TITLE_MAX_CHARS - 1].rstrip() + "…"
    return title or DEFAULT_SESSION_TITLE


async def get_owned_session(
    db: AsyncSession, user_id: int, session_id: int
) -> ChatSession | None:
    return await db.scalar(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
    )


async def create_session(
    db: AsyncSession,
    user_id: int,
    *,
    title: str | None = None,
    document_ids: list[int] | None = None,
) -> ChatSession:
    chat = ChatSession(
        user_id=user_id,
        title=title or DEFAULT_SESSION_TITLE,
        document_ids=document_ids or None,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


async def load_history(db: AsyncSession, session_id: int, limit: int) -> list[ChatMessageSchema]:
    """Oturumdaki son `limit` mesajı (eskiden yeniye) prompt şemasına çevirerek döner."""
    rows = (
        await db.execute(
            select(ChatMessage.role, ChatMessage.content)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
    ).all()
    rows.reverse()
    return [
        ChatMessageSchema(role=row.role, content=row.content[:_PROMPT_MESSAGE_LIMIT])
        for row in rows
        if row.content.strip()
    ]


async def save_exchange(
    session_id: int,
    *,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    model: str | None,
    route: str | None,
    document_ids: list[int] | None,
) -> bool:
    """Soru ve cevabı oturuma kaydeder. Kendi DB oturumunu açar (akış bittikten sonra çağrılır).

    İlk mesajda başlık sorudan üretilir. Oturum bu sırada silinmişse False döner.
    Hata fırlatmaz: kayıt başarısız olsa da kullanıcı cevabını almış olmalıdır.
    """
    try:
        async with AsyncSessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            if chat is None:
                return False

            db.add(ChatMessage(session_id=session_id, role="user", content=question))
            await db.flush()  # kimlik sırası: önce soru, sonra cevap
            db.add(
                ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                    sources=sources or None,
                    model=model,
                    route=route,
                )
            )
            if not chat.title or chat.title == DEFAULT_SESSION_TITLE:
                chat.title = make_title(question)
            chat.document_ids = document_ids or None
            chat.updated_at = func.now()
            await db.commit()
        return True
    except Exception:
        logger.exception("Oturum %s için soru-cevap kaydedilemedi.", session_id)
        return False
