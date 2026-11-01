"""Sohbet oturumu (geçmiş) endpoint'leri.

  POST   /sessions        -> yeni oturum
  GET    /sessions        -> oturum listesi (en son kullanılan önce)
  GET    /sessions/{id}   -> oturum + mesajları
  PATCH  /sessions/{id}   -> başlığı değiştir
  DELETE /sessions/{id}   -> oturumu ve mesajlarını sil

Mesajlar `POST /chat` ve `POST /chat/stream` çağrılarına `session_id` verilince otomatik kaydedilir.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import SESSION_MESSAGES_MAX
from app.database import get_db
from app.dependencies import get_current_user
from app.models import ChatMessage, ChatSession, User
from app.rate_limit import limit_default
from app.schemas.session import (
    MessageOut,
    SessionCreate,
    SessionDetail,
    SessionList,
    SessionOut,
    SessionUpdate,
)
from app.services.session_service import create_session, get_owned_session

router = APIRouter(tags=["sessions"], dependencies=[Depends(limit_default)])


def _to_out(chat: ChatSession, message_count: int = 0) -> SessionOut:
    out = SessionOut.model_validate(chat)
    out.message_count = message_count
    return out


async def _require_session(db: AsyncSession, user: User, session_id: int) -> ChatSession:
    chat = await get_owned_session(db, user.id, session_id)
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sohbet bulunamadı.")
    return chat


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionOut:
    chat = await create_session(
        db, user.id, title=payload.title, document_ids=payload.document_ids
    )
    return _to_out(chat)


@router.get("/sessions", response_model=SessionList)
async def list_chat_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionList:
    """Kullanıcının sohbetlerini (en son güncellenen önce) mesaj sayılarıyla listeler."""
    total = await db.scalar(
        select(func.count()).select_from(ChatSession).where(ChatSession.user_id == user.id)
    )
    stmt = (
        select(ChatSession, func.count(ChatMessage.id))
        .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
        .where(ChatSession.user_id == user.id)
        .group_by(ChatSession.id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return SessionList(total=total or 0, items=[_to_out(c, n) for c, n in rows])


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_chat_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionDetail:
    """Oturumu ve mesajlarını (eskiden yeniye, en fazla SESSION_MESSAGES_MAX adet) döner."""
    chat = await _require_session(db, user, session_id)
    rows = (
        await db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == chat.id)
            .order_by(ChatMessage.id.desc())
            .limit(SESSION_MESSAGES_MAX)
        )
    ).all()
    messages = [MessageOut.model_validate(m) for m in reversed(rows)]
    # SessionDetail.model_validate(chat) KULLANMA: ORM'deki `messages` ilişkisini tembel yüklemeye
    # çalışır (async oturumda MissingGreenlet). Alanlar SessionOut'tan alınır, mesajlar elle eklenir.
    base = SessionOut.model_validate(chat).model_dump()
    base["message_count"] = len(messages)
    return SessionDetail(**base, messages=messages)


@router.patch("/sessions/{session_id}", response_model=SessionOut)
async def rename_chat_session(
    session_id: int,
    payload: SessionUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionOut:
    chat = await _require_session(db, user, session_id)
    await db.execute(
        update(ChatSession).where(ChatSession.id == chat.id).values(title=payload.title)
    )
    await db.commit()
    await db.refresh(chat)
    return _to_out(chat)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    chat = await _require_session(db, user, session_id)
    # Doğrudan DELETE: mesajlar FK ON DELETE CASCADE ile veritabanı tarafından silinir.
    await db.execute(delete(ChatSession).where(ChatSession.id == chat.id))
    await db.commit()
