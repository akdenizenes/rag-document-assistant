"""Ortak FastAPI dependency'leri.

NOT: Gerçek kimlik doğrulama (JWT) sonraki aşamada eklenecek. O zamana kadar
tüm istekler tek bir "demo" kullanıcı adına çalışır. Route'lar yalnızca
`get_current_user` dependency'sine bağlı olduğu için, auth eklendiğinde
sadece bu fonksiyon değişecek.
"""

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User

DEMO_EMAIL = "demo@local.dev"


async def get_current_user(db: AsyncSession = Depends(get_db)) -> User:
    result = await db.execute(select(User).where(User.email == DEMO_EMAIL))
    user = result.scalar_one_or_none()
    if user is not None:
        return user

    user = User(
        email=DEMO_EMAIL,
        full_name="Demo Kullanıcı",
        hashed_password="!placeholder",  # geçerli bir hash değil, giriş yapılamaz
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        # Eşzamanlı ilk isteklerde başka bir istek kullanıcıyı zaten oluşturmuş olabilir.
        await db.rollback()
        result = await db.execute(select(User).where(User.email == DEMO_EMAIL))
        return result.scalar_one()
    await db.refresh(user)
    return user
