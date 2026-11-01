"""Belge yükleme ve yönetim endpoint'leri."""

import asyncio
import io
import zipfile
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    GENERIC_MIMES,
    MAX_UPLOAD_MB_BY_EXT,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
)
from app.database import get_db
from app.dependencies import get_current_user
from app.models import Document, DocumentChunk, DocumentStatus, User
from app.rate_limit import limit_default, limit_upload
from app.schemas.document import DocumentList, DocumentOut
from app.services.document_parser import decode_text_bytes
from app.services.document_service import delete_file, process_document, save_file

router = APIRouter(tags=["documents"], dependencies=[Depends(limit_default)])


def _to_out(doc: Document, chunk_count: int = 0) -> DocumentOut:
    out = DocumentOut.model_validate(doc)
    out.chunk_count = chunk_count
    return out


_SUPPORTED_LABEL = ", ".join(SUPPORTED_EXTENSIONS)


def _max_bytes(extension: str) -> int:
    return MAX_UPLOAD_MB_BY_EXT[extension] * 1024 * 1024


async def _read_limited(file: UploadFile, extension: str) -> bytes:
    """Dosyayı parça parça okur, biçime özgü boyut sınırını aşarsa 413 döner."""
    limit = _max_bytes(extension)
    buffer = bytearray()
    while block := await file.read(1024 * 1024):
        buffer.extend(block)
        if len(buffer) > limit:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Dosya çok büyük ({extension} için en fazla {MAX_UPLOAD_MB_BY_EXT[extension]} MB).",
            )
    return bytes(buffer)


def _validate_extension_and_mime(filename: str, content_type: str | None) -> str:
    """Uzantıyı ve (varsa) istemcinin bildirdiği MIME türünü doğrular; küçük harfli uzantıyı döner."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Desteklenmeyen dosya biçimi. Kabul edilenler: {_SUPPORTED_LABEL}",
        )
    # Content-Type istemci tarafından gönderilir, güvenilmezdir: yalnızca açıkça başka bir
    # biçimi bildiriyorsa (ör. .pdf uzantılı image/png) reddedilir. Asıl doğrulama içeriktedir.
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime not in GENERIC_MIMES and mime not in SUPPORTED_FORMATS[extension]["accepted_mimes"]:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Dosya türü ({mime}) {extension} uzantısıyla uyuşmuyor.",
        )
    return extension


def _validate_content(extension: str, data: bytes) -> None:
    """Dosya içeriğini (magic bytes) uzantıyla karşılaştırır; uyuşmazlıkta 400 döner."""
    if extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Dosya geçerli bir PDF değil.")
    elif extension == ".docx":
        if not data.startswith(b"PK\x03\x04"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Dosya geçerli bir DOCX değil.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        "Dosya geçerli bir DOCX değil (eski .doc veya başka bir ZIP olabilir).",
                    )
        except zipfile.BadZipFile as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "DOCX dosyası bozuk veya geçerli değil."
            ) from exc
    elif extension == ".txt":
        utf16 = data.startswith((b"\xff\xfe", b"\xfe\xff"))
        if b"\x00" in data and not utf16:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Dosya düz metin değil (ikili içerik algılandı)."
            )
        if not decode_text_bytes(data).strip():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Metin dosyası yalnızca boşluk içeriyor.")


async def _get_owned_document(db: AsyncSession, user: User, document_id: int) -> Document:
    doc = await db.scalar(
        select(Document).where(Document.id == document_id, Document.user_id == user.id)
    )
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Belge bulunamadı.")
    return doc


@router.post(
    "/upload",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(limit_upload)],
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Yüklenecek belge (.pdf, .docx veya .txt)"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentOut:
    """PDF/DOCX/TXT dosyasını kaydeder ve işlemeyi (chunking + embedding) arka planda başlatır.

    Yanıt hemen döner (202). İlerlemeyi `GET /documents/{id}` ile `status` alanından izle:
    pending -> processing -> completed | failed
    """
    filename = Path(file.filename or "").name
    extension = _validate_extension_and_mime(filename, file.content_type)

    data = await _read_limited(file, extension)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Dosya boş.")
    _validate_content(extension, data)

    doc = Document(
        user_id=user.id,
        filename=filename[:255],
        content_type=SUPPORTED_FORMATS[extension]["mime"],
        file_size=len(data),
        status=DocumentStatus.PENDING.value,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        await asyncio.to_thread(save_file, doc.id, data, extension)
    except OSError as exc:
        await db.execute(delete(Document).where(Document.id == doc.id))
        await db.commit()
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Dosya diske kaydedilemedi."
        ) from exc

    background_tasks.add_task(process_document, doc.id)
    return _to_out(doc)


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentList:
    """Kullanıcının belgelerini (en yeni önce) parça sayılarıyla listeler."""
    total = await db.scalar(
        select(func.count()).select_from(Document).where(Document.user_id == user.id)
    )
    stmt = (
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(Document.user_id == user.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return DocumentList(total=total or 0, items=[_to_out(d, n) for d, n in rows])


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentOut:
    """Tek bir belgenin durumunu ve bilgilerini döner."""
    doc = await _get_owned_document(db, user, document_id)
    chunk_count = await db.scalar(
        select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    )
    return _to_out(doc, chunk_count or 0)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    """Belgeyi, parçalarını (DB seviyesinde CASCADE) ve diskteki dosyasını siler."""
    doc = await _get_owned_document(db, user, document_id)
    # ORM yerine doğrudan DELETE: parçaları belleğe yüklemeden FK CASCADE devreye girer.
    await db.execute(delete(Document).where(Document.id == doc.id))
    await db.commit()
    await asyncio.to_thread(delete_file, document_id)
