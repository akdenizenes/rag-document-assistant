"""RAG Belge Asistanı - Streamlit arayüzü.

Çalıştırma (proje kökünden, API'nin ayakta olduğundan emin ol):
    streamlit run frontend/streamlit_app.py

Ayarlar .env dosyasından okunur: API_BASE_URL, STATUS_POLL_SECONDS, API_TIMEOUT_SECONDS,
CHAT_TIMEOUT_SECONDS, DEFAULT_TOP_K, MAX_UPLOAD_MB, MAX_SCOPE_DOCUMENTS.

Arayüz hataya dayanıklıdır: API/veritabanı/LLM hataları sayfayı çökertmez; ilgili bölümde
anlaşılır bir mesaj gösterilir, geri kalan arayüz çalışmaya devam eder.
"""

from __future__ import annotations

import functools
import itertools
import logging
import os
import sys
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import ApiClient, ApiError  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("streamlit_app")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT_SECONDS", "200"))
POLL_SECONDS = float(os.getenv("STATUS_POLL_SECONDS", "3"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_SCOPE_DOCUMENTS = int(os.getenv("MAX_SCOPE_DOCUMENTS", "20"))

STATUS_LABELS = {
    "pending": "⏳ Sırada",
    "processing": "⚙️ İşleniyor",
    "completed": "✅ Hazır",
    "failed": "❌ Hata",
}
PENDING_STATUSES = {"pending", "processing"}
# Sorunun hangi yoldan yanıtlandığı (backend `route` alanı)
ROUTE_LABELS = {
    "summary": "📝 Belge özeti",
    "full_text": "📖 Tam metin",
    "hybrid": "🔎 Hibrit arama",
}
MATCH_LABELS = {
    "vector": "anlamsal",
    "fts": "anahtar kelime",
    "both": "anlamsal + anahtar kelime",
}
# Özet henüz hazırlanıyor sayılan en uzun süre (saniye); daha eski 'pending' = hiç başlamamış
SUMMARY_WAIT_SECONDS = 600
SESSION_TITLE_DISPLAY = 32  # kenar çubuğunda gösterilen en fazla başlık uzunluğu

st.set_page_config(page_title="RAG Belge Asistanı", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Hata dayanıklılığı
# --------------------------------------------------------------------------- #
def guarded(label: str) -> Callable:
    """Bir arayüz bölümünü sarar: hata olursa sayfa çökmez, bölümde mesaj gösterilir."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ApiError as exc:
                st.error(f"{label}: {exc}")
            except Exception:
                logger.exception("%s sırasında beklenmeyen hata", label)
                st.error(f"{label}: beklenmeyen bir hata oluştu. Sayfayı yenilemeyi dene.")
                with st.expander("Teknik ayrıntı"):
                    st.code(traceback.format_exc())
            return None

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
@st.cache_resource
def _get_api_client() -> ApiClient:
    return ApiClient(
        API_BASE_URL,
        timeout=API_TIMEOUT,
        upload_timeout=max(API_TIMEOUT, 120.0),
        chat_timeout=CHAT_TIMEOUT,
    )


def get_api() -> ApiClient:
    """API istemcisini güncel oturum token'ı ile döndürür."""
    client = _get_api_client()
    client.set_token(st.session_state.get("token"))
    return client


def init_state() -> None:
    state = st.session_state
    state.setdefault("token", None)  # Kullanıcı JWT erişim token'ı
    state.setdefault("user_email", None)
    state.setdefault("user_name", None)
    state.setdefault("docs", [])  # son bilinen belge listesi
    state.setdefault("messages", [])  # ekrandaki sohbetin mesajları
    state.setdefault("session_id", None)  # açık sohbet oturumu
    state.setdefault("sessions", None)  # sohbet listesi önbelleği
    state.setdefault("sessions_error", None)
    state.setdefault("scope_ids", [])  # seçili belgeler
    state.setdefault("uploader_key", 0)  # yükleme alanını sıfırlamak için
    state.setdefault("flash", [])  # rerun sonrası gösterilecek mesajlar
    state.setdefault("_polling", False)
    state.setdefault("_has_pending", False)


def flash(kind: str, message: str) -> None:
    st.session_state.flash.append((kind, message))


def show_flash() -> None:
    renderers = {"error": st.error, "success": st.success, "warning": st.warning}
    for kind, message in st.session_state.flash:
        renderers.get(kind, st.info)(message)
    st.session_state.flash = []


def logout() -> None:
    """Oturumu sonlandırır ve tüm state'i temizler."""
    token_keys = ["token", "user_email", "user_name", "session_id", "messages", "sessions", "docs", "scope_ids"]
    for key in token_keys:
        st.session_state[key] = None
    st.session_state["messages"] = []
    st.session_state["docs"] = []
    st.session_state["scope_ids"] = []
    get_api().set_token(None)
    st.rerun()


def fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def fmt_datetime(value: str | None) -> str:
    """ISO tarihini yerel saatte 'gg.aa ss:dd' olarak biçimlendirir."""
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d.%m %H:%M")
    except (TypeError, ValueError):
        return ""


def summary_in_flight(doc: dict) -> bool:
    """Belge hazır ama özeti hâlâ üretiliyor mu?"""
    if doc["status"] != "completed":
        return False
    status = doc.get("summary_status")
    if status == "processing":
        return True
    if status != "pending":
        return False
    try:
        created = datetime.fromisoformat(doc["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds() < SUMMARY_WAIT_SECONDS
    except (KeyError, ValueError, TypeError):
        return False


def render_doc_summary(doc: dict) -> None:
    summary = doc.get("summary")
    meta = doc.get("doc_metadata") or {}
    if summary:
        with st.expander("📝 Özet ve konular"):
            if meta.get("title"):
                st.markdown(f"**{meta['title']}**")
            facts = []
            if meta.get("document_type"):
                facts.append(str(meta["document_type"]))
            if meta.get("question_count") is not None:
                facts.append(f"~{meta['question_count']} soru")
            if meta.get("section_count") is not None:
                facts.append(f"~{meta['section_count']} bölüm")
            if facts:
                st.caption(" · ".join(facts))
            st.write(summary)
            topics = meta.get("main_topics") or []
            if topics:
                st.caption("Ana konular: " + "; ".join(topics))
    elif summary_in_flight(doc):
        st.caption("📝 Özet hazırlanıyor...")
    elif doc.get("summary_status") == "failed":
        st.caption("📝 Özet üretilemedi")


# --------------------------------------------------------------------------- #
# Kimlik Doğrulama Ekranı
# --------------------------------------------------------------------------- #
def render_auth_screen() -> None:
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.title("🔐 RAG Belge Asistanı")
        st.caption("Belgeleriniz ve sohbet geçmişinize erişmek için lütfen giriş yapın.")
        show_flash()

        tab_login, tab_register = st.tabs(["🔑 Giriş Yap", "📝 Kayıt Ol"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("E-posta", placeholder="ornek@domain.com")
                password = st.text_input("Parola", type="password")
                submit = st.form_submit_button("Giriş Yap", use_container_width=True, type="primary")

                if submit:
                    if not email or not password:
                        st.error("Lütfen e-posta ve parolanızı girin.")
                    else:
                        try:
                            data = get_api().login(email=email.strip(), password=password)
                            st.session_state["token"] = data["access_token"]
                            st.session_state["user_email"] = data["email"]
                            st.session_state["user_name"] = data.get("full_name")
                            flash("success", f"Hoş geldiniz, {data.get('full_name') or data['email']}!")
                            st.rerun()
                        except ApiError as exc:
                            st.error(f"Giriş yapılamadı: {exc}")

        with tab_register:
            with st.form("register_form"):
                reg_name = st.text_input("Ad Soyad (isteğe bağlı)")
                reg_email = st.text_input("E-posta", placeholder="ornek@domain.com")
                reg_password = st.text_input("Parola (en az 6 karakter)", type="password")
                submit_reg = st.form_submit_button("Kayıt Ol", use_container_width=True)

                if submit_reg:
                    if not reg_email or not reg_password:
                        st.error("E-posta ve parola alanları zorunludur.")
                    elif len(reg_password) < 6:
                        st.error("Parola en az 6 karakter olmalıdır.")
                    else:
                        try:
                            data = get_api().register(
                                email=reg_email.strip(),
                                password=reg_password,
                                full_name=reg_name.strip() if reg_name else None,
                            )
                            st.session_state["token"] = data["access_token"]
                            st.session_state["user_email"] = data["email"]
                            st.session_state["user_name"] = data.get("full_name")
                            flash("success", "Hesabınız başarıyla oluşturuldu!")
                            st.rerun()
                        except ApiError as exc:
                            st.error(f"Kayıt oluşturulamadı: {exc}")


# --------------------------------------------------------------------------- #
# Sohbet oturumları (geçmiş)
# --------------------------------------------------------------------------- #
def message_from_api(message: dict) -> dict:
    """API'den gelen kayıtlı mesajı ekran biçimine çevirir."""
    return {
        "role": message["role"],
        "content": message["content"],
        "sources": message.get("sources") or [],
        "model": message.get("model"),
        "route": message.get("route"),
    }


def start_new_chat() -> None:
    """Yeni bir sohbet başlatır. İlk soruda veritabanında oturum oluşturulur."""
    st.session_state.session_id = None
    st.session_state.messages = []


def open_session(session_id: int) -> None:
    """Kayıtlı bir sohbeti açar (mesajlarını ve belge kapsamını geri yükler)."""
    state = st.session_state
    try:
        detail = get_api().get_session(session_id)
    except ApiError as exc:
        flash("error", f"Sohbet açılamadı: {exc}")
        state.sessions = None
        return
    state.session_id = detail["id"]
    state.messages = [message_from_api(m) for m in detail.get("messages", [])]
    ready_ids = {d["id"] for d in state.docs if d["status"] == "completed"}
    state.scope_ids = [i for i in (detail.get("document_ids") or []) if i in ready_ids]


def delete_session(session_id: int) -> None:
    state = st.session_state
    try:
        get_api().delete_session(session_id)
    except ApiError as exc:
        flash("error", f"Sohbet silinemedi: {exc}")
        return
    if state.session_id == session_id:
        start_new_chat()
    state.sessions = None
    flash("success", "Sohbet silindi.")


def refresh_sessions(force: bool = False) -> None:
    """Sohbet listesini yükler."""
    state = st.session_state
    if state.sessions is not None and not force:
        return
    try:
        state.sessions = get_api().list_sessions()
        state.sessions_error = None
    except ApiError as exc:
        state.sessions = state.sessions or []
        state.sessions_error = str(exc)


@guarded("Sohbet geçmişi")
def render_sessions() -> None:
    state = st.session_state
    st.button("➕ Yeni sohbet", on_click=start_new_chat, type="primary", key="new_chat")
    refresh_sessions()

    if state.sessions_error:
        st.caption(f":red[Geçmiş yüklenemedi: {state.sessions_error}]")
        if st.button("Tekrar dene", key="retry_sessions"):
            state.sessions = None
            st.rerun()

    if not any(i.get("message_count") for i in state.sessions or []) and not state.sessions_error:
        st.caption("Henüz kayıtlı sohbet yok. İlk sorunu sorunca burada görünür.")

    for item in state.sessions or []:
        active = item["id"] == state.session_id
        if not active and not item.get("message_count"):
            continue
        title = item["title"]
        if len(title) > SESSION_TITLE_DISPLAY:
            title = title[: SESSION_TITLE_DISPLAY - 1] + "…"
        col_open, col_delete = st.columns([5, 1])
        with col_open:
            st.button(
                ("▶ " if active else "") + title,
                key=f"open_{item['id']}",
                on_click=open_session,
                args=(item["id"],),
                disabled=active,
                help=f"{item.get('message_count', 0)} mesaj · {fmt_datetime(item.get('updated_at'))}",
            )
        with col_delete:
            with st.popover("🗑️"):
                st.write("Bu sohbet ve mesajları kalıcı olarak silinecek.")
                st.button(
                    "Evet, sil",
                    key=f"delete_session_{item['id']}",
                    type="primary",
                    on_click=delete_session,
                    args=(item["id"],),
                )


# --------------------------------------------------------------------------- #
# Kenar çubuğu: yükleme + belge listesi
# --------------------------------------------------------------------------- #
@guarded("Belge yükleme")
def render_upload() -> None:
    files = st.file_uploader(
        "PDF yükle",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"Dosya başına en fazla {MAX_UPLOAD_MB} MB.",
    )
    if not st.button("Yükle ve işle", type="primary", disabled=not files):
        return

    uploaded = 0
    for file in files:
        if file.size > MAX_UPLOAD_MB * 1024 * 1024:
            flash("warning", f"{file.name}: dosya {MAX_UPLOAD_MB} MB sınırını aşıyor.")
            continue
        try:
            with st.spinner(f"{file.name} yükleniyor..."):
                get_api().upload_document(file.name, file.getvalue())
            uploaded += 1
        except ApiError as exc:
            flash("error", f"{file.name}: {exc}")

    if uploaded:
        flash("success", f"{uploaded} dosya yüklendi, işleme başladı.")
        st.session_state.uploader_key += 1
    st.rerun()


@guarded("Belge listesi")
def render_documents() -> None:
    """Belge listesi."""
    api = get_api()
    state = st.session_state
    try:
        docs = api.list_documents()
    except ApiError as exc:
        st.error(f"Belgeler alınamadı: {exc}")
        if st.button("Tekrar dene", key="retry_docs"):
            st.rerun()
        return

    completed_ids = frozenset(d["id"] for d in docs if d["status"] == "completed")
    has_pending = any(d["status"] in PENDING_STATUSES or summary_in_flight(d) for d in docs)
    previous_completed = state.get("_completed_ids")

    state.docs = docs
    state._completed_ids = completed_ids
    state._has_pending = has_pending

    needs_rerun = has_pending != state._polling or (
        previous_completed is not None and previous_completed != completed_ids
    )

    st.subheader(f"Belgelerim ({len(docs)})")
    if not docs:
        st.caption("Henüz belge yok. Yukarıdan bir PDF yükle.")
    st.button("🔄 Yenile", key="refresh_docs")

    for doc in docs:
        with st.container(border=True):
            st.markdown(f"**{doc['filename']}**")
            parts = [STATUS_LABELS.get(doc["status"], doc["status"])]
            if doc["status"] == "completed":
                parts += [f"{doc['page_count']} sayfa", f"{doc['chunk_count']} parça"]
            parts.append(fmt_size(doc["file_size"]))
            st.caption(" · ".join(parts))
            if doc["status"] == "failed" and doc.get("error_message"):
                st.caption(f":red[{doc['error_message']}]")
            render_doc_summary(doc)

            with st.popover("🗑️ Sil"):
                st.write("Belge ve tüm parçaları kalıcı olarak silinecek.")
                if st.button("Evet, sil", key=f"delete_{doc['id']}", type="primary"):
                    try:
                        api.delete_document(doc["id"])
                        flash("success", f"{doc['filename']} silindi.")
                    except ApiError as exc:
                        flash("error", f"Silinemedi: {exc}")
                    st.rerun()

    if needs_rerun:
        st.rerun()


def render_sidebar() -> tuple[int, bool]:
    with st.sidebar:
        # Kullanıcı bilgisi ve çıkış butonu
        user_display = st.session_state.get("user_name") or st.session_state.get("user_email") or "Kullanıcı"
        col_user, col_logout = st.columns([3, 1])
        with col_user:
            st.markdown(f"👤 **{user_display}**")
        with col_logout:
            if st.button("Çıkış", help="Oturumu kapat"):
                logout()

        st.divider()
        st.header("📄 RAG Asistanı")
        show_flash()
        tab_docs, tab_chats = st.tabs(["📄 Belgeler", "💬 Sohbetler"])

        with tab_docs:
            render_upload()
            st.divider()
            polling = bool(st.session_state._has_pending)
            st.session_state._polling = polling
            st.fragment(run_every=POLL_SECONDS if polling else None)(render_documents)()

        with tab_chats:
            render_sessions()

        st.divider()
        top_k = st.slider(
            "Getirilecek parça sayısı (top_k)",
            min_value=1,
            max_value=10,
            value=min(max(DEFAULT_TOP_K, 1), 10),
            help="Hibrit arama rotasında cevap için kaç parça getirileceği.",
        )
        use_stream = st.toggle(
            "Cevabı canlı yaz (streaming)",
            value=True,
            help="Kapalıyken cevap hazır olunca tek seferde gösterilir.",
        )
    return top_k, use_stream


# --------------------------------------------------------------------------- #
# Ana alan: sohbet
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Kaynaklar ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            matched_by = source.get("matched_by")
            if matched_by in ("summary", "full_text"):
                st.markdown(f"**[{index}] {source['document_name']}** · {ROUTE_LABELS[matched_by]}")
            else:
                page = (
                    f"sayfa {source['page_number']}"
                    if source.get("page_number")
                    else "sayfa belirtilmemiş"
                )
                how = f" · {MATCH_LABELS[matched_by]}" if matched_by in MATCH_LABELS else ""
                rerank = (
                    f" · yeniden sıralama `{source['rerank_score']:.2f}`"
                    if source.get("rerank_score") is not None
                    else ""
                )
                st.markdown(
                    f"**[{index}] {source['document_name']}** · {page} · "
                    f"skor `{source['score']:.2f}`{how}{rerank}"
                )
            snippet = " ".join(source["content"].split())
            st.text(snippet if len(snippet) <= 400 else snippet[:400] + "…")


def caption_model(model: str | None, route: str | None) -> None:
    parts = []
    if model:
        parts.append(f"Model: {model}")
    if route:
        parts.append(f"Rota: {ROUTE_LABELS.get(route, route)}")
    if parts:
        st.caption(" · ".join(parts))


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        content = message.get("content") or message.get("answer") or ""
        st.markdown(content)
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            caption_model(message.get("model"), message.get("route"))


def ensure_session(scope_ids: list[int]) -> int | None:
    """Açık oturum yoksa oluşturur."""
    state = st.session_state
    if state.session_id is not None:
        return state.session_id
    try:
        created = get_api().create_session(document_ids=scope_ids or None)
    except ApiError as exc:
        st.warning(f"Sohbet geçmişine kaydedilemiyor ({exc}). Bu soru kaydedilmeden yanıtlanacak.")
        return None
    state.session_id = created["id"]
    state.sessions = None
    return created["id"]


def handle_chat_error(exc: ApiError) -> None:
    state = st.session_state
    st.error(str(exc))
    if state.session_id is not None and "Sohbet bulunamadı" in str(exc):
        state.session_id = None
        state.sessions = None
        st.info("Bu sohbet silinmiş. Sorun yeni bir sohbet olarak devam edecek.")


def answer_streaming(
    prompt: str, scope_ids: list[int], history: list[dict], top_k: int, session_id: int | None
) -> dict | None:
    try:
        stream = get_api().chat_stream(
            prompt, document_ids=scope_ids, history=history, top_k=top_k, session_id=session_id
        )
    except ApiError as exc:
        handle_chat_error(exc)
        return None

    tokens = stream.tokens()
    with st.spinner("Belgelerde aranıyor ve cevap hazırlanıyor..."):
        first = next(tokens, None)

    answer = ""
    if first is not None:
        answer = st.write_stream(itertools.chain([first], tokens))
    stream.close()

    if stream.error:
        st.error(stream.error)
        return None
    if not answer:
        st.error("Modelden boş yanıt geldi.")
        return None

    render_sources(stream.sources)
    caption_model(stream.model, stream.route)
    return {
        "answer": answer,
        "sources": stream.sources,
        "model": stream.model,
        "route": stream.route,
    }


def answer_blocking(
    prompt: str, scope_ids: list[int], history: list[dict], top_k: int, session_id: int | None
) -> dict | None:
    try:
        with st.spinner("Belgelerde aranıyor ve cevap hazırlanıyor..."):
            result = get_api().chat(
                prompt, document_ids=scope_ids, history=history, top_k=top_k, session_id=session_id
            )
    except ApiError as exc:
        handle_chat_error(exc)
        return None
    st.markdown(result["answer"])
    render_sources(result["sources"])
    caption_model(result["model"], result.get("route"))
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "model": result["model"],
        "route": result.get("route"),
    }


@guarded("Sohbet")
def render_chat(top_k: int, use_stream: bool) -> None:
    state = st.session_state
    ready = [d for d in state.docs if d["status"] == "completed"]
    names = {d["id"]: d["filename"] for d in ready}

    state.scope_ids = [i for i in state.scope_ids if i in names]

    scope_ids: list[int] = st.multiselect(
        "Sohbet kapsamı",
        options=list(names),
        key="scope_ids",
        format_func=lambda doc_id: f"📄 {names[doc_id]}",
        placeholder="📚 Tüm belgelerde ara (belge seçerek daraltabilirsin)",
        disabled=not ready,
        max_selections=MAX_SCOPE_DOCUMENTS,
        help="Boş bırakırsan tüm belgelerde aranır; bir ya da birden fazla belge seçersen yalnızca onlarda aranır.",
    )

    if not ready:
        st.info("Başlamak için soldan bir PDF yükle. İşlem bitince burada soru sorabilirsin.")
    elif not state.messages:
        st.caption("Seçili kapsamdaki belgeler hakkında bir soru sor. Takip soruları da sorabilirsin.")

    for message in state.messages:
        render_message(message)

    prompt = st.chat_input("Sorunu yaz...", disabled=not ready, max_chars=2000)
    if not prompt:
        return

    first_exchange = not state.messages
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        session_id = ensure_session(scope_ids)
        answer_fn = answer_streaming if use_stream else answer_blocking
        result = answer_fn(prompt, scope_ids, state.messages, top_k, session_id)
    if result is None:
        return

    state.messages.append({"role": "user", "content": prompt})
    state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "model": result.get("model"),
        "route": result.get("route"),
    })
    state.sessions = None
    if first_exchange:
        st.rerun()


# --------------------------------------------------------------------------- #
# Ana Uygulama Döngüsü
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()

    try:
        api_up = get_api().health()
    except Exception:
        logger.exception("Sağlık kontrolü başarısız")
        api_up = False

    if not api_up:
        st.error(
            f"API'ye ulaşılamıyor ({API_BASE_URL}). Önce şunu çalıştır:\n\n"
            "`uvicorn app.main:app --reload`\n\n"
            "(İlk açılışta embedding modeli yüklenirken birkaç dakika sürebilir.)"
        )
        if st.button("Tekrar dene"):
            st.rerun()
        st.stop()

    if not get_api().ready():
        st.warning(
            "Veritabanına şu anda ulaşılamıyor; belgeler ve sohbet geçmişi yüklenemeyebilir. "
            "PostgreSQL'in çalıştığından emin ol (`docker compose up -d`)."
        )

    # Oturum açılmamışsa sadece Login / Register formunu göster
    if not st.session_state.get("token"):
        render_auth_screen()
        return

    st.title("📄 RAG Belge Asistanı")
    sidebar = guarded("Kenar çubuğu")(render_sidebar)()
    top_k, use_stream = sidebar if sidebar else (DEFAULT_TOP_K, True)
    render_chat(top_k, use_stream)


main()