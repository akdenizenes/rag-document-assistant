"""RAG Belge Asistanı - Streamlit arayüzü (SSE Canlı Akış + Rota & Özet/Metaveri Desteği).

Çalıştırma (proje kökünden, API'nin ayakta olduğundan emin ol):
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import ApiClient, ApiError  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT_SECONDS", "200"))
POLL_SECONDS = float(os.getenv("STATUS_POLL_SECONDS", "3"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

STATUS_LABELS = {
    "pending": "⏳ Sırada",
    "processing": "⚙️ İşleniyor",
    "completed": "✅ Hazır",
    "failed": "❌ Hata",
}
ROUTE_LABELS = {
    "summary": "📋 Özet Rotası (Doğrudan Özet)",
    "full_text": "📖 Tam Metin Rotası (Sayma / Tarama)",
    "hybrid": "🔍 Hibrit Arama Rotası (pgvector + FTS RRF)",
}
PENDING_STATUSES = {"pending", "processing"}

st.set_page_config(page_title="RAG Belge Asistanı", page_icon="📄", layout="wide")


@st.cache_resource
def get_api() -> ApiClient:
    return ApiClient(
        API_BASE_URL,
        timeout=API_TIMEOUT,
        upload_timeout=max(API_TIMEOUT, 120.0),
        chat_timeout=CHAT_TIMEOUT,
    )


def init_state() -> None:
    state = st.session_state
    state.setdefault("docs", [])
    state.setdefault("chats", {})
    state.setdefault("uploader_key", 0)
    state.setdefault("flash", [])
    state.setdefault("scope", None)
    state.setdefault("_polling", False)
    state.setdefault("_has_pending", False)


def flash(kind: str, message: str) -> None:
    st.session_state.flash.append((kind, message))


def show_flash() -> None:
    renderers = {"error": st.error, "success": st.success, "warning": st.warning}
    for kind, message in st.session_state.flash:
        renderers.get(kind, st.info)(message)
    st.session_state.flash = []


def fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# --------------------------------------------------------------------------- #
# Kenar çubuğu: yükleme + belge listesi
# --------------------------------------------------------------------------- #
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


def render_documents() -> None:
    """Belge listesi. Bekleyen belge veya özet varken kendi kendini yeniler."""
    api = get_api()
    try:
        docs = api.list_documents()
    except ApiError as exc:
        st.error(str(exc))
        return

    state = st.session_state
    completed_ids = frozenset(d["id"] for d in docs if d["status"] == "completed")
    has_pending = any(
        d["status"] in PENDING_STATUSES
        or (d["status"] == "completed" and d.get("summary_status") in PENDING_STATUSES)
        for d in docs
    )
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
                sum_status = doc.get("summary_status", "pending")
                if sum_status == "completed":
                    parts.append("✨ Özet hazır")
                elif sum_status in PENDING_STATUSES:
                    parts.append("⏳ Özetleniyor")
            parts.append(fmt_size(doc["file_size"]))
            st.caption(" · ".join(parts))

            if doc["status"] == "failed" and doc.get("error_message"):
                st.caption(f":red[{doc['error_message']}]")

            if doc.get("summary"):
                with st.expander("📋 Belge Özeti ve Metaveri"):
                    st.write(doc["summary"])
                    meta = doc.get("doc_metadata") or {}
                    if meta:
                        st.json(meta)

            with st.popover("🗑️ Sil"):
                st.write("Belge ve tüm parçaları kalıcı olarak silinecek.")
                if st.button("Evet, sil", key=f"delete_{doc['id']}", type="primary"):
                    try:
                        api.delete_document(doc["id"])
                        state.chats.pop(doc["id"], None)
                        flash("success", f"{doc['filename']} silindi.")
                    except ApiError as exc:
                        flash("error", f"Silinemedi: {exc}")
                    st.rerun()

    if needs_rerun:
        st.rerun()


def render_sidebar() -> int:
    with st.sidebar:
        st.header("📄 Belgeler")
        show_flash()
        render_upload()
        st.divider()

        polling = bool(st.session_state._has_pending)
        st.session_state._polling = polling
        st.fragment(run_every=POLL_SECONDS if polling else None)(render_documents)()

        st.divider()
        top_k = st.slider(
            "Getirilecek parça sayısı (top_k)",
            min_value=1,
            max_value=10,
            value=min(max(DEFAULT_TOP_K, 1), 10),
            help="Hibrit aramada belgeden kaç parça getirileceği.",
        )
    return top_k


# --------------------------------------------------------------------------- #
# Ana alan: sohbet
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict], route: str = "hybrid") -> None:
    if not sources:
        return
    with st.expander(f"📎 Kaynaklar ({len(sources)}) · {ROUTE_LABELS.get(route, route)}"):
        for index, source in enumerate(sources, start=1):
            page = (
                f"sayfa {source['page_number']}"
                if source.get("page_number")
                else "sayfa belirtilmemiş"
            )
            st.markdown(
                f"**[{index}] {source['document_name']}** · {page} · skor `{source['score']:.2f}`"
            )
            snippet = " ".join(source["content"].split())
            st.text(snippet if len(snippet) <= 400 else snippet[:400] + "…")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            route = message.get("route", "hybrid")
            render_sources(message.get("sources", []), route=route)
            caption_parts = []
            if message.get("model"):
                caption_parts.append(f"Model: {message['model']}")
            if route:
                caption_parts.append(ROUTE_LABELS.get(route, route))
            if caption_parts:
                st.caption(" · ".join(caption_parts))


def render_chat(top_k: int) -> None:
    state = st.session_state
    ready = [d for d in state.docs if d["status"] == "completed"]
    names = {d["id"]: d["filename"] for d in ready}
    options = [None, *names]

    if state.scope not in options:
        state.scope = None

    col_scope, col_clear = st.columns([4, 1], vertical_alignment="bottom")
    with col_scope:
        scope = st.selectbox(
            "Sohbet kapsamı",
            options,
            key="scope",
            format_func=lambda x: "📚 Tüm belgelerde ara" if x is None else f"📄 {names[x]}",
            disabled=not ready,
        )
    history: list[dict] = state.chats.setdefault(scope, [])
    with col_clear:
        if st.button("Sohbeti temizle", disabled=not history):
            state.chats[scope] = []
            st.rerun()

    if not ready:
        st.info("Başlamak için soldan bir PDF yükle. İşlem bitince burada soru sorabilirsin.")
    elif not history:
        st.caption(
            "Seçili kapsamdaki belgeler hakkında soru sorabilirsin (Örn: 'Bu belgeyi özetle', "
            "'Toplam kaç soru var?', veya spesifik bir konu)."
        )

    for message in history:
        render_message(message)

    prompt = st.chat_input("Sorunu yaz...", disabled=not ready, max_chars=2000)
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_meta: dict[str, Any] = {
            "sources": [],
            "route": "hybrid",
            "model": "",
        }
        sources_placeholder = st.empty()

        def _on_sources(event: dict[str, Any]) -> None:
            stream_meta["sources"] = event.get("sources", [])
            stream_meta["route"] = event.get("route", "hybrid")
            stream_meta["model"] = event.get("model", "")

        try:
            answer_text = st.write_stream(
                get_api().chat_stream(
                    prompt,
                    document_id=scope,
                    history=history,
                    top_k=top_k,
                    on_sources=_on_sources,
                )
            )
        except ApiError as exc:
            st.error(str(exc))
            return

        with sources_placeholder.container():
            render_sources(stream_meta["sources"], route=stream_meta["route"])
        caption_parts = []
        if stream_meta["model"]:
            caption_parts.append(f"Model: {stream_meta['model']}")
        caption_parts.append(ROUTE_LABELS.get(stream_meta["route"], stream_meta["route"]))
        st.caption(" · ".join(caption_parts))

    history.append({"role": "user", "content": prompt})
    history.append(
        {
            "role": "assistant",
            "content": str(answer_text),
            "sources": stream_meta["sources"],
            "model": stream_meta["model"],
            "route": stream_meta["route"],
        }
    )


def main() -> None:
    init_state()
    st.title("📄 RAG Belge Asistanı")

    if not get_api().health():
        st.error(
            f"API'ye ulaşılamıyor ({API_BASE_URL}). Önce şunu çalıştır:\n\n"
            "`uvicorn app.main:app --reload`\n\n"
            "(İlk açılışta embedding modeli yüklenirken birkaç dakika sürebilir.)"
        )
        if st.button("Tekrar dene"):
            st.rerun()
        st.stop()

    top_k = render_sidebar()
    render_chat(top_k)


main()
