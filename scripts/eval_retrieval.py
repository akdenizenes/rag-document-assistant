"""Arama kalitesi değerlendirmesi: vektör / FTS / hibrit / yeniden sıralama karşılaştırması.

Her soru için "doğru sayfa" etiketli bir soru setiyle (JSONL) şu arama modlarını aynı koşullarda
çalıştırır ve Hit@k + MRR + gecikme tablosu üretir:

  vector                 yalnızca pgvector (kosinüs)
  fts                    yalnızca PostgreSQL Türkçe tam metin araması
  hybrid                 vektör + FTS, RRF ile birleştirilmiş (eşikler uygulanır)
  hybrid_score           hybrid + modelsiz skor filtresi  (RERANK_MODE=score, uygulamanın varsayılanı)
  hybrid_cross_encoder   hybrid adayları cross-encoder ile yeniden sıralanır

Soru seti biçimi (eval/questions.example.jsonl): satır başına bir JSON nesnesi.
  {"id": "q01", "question": "...", "doc": "Software Products", "pages": [13], "lang": "tr"}
    question : sorulacak metin
    doc      : belge adında geçen ifade (büyük/küçük harf duyarsız, "içerir" eşleşmesi)
    pages    : doğru cevabın bulunduğu sayfa(lar) -- parçanın `page_number` değeriyle karşılaştırılır
    lang     : isteğe bağlı etiket (tr / en ...); varsa dile göre ayrı tablo da üretilir
  `#` ile başlayan ve boş satırlar yok sayılır.

Ölçüt notları:
  - Eşleşme SAYFA düzeyindedir: doğru sayfadan gelen herhangi bir parça "isabet" sayılır. Aynı
    sayfada birden çok parça varsa bu hafif iyimser bir ölçümdür; sonuçları yorumlarken akılda tut.
  - Hit@k   : doğru sayfanın ilk k sonuç içinde bulunduğu soruların oranı (tek doğru sayfa için
              Recall@k ile aynıdır).
  - MRR     : ilk doğru sonucun sırasının tersinin ortalaması (bulunamazsa 0), en fazla max(k) sıra.
  - Eşikler (RETRIEVAL_MIN_SCORE vb.) uygulamadaki gibi UYGULANIR; yani rakamlar canlı davranışı yansıtır.

Kullanım (Docker ile; `eval/` klasörü imajda yoktur, bu yüzden bağlanır):
    docker compose run --rm -T -v "$PWD/eval:/app/eval:ro" api sh -c \\
      'export DATABASE_URL="$(python scripts/docker_db.py url)"; \\
       python scripts/eval_retrieval.py --email SENIN@EPOSTAN --questions eval/questions.jsonl' \\
      > eval/results.md

Yerelde (venv + .env hazırsa):
    python scripts/eval_retrieval.py --email SENIN@EPOSTAN --json-out eval/results.json

Markdown raporu stdout'a, günlükler stderr'e yazılır; bu yüzden `> eval/results.md` temiz çıktı verir.
Rapor, README'ye yapıştırılacağı için İngilizce üretilir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALL_MODES: tuple[str, ...] = (
    "vector",
    "fts",
    "hybrid",
    "hybrid_score",
    "hybrid_cross_encoder",
)
MODE_LABELS: dict[str, str] = {
    "vector": "Vector only (pgvector)",
    "fts": "Full-text only (tsvector)",
    "hybrid": "Hybrid (vector + FTS, RRF)",
    "hybrid_score": "Hybrid + score filter (app default)",
    "hybrid_cross_encoder": "Hybrid + cross-encoder rerank",
}

Retriever = Callable[[str, int], Awaitable[list]]


# --------------------------------------------------------------------------- #
# Soru seti
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    question: str
    doc: str
    pages: tuple[int, ...]
    lang: str = "-"


def load_questions(path: Path) -> list[EvalQuestion]:
    """JSONL soru setini okur ve doğrular. Hatalı satırda satır numarasıyla ValueError fırlatır."""
    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: geçersiz JSON ({exc.msg})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: satır bir JSON nesnesi olmalı.")

        text = obj.get("question")
        doc = obj.get("doc")
        pages = obj.get("pages")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{lineno}: 'question' boş olmayan bir metin olmalı.")
        if not isinstance(doc, str) or not doc.strip():
            raise ValueError(f"{path}:{lineno}: 'doc' boş olmayan bir metin olmalı.")
        if (
            not isinstance(pages, list)
            or not pages
            or not all(isinstance(p, int) and not isinstance(p, bool) and p > 0 for p in pages)
        ):
            raise ValueError(f"{path}:{lineno}: 'pages' pozitif tam sayılardan oluşan dolu bir liste olmalı.")

        qid = str(obj.get("id") or f"q{len(questions) + 1:02d}")
        if qid in seen_ids:
            raise ValueError(f"{path}:{lineno}: '{qid}' kimliği tekrar ediyor.")
        seen_ids.add(qid)
        questions.append(
            EvalQuestion(
                id=qid,
                question=text.strip(),
                doc=doc.strip(),
                pages=tuple(pages),
                lang=str(obj.get("lang") or "-"),
            )
        )
    if not questions:
        raise ValueError(f"{path}: hiç soru bulunamadı.")
    return questions


# --------------------------------------------------------------------------- #
# Metrikler (saf fonksiyonlar)
# --------------------------------------------------------------------------- #
def is_relevant(chunk, question: EvalQuestion) -> bool:
    """Parça doğru belgeden ve doğru sayfalardan biriyse True."""
    name = (getattr(chunk, "document_name", "") or "").lower()
    return question.doc.lower() in name and getattr(chunk, "page_number", None) in question.pages


def first_hit_rank(chunks: Sequence, question: EvalQuestion, max_k: int) -> int | None:
    """İlk `max_k` sonuç içindeki ilk isabetin sırası (1'den başlar); yoksa None."""
    for rank, chunk in enumerate(chunks[:max_k], start=1):
        if is_relevant(chunk, question):
            return rank
    return None


def summarize(ranks: Sequence[int | None], ks: Sequence[int]) -> dict:
    """Sıra listesinden Hit@k ve MRR hesaplar. Boş girdi için sıfırlar döner."""
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit": {k: 0.0 for k in ks}, "mrr": 0.0}
    hits = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return {"n": n, "hit": hits, "mrr": mrr}


@dataclass(slots=True)
class ModeResult:
    mode: str
    ranks: dict[str, int | None] = field(default_factory=dict)  # soru id -> ilk isabet sırası
    top: dict[str, list[dict]] = field(default_factory=dict)  # soru id -> ilk sonuçlar (hata ayıklama)
    latencies_ms: list[float] = field(default_factory=list)
    fallback_count: int = 0  # cross-encoder kullanılamayıp skor filtresine düşülen soru sayısı

    @property
    def invalid(self) -> bool:
        """Cross-encoder bir soruda bile devre dışı kaldıysa bu satırın rakamları güvenilmezdir."""
        return self.mode == "hybrid_cross_encoder" and self.fallback_count > 0


async def evaluate_mode(
    mode: str,
    retrieve: Retriever,
    questions: Sequence[EvalQuestion],
    *,
    max_k: int,
    is_fallback: Callable[[list], bool] | None = None,
) -> ModeResult:
    """Bir arama modunu tüm sorularda çalıştırır. İlk çağrı (model yükleme) ölçüme katılmaz."""
    await retrieve(questions[0].question, max_k)  # ısınma: modelleri yükler, süre ölçülmez

    result = ModeResult(mode=mode)
    for q in questions:
        started = time.perf_counter()
        chunks = await retrieve(q.question, max_k)
        result.latencies_ms.append((time.perf_counter() - started) * 1000.0)

        if is_fallback is not None and is_fallback(chunks):
            result.fallback_count += 1
        result.ranks[q.id] = first_hit_rank(chunks, q, max_k)
        result.top[q.id] = [
            {
                "doc": getattr(c, "document_name", None),
                "page": getattr(c, "page_number", None),
                "score": getattr(c, "score", None),
            }
            for c in chunks[:max_k]
        ]
    return result


# --------------------------------------------------------------------------- #
# Rapor
# --------------------------------------------------------------------------- #
def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _table(
    results: dict[str, ModeResult],
    questions: Sequence[EvalQuestion],
    ks: Sequence[int],
    *,
    with_latency: bool,
) -> list[str]:
    header = ["Retrieval mode", *[f"Hit@{k}" for k in ks], f"MRR@{max(ks)}"]
    if with_latency:
        header.append("Median latency")
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]

    ids = [q.id for q in questions]
    for mode, res in results.items():
        label = MODE_LABELS.get(mode, mode)
        if res.invalid:
            cells = [f"{label} ⚠"] + ["n/a"] * (len(header) - 1)
        else:
            s = summarize([res.ranks[i] for i in ids], ks)
            cells = [label, *[_pct(s["hit"][k]) for k in ks], f"{s['mrr']:.3f}"]
            if with_latency:
                cells.append(f"{statistics.median(res.latencies_ms):.0f} ms" if res.latencies_ms else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def format_markdown(
    info: dict[str, str],
    questions: Sequence[EvalQuestion],
    results: dict[str, ModeResult],
    ks: Sequence[int],
) -> str:
    """README'ye yapıştırılabilir Markdown raporu üretir."""
    out: list[str] = [f"## Retrieval evaluation ({date.today().isoformat()})", ""]
    out.append(
        f"{len(questions)} labelled questions; a hit means a chunk from the correct document and "
        "page appears in the top-k results."
    )
    out += ["", *_table(results, questions, ks, with_latency=True)]

    invalid = [r for r in results.values() if r.invalid]
    for r in invalid:
        out += [
            "",
            f"> ⚠ **{MODE_LABELS[r.mode]}**: the cross-encoder was unavailable for "
            f"{r.fallback_count}/{len(questions)} questions (the app silently falls back to the score filter), "
            "so its numbers are withheld. Check the model download / network and re-run.",
        ]

    langs = sorted({q.lang for q in questions})
    if len(langs) > 1:
        for lang in langs:
            subset = [q for q in questions if q.lang == lang]
            out += ["", f"### Language `{lang}` ({len(subset)} questions)", ""]
            out += _table(results, subset, ks, with_latency=False)

    focus = "hybrid" if "hybrid" in results else next(iter(results))
    misses = [q for q in questions if results[focus].ranks.get(q.id) is None]
    out += ["", f"### Misses for `{focus}` (not in top {max(ks)})", ""]
    if misses:
        for q in misses:
            out.append(f"- `{q.id}` ({q.lang}): {q.question}  → expected {q.doc}, p. {list(q.pages)}")
    else:
        out.append("None.")

    out += ["", "<details><summary>Setup</summary>", ""]
    out += [f"- {key}: {value}" for key, value in info.items()]
    out += ["", "</details>", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Çalıştırma (veritabanı gerektirir)
# --------------------------------------------------------------------------- #
def _parse_ks(raw: str) -> list[int]:
    try:
        ks = sorted({int(part) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks virgülle ayrılmış tam sayılar olmalı (örn. 1,3,5)") from exc
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--ks en az bir pozitif tam sayı içermeli.")
    return ks


def _parse_modes(raw: str) -> list[str]:
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in modes if m not in ALL_MODES]
    if not modes or unknown:
        raise argparse.ArgumentTypeError(
            f"Geçersiz mod: {unknown or raw!r}. Seçenekler: {', '.join(ALL_MODES)}"
        )
    return modes


async def run(args: argparse.Namespace) -> int:
    # Ağır/ortam bağımlı modüller yalnızca burada içe aktarılır; böylece metrik fonksiyonları
    # (ve testleri) veritabanı, sentence-transformers ya da DATABASE_URL olmadan çalışır.
    from sqlalchemy import func, select

    from app import config
    from app.database import AsyncSessionLocal, close_db
    from app.models import Document, DocumentChunk, DocumentStatus, User
    from app.services import retrieval_service as rs
    from app.services.embedding_service import embed_texts_async
    from app.services.rerank_service import candidates_needed, rerank_chunks

    try:
        questions = load_questions(Path(args.questions))
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2
    max_k = max(args.ks)

    try:
        async with AsyncSessionLocal() as db:
            user_id = await db.scalar(select(User.id).where(User.email == args.email))
            if user_id is None:
                print(f"HATA: '{args.email}' e-postalı kullanıcı bulunamadı.", file=sys.stderr)
                return 2

            filenames = list(
                await db.scalars(
                    select(Document.filename).where(
                        Document.user_id == user_id,
                        Document.status == DocumentStatus.COMPLETED.value,
                    )
                )
            )
            n_chunks = await db.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    Document.user_id == user_id,
                    Document.status == DocumentStatus.COMPLETED.value,
                )
            )
            await db.rollback()

            missing = sorted(
                {q.doc for q in questions if not any(q.doc.lower() in f.lower() for f in filenames)}
            )
            if missing:
                print(
                    "HATA: soru setindeki şu 'doc' ifadeleri kullanıcının hazır belgelerinde geçmiyor: "
                    f"{missing}\nYüklü belgeler: {filenames or '(yok)'}",
                    file=sys.stderr,
                )
                return 2

            async def finish(chunks: list) -> list:
                await db.rollback()  # uygulamadaki gibi: her aramadan sonra transaction'ı bırak
                return chunks

            async def vector(q: str, k: int) -> list:
                return await finish(await rs.search_similar_chunks(db, user_id=user_id, query=q, top_k=k))

            async def fts(q: str, k: int) -> list:
                [vec] = await embed_texts_async([q])
                return await finish(
                    await rs._fts_search(db, q, vec, user_id=user_id, document_ids=None, limit=k)
                )

            async def hybrid(q: str, k: int) -> list:
                return await finish(await rs.hybrid_search_chunks(db, user_id=user_id, query=q, top_k=k))

            async def hybrid_score(q: str, k: int) -> list:
                chunks = await finish(await rs.hybrid_search_chunks(db, user_id=user_id, query=q, top_k=k))
                return await rerank_chunks(q, chunks, top_k=k, mode="score")

            async def hybrid_cross_encoder(q: str, k: int) -> list:
                wide = candidates_needed(k, "cross_encoder")
                chunks = await finish(
                    await rs.hybrid_search_chunks(db, user_id=user_id, query=q, top_k=wide)
                )
                return await rerank_chunks(q, chunks, top_k=k, mode="cross_encoder")

            def ce_fell_back(chunks: list) -> bool:
                # Cross-encoder çalışırsa her sonucun rerank_score'u dolu olur; boşsa uygulama
                # sessizce skor filtresine düşmüştür (bkz. rerank_service.rerank_chunks).
                return bool(chunks) and any(c.rerank_score is None for c in chunks)

            retrievers: dict[str, tuple[Retriever, Callable[[list], bool] | None]] = {
                "vector": (vector, None),
                "fts": (fts, None),
                "hybrid": (hybrid, None),
                "hybrid_score": (hybrid_score, None),
                "hybrid_cross_encoder": (hybrid_cross_encoder, ce_fell_back),
            }

            results: dict[str, ModeResult] = {}
            for mode in args.modes:
                retrieve, fallback = retrievers[mode]
                print(f"[{mode}] {len(questions)} soru değerlendiriliyor...", file=sys.stderr)
                results[mode] = await evaluate_mode(
                    mode, retrieve, questions, max_k=max_k, is_fallback=fallback
                )

        info = {
            "Documents / chunks in library": f"{len(filenames)} / {n_chunks}",
            "Embedding model": config.EMBEDDING_MODEL,
            "Chunking": f"{config.CHUNK_SIZE} chars, {config.CHUNK_OVERLAP} overlap",
            "Hybrid": (
                f"RRF k={config.HYBRID_RRF_K}, weights vector/FTS="
                f"{config.HYBRID_VECTOR_WEIGHT}/{config.HYBRID_FTS_WEIGHT}, "
                f"candidates per branch ≥ {config.HYBRID_CANDIDATES}"
            ),
            "Min cosine score": f"{config.RETRIEVAL_MIN_SCORE} (FTS-only: {config.HYBRID_FTS_MIN_SCORE})",
            "Score filter": f"cutoff {config.RERANK_RELATIVE_CUTOFF}, keep ≥ {config.RERANK_MIN_KEEP}",
            "Cross-encoder": f"{config.RERANK_MODEL} ({config.RERANK_CANDIDATES} candidates)",
        }
        print(format_markdown(info, questions, results, args.ks))

        if args.json_out:
            payload = {
                "ks": args.ks,
                "info": info,
                "questions": [
                    {"id": q.id, "question": q.question, "doc": q.doc, "pages": list(q.pages), "lang": q.lang}
                    for q in questions
                ],
                "modes": {
                    mode: {
                        "ranks": res.ranks,
                        "top": res.top,
                        "latencies_ms": [round(v, 1) for v in res.latencies_ms],
                        "fallback_count": res.fallback_count,
                    }
                    for mode, res in results.items()
                },
            }
            Path(args.json_out).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Ayrıntılı sonuçlar yazıldı: {args.json_out}", file=sys.stderr)
        return 0
    finally:
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description="Arama kalitesi değerlendirmesi (Hit@k, MRR).")
    parser.add_argument("--email", required=True, help="Belgelerin sahibi kullanıcının e-postası.")
    parser.add_argument("--questions", default="eval/questions.jsonl", help="JSONL soru seti.")
    parser.add_argument("--ks", type=_parse_ks, default=[1, 3, 5], help="Virgülle ayrılmış k değerleri (1,3,5).")
    parser.add_argument(
        "--modes",
        type=_parse_modes,
        default=list(ALL_MODES),
        help=f"Çalıştırılacak modlar. Seçenekler: {', '.join(ALL_MODES)}",
    )
    parser.add_argument("--json-out", help="Soru bazlı ayrıntıları bu JSON dosyasına da yaz.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)  # uygulamanın INFO günlükleri sussun
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
