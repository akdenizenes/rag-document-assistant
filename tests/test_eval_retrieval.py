"""scripts/eval_retrieval.py içindeki saf mantığın (soru yükleme, Hit@k, MRR, rapor) testleri.

Veritabanı, sentence-transformers ya da ağ gerektirmez.
"""

import asyncio
from dataclasses import dataclass

import pytest

from scripts import eval_retrieval as ev


@dataclass
class FakeChunk:
    document_name: str
    page_number: int | None
    score: float = 0.5
    rerank_score: float | None = None


def q(doc="Software Products", pages=(13,), qid="q1", lang="tr", text="moore vizyonu nedir"):
    return ev.EvalQuestion(id=qid, question=text, doc=doc, pages=tuple(pages), lang=lang)


# --------------------------------------------------------------------------- #
# load_questions
# --------------------------------------------------------------------------- #
def write(tmp_path, content):
    path = tmp_path / "q.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_questions_skips_comments_and_blank_lines(tmp_path):
    path = write(
        tmp_path,
        '# comment\n\n{"question": "a?", "doc": "X", "pages": [1, 2], "lang": "tr"}\n'
        '{"id": "custom", "question": "b?", "doc": "Y", "pages": [3]}\n',
    )
    loaded = ev.load_questions(path)
    assert [x.id for x in loaded] == ["q01", "custom"]
    assert loaded[0].pages == (1, 2) and loaded[0].lang == "tr"
    assert loaded[1].lang == "-"


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("{not json}", "geçersiz JSON"),
        ('["a"]', "JSON nesnesi"),
        ('{"question": "", "doc": "X", "pages": [1]}', "'question'"),
        ('{"question": "a", "doc": "", "pages": [1]}', "'doc'"),
        ('{"question": "a", "doc": "X", "pages": []}', "'pages'"),
        ('{"question": "a", "doc": "X", "pages": [0]}', "'pages'"),
        ('{"question": "a", "doc": "X", "pages": ["1"]}', "'pages'"),
        ('{"question": "a", "doc": "X", "pages": [true]}', "'pages'"),
    ],
)
def test_load_questions_rejects_bad_rows_with_line_number(tmp_path, line, fragment):
    path = write(tmp_path, line + "\n")
    with pytest.raises(ValueError) as err:
        ev.load_questions(path)
    assert fragment in str(err.value) and ":1:" in str(err.value)


def test_load_questions_rejects_duplicate_ids_and_empty_files(tmp_path):
    dup = write(
        tmp_path,
        '{"id": "a", "question": "x", "doc": "D", "pages": [1]}\n'
        '{"id": "a", "question": "y", "doc": "D", "pages": [2]}\n',
    )
    with pytest.raises(ValueError, match="tekrar"):
        ev.load_questions(dup)
    with pytest.raises(ValueError, match="hiç soru"):
        ev.load_questions(write(tmp_path, "# only comments\n"))


def test_shipped_example_file_is_valid():
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "eval" / "questions.example.jsonl"
    loaded = ev.load_questions(example)
    assert loaded and loaded[0].doc == "Software Products" and loaded[0].pages == (13,)


# --------------------------------------------------------------------------- #
# Alaka ve sıra
# --------------------------------------------------------------------------- #
def test_is_relevant_matches_doc_substring_case_insensitively_and_page():
    question = q()
    assert ev.is_relevant(FakeChunk("1. Software Products.pdf", 13), question)
    assert ev.is_relevant(FakeChunk("software products.PDF", 13), question)
    assert not ev.is_relevant(FakeChunk("1. Software Products.pdf", 14), question)
    assert not ev.is_relevant(FakeChunk("Agile Software Engineering.pdf", 13), question)
    assert not ev.is_relevant(FakeChunk("1. Software Products.pdf", None), question)


def test_first_hit_rank_is_one_based_and_respects_max_k():
    question = q()
    hit = FakeChunk("Software Products.pdf", 13)
    miss = FakeChunk("Software Products.pdf", 2)
    assert ev.first_hit_rank([hit, miss], question, 5) == 1
    assert ev.first_hit_rank([miss, miss, hit], question, 5) == 3
    assert ev.first_hit_rank([miss, miss, hit], question, 2) is None  # k dışında kaldı
    assert ev.first_hit_rank([], question, 5) is None


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def test_summarize_hit_at_k_and_mrr():
    ranks = [1, 3, None, 2]
    s = ev.summarize(ranks, [1, 3, 5])
    assert s["n"] == 4
    assert s["hit"] == {1: 0.25, 3: 0.75, 5: 0.75}
    assert s["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


def test_summarize_empty_does_not_divide_by_zero():
    s = ev.summarize([], [1, 3])
    assert s == {"n": 0, "hit": {1: 0.0, 3: 0.0}, "mrr": 0.0}


# --------------------------------------------------------------------------- #
# evaluate_mode
# --------------------------------------------------------------------------- #
def test_evaluate_mode_warms_up_once_and_records_ranks():
    questions = [q(qid="a", pages=(1,)), q(qid="b", pages=(2,))]
    calls: list[str] = []

    async def retrieve(text: str, k: int):
        calls.append(text)
        # "a" -> doğru sayfa 1. sırada; "b" -> doğru sayfa (2) hiç dönmüyor
        return [FakeChunk("Software Products.pdf", 1), FakeChunk("Software Products.pdf", 9)]

    result = asyncio.run(ev.evaluate_mode("hybrid", retrieve, questions, max_k=5))
    assert len(calls) == 3  # 1 ısınma + 2 soru
    assert result.ranks == {"a": 1, "b": None}
    assert len(result.latencies_ms) == 2 and not result.invalid
    assert result.top["a"][0] == {"doc": "Software Products.pdf", "page": 1, "score": 0.5}


def test_cross_encoder_fallback_invalidates_the_row():
    questions = [q(qid="a"), q(qid="b")]

    async def retrieve(text: str, k: int):
        return [FakeChunk("Software Products.pdf", 13, rerank_score=None)]  # skora düşülmüş

    def fell_back(chunks):
        return bool(chunks) and any(c.rerank_score is None for c in chunks)

    result = asyncio.run(
        ev.evaluate_mode("hybrid_cross_encoder", retrieve, questions, max_k=3, is_fallback=fell_back)
    )
    assert result.fallback_count == 2 and result.invalid

    # Aynı düzenek başka bir modda geçersiz sayılmamalı
    other = ev.ModeResult(mode="hybrid", fallback_count=2)
    assert not other.invalid


# --------------------------------------------------------------------------- #
# format_markdown
# --------------------------------------------------------------------------- #
def make_results():
    good = ev.ModeResult(mode="hybrid", ranks={"a": 1, "b": None}, latencies_ms=[10.0, 30.0])
    vec = ev.ModeResult(mode="vector", ranks={"a": 2, "b": 1}, latencies_ms=[8.0, 12.0])
    return {"vector": vec, "hybrid": good}


def test_format_markdown_table_and_misses():
    questions = [q(qid="a", lang="tr"), q(qid="b", lang="en", text="what is the vision template")]
    md = ev.format_markdown({"Embedding model": "m"}, questions, make_results(), [1, 3])

    assert "| Retrieval mode | Hit@1 | Hit@3 | MRR@3 | Median latency |" in md
    assert "| Hybrid (vector + FTS, RRF) | 50.0% | 50.0% | 0.500 | 20 ms |" in md
    assert "| Vector only (pgvector) | 50.0% | 100.0% | 0.750 | 10 ms |" in md
    assert "### Language `tr` (1 questions)" in md and "### Language `en` (1 questions)" in md
    assert "### Misses for `hybrid`" in md and "`b` (en): what is the vision template" in md
    assert "- Embedding model: m" in md


def test_format_markdown_withholds_numbers_for_invalid_cross_encoder():
    questions = [q(qid="a")]
    results = {
        "hybrid_cross_encoder": ev.ModeResult(
            mode="hybrid_cross_encoder", ranks={"a": 1}, latencies_ms=[5.0], fallback_count=1
        )
    }
    md = ev.format_markdown({}, questions, results, [1])
    assert "| Hybrid + cross-encoder rerank ⚠ | n/a | n/a | n/a |" in md
    assert "numbers are withheld" in md
    assert "100.0%" not in md


def test_parse_ks_and_modes_validation():
    import argparse

    assert ev._parse_ks("5,1,3,3") == [1, 3, 5]
    assert ev._parse_modes("vector, hybrid") == ["vector", "hybrid"]
    for bad in ("", "0", "a,b", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            ev._parse_ks(bad)
    with pytest.raises(argparse.ArgumentTypeError):
        ev._parse_modes("vector,nope")
