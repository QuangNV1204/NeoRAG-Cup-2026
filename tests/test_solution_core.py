from __future__ import annotations

from neorag.data import Document
from neorag.solution.config import SolutionConfig
from neorag.solution.corpus import build_chunks
from neorag.solution.graph import KnowledgeGraph
from neorag.solution.index import BM25Index
from neorag.solution.pipeline import RAGAnythingPipeline
from neorag.solution.query import QueryAnalyzer, QueryExpander
from neorag.solution.vietnamese import (
    ascii_content_words_raw,
    detect_clubs,
    is_multi_club_question,
    strip_diacritics,
)


def test_vietnamese_normalization_and_detection():
    assert strip_diacritics("CLB Đa phương tiện") == "CLB Da phuong tien"
    assert detect_clubs("CLB Lập Trình PTIT") == ["CLB Lập Trình PTIT"]
    assert detect_clubs("clb lap trinh ptit") == ["CLB Lập Trình PTIT"]
    assert detect_clubs("MPC tuyển thành viên") == ["CLB Multimedia PTIT"]
    assert detect_clubs("GCC làm game") == ["CLB Nhà Sáng Tạo Game PTIT (GCC)"]
    assert is_multi_club_question("So sánh ProPTIT và GCC") is True
    assert is_multi_club_question("ProPTIT thành lập ngày nào?") is False


def test_chunking_uses_sentence_windows_and_parent_ids(corpus):
    chunks = build_chunks(corpus)
    assert len(chunks) >= len(corpus)
    assert {c.doc_id for c in chunks} == {d.id for d in corpus}
    for chunk in chunks:
        assert chunk.context.startswith(f"{chunk.club} — {chunk.topic}. ")
        assert chunk.doc_id in {d.id for d in corpus}


def test_bm25_ascii_handles_accented_corpus():
    docs = [
        "CLB Lập Trình PTIT được thành lập ngày 09/10/2011.",
        "CLB Multimedia PTIT được thành lập ngày 10/11/2012.",
        "GCC ra đời năm 2023.",
    ]
    index = BM25Index(docs, tokenizer=ascii_content_words_raw)
    scores = index.scores("clb lap trinh ptit thanh lap ngay nao")
    assert int(scores.argmax()) == 0
    assert scores[0] > scores[1]


def test_knowledge_graph_links_related_docs():
    docs = [
        Document(1, "CLB Lập Trình PTIT", "Lịch sử", "ProPTIT thành lập năm 2011 tại Học viện."),
        Document(2, "CLB Lập Trình PTIT", "Giải thưởng", "ProPTIT đạt giải ICPC Regional."),
        Document(3, "CLB Multimedia PTIT", "Liên hệ", "Multimedia có fanpage riêng."),
    ]
    graph = KnowledgeGraph(docs)
    related = [doc_id for doc_id, _ in graph.related([1], limit=3)]
    assert 2 in related
    assert 1 not in related
    assert graph.summary()["documents"] == 3


def test_query_analyzer_and_expander_are_deterministic_without_llm():
    config = SolutionConfig(use_query_expansion=False)
    analysis = QueryAnalyzer().analyze("So sánh ProPTIT và GCC")
    assert analysis.is_multi is True
    assert set(analysis.main_clubs) == {
        "CLB Lập Trình PTIT",
        "CLB Nhà Sáng Tạo Game PTIT (GCC)",
    }
    queries = QueryExpander(None, config).expand(analysis.query, analysis)
    assert queries[0].text == analysis.query
    assert any(item.kind == "club" for item in queries)


class _OfflineLLM:
    """Trả lời cố định; đủ để kiểm tra luồng pipeline mà không gọi gateway."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, model, temperature=None):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "tạo tối đa 3 truy vấn" in prompt:
            return "ProPTIT thành lập năm nào\nMultimedia thành lập năm nào\nGCC thành lập năm nào"
        if "BẮT BUỘC TRẢ VỀ ĐÚNG 2 PHẦN" in prompt or "NHIỆM VỤ:" in str(messages):
            return "ID: 1\nTRẢ LỜI: CLB Lập Trình PTIT được thành lập ngày 09/10/2011."
        return "CLB Lập Trình PTIT được thành lập ngày 09/10/2011."


def test_pipeline_offline_end_to_end(corpus, tmp_path):
    config = SolutionConfig(
        cache_dir=tmp_path / "cache",
        use_query_expansion=True,
        use_verification=True,
        context_docs=2,
    )
    from conftest import FakeEmbedder

    pipeline = RAGAnythingPipeline(
        config=config,
        llm=_OfflineLLM(),
        embedder=FakeEmbedder(),
    )
    pipeline.prepare()
    pipeline.index(corpus)
    output = pipeline.run("CLB Lập Trình PTIT thành lập ngày nào?")
    assert output.answer
    assert 1 <= len(output.retrieved_ids) <= 10
    assert output.retrieved_ids[0] == 1
    assert pipeline.last_debug["search_queries"]
    assert pipeline.last_debug["retrieved_ids"] == output.retrieved_ids


def test_cag_pipeline_offline_end_to_end(corpus, tmp_path):
    from conftest import FakeEmbedder
    from neorag.solution.cag import CAGPipeline

    config = SolutionConfig(cache_dir=tmp_path / "cache_cag")
    pipeline = CAGPipeline(
        config=config,
        llm=_OfflineLLM(),
        embedder=FakeEmbedder(),
    )
    pipeline.prepare()
    pipeline.index(corpus)
    output = pipeline.run("CLB Lập Trình PTIT thành lập ngày nào?")
    assert output.answer == "CLB Lập Trình PTIT được thành lập ngày 09/10/2011."
    assert output.retrieved_ids[0] == 1
    assert len(output.retrieved_ids) <= 10

    # Kiểm tra coverage câu hỏi so sánh 3 CLB
    doc_map = {d.id: d for d in corpus}
    # Giả sử ban đầu retrieved chỉ có docs của ProPTIT (id 1, 2)
    # _ensure_club_coverage phải kéo đại diện của MPC và GCC lên
    covered = pipeline._ensure_club_coverage(
        [1, 2],
        ["CLB Lập Trình PTIT", "CLB Multimedia PTIT", "CLB Nhà Sáng Tạo Game PTIT (GCC)"],
    )
    covered_clubs = [doc_map[i].club for i in covered[:3]]
    assert "CLB Lập Trình PTIT" in covered_clubs
    assert "CLB Multimedia PTIT" in covered_clubs
    assert "CLB Nhà Sáng Tạo Game PTIT (GCC)" in covered_clubs

