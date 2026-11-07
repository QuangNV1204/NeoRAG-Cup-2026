from __future__ import annotations

from neorag.data import Document
from neorag.solution import NeoRAGPipeline
from neorag.solution.config import SolutionConfig
from neorag.solution.corpus import (
    build_chunks,
    build_smart_units,
    format_passage_dense_text,
)
from neorag.solution.graph import KnowledgeGraph
from neorag.solution.index import BM25Index
from neorag.solution.pipeline import RAGAnythingPipeline
from neorag.solution.query import QueryAnalyzer, QueryExpander
from neorag.solution.vietnamese import (
    accented_bm25_tokens,
    ascii_bm25_tokens,
    ascii_content_words_raw,
    detect_clubs,
    is_multi_club_question,
    sanitize_answer,
    strip_diacritics,
)


def test_vietnamese_normalization_and_detection():
    assert strip_diacritics("CLB Đa phương tiện") == "CLB Da phuong tien"
    assert detect_clubs("CLB Lập Trình PTIT") == ["CLB Lập Trình PTIT"]
    assert detect_clubs("clb lap trinh ptit") == ["CLB Lập Trình PTIT"]
    assert detect_clubs("MPC tuyển thành viên") == ["CLB Multimedia PTIT"]
    assert detect_clubs("GCC làm game") == ["CLB Nhà Sáng Tạo Game PTIT (GCC)"]
    # "bên code" trong câu hỏi về GCC tuyệt đối không được nhận nhầm sang ProPTIT
    assert detect_clubs("o gcc ben code dong hon hay ben ve dong hon") == [
        "CLB Nhà Sáng Tạo Game PTIT (GCC)"
    ]
    assert is_multi_club_question("o gcc ben code dong hon hay ben ve dong hon") is False
    assert is_multi_club_question("So sánh ProPTIT và GCC") is True
    assert is_multi_club_question("ProPTIT thành lập ngày nào?") is False


def test_tech_tokens_and_sanitize_answer():
    acc = accented_bm25_tokens("Học ngôn ngữ C, C++ và C# tại CLB.")
    asc = ascii_bm25_tokens("hoc ngon ngu c, c++ va c# tai clb")
    assert {"c", "cpp", "csharp"} <= set(acc)
    assert {"c", "cpp", "csharp"} <= set(asc)

    # Bảo vệ abstention: gỡ tiền tố "Không." sai khi ngay sau đó nói tài liệu không có thông tin
    cleaned = sanitize_answer(
        "Không. Tài liệu hiện có không nêu phần mềm Premiere.",
        status="UNANSWERABLE",
    )
    assert cleaned == "Tài liệu hiện có không nêu phần mềm Premiere."

    # Không cắt hỏng cụm "Theo Phòng truyền thống..." hoặc "Theo tài liệu hiện có..."
    preserved = sanitize_answer(
        "Theo Phòng truyền thống của CLB Lập Trình PTIT, Chủ nhiệm là Đàm Bá Quyền."
    )
    assert preserved.startswith("Theo Phòng truyền thống")


def test_chunking_and_smart_units(corpus):
    chunks = build_chunks(corpus)
    assert len(chunks) >= len(corpus)
    assert {c.doc_id for c in chunks} == {d.id for d in corpus}
    for chunk in chunks:
        assert chunk.context.startswith(f"{chunk.club} — {chunk.topic}. ")

    long_doc = Document(
        58,
        "CLB Lập Trình PTIT",
        "Giải thưởng",
        "Giải thưởng của CLB Lập Trình PTIT giai đoạn 2022-2024: "
        "Giải Nhất Huawei ICT 2024; Giải Nữ xuất sắc ICPC 2024; "
        "Giải Ba ICPC PTIT 2024; Giải Nhì AI VietFuture 2023; "
        "Giải Nhì Kalapa Challenge Medical Question Answering 2023; "
        "Giải Nhất ICPC PTIT 2022; Giải Ba Chuyên Tin 2022.",
    )
    units = build_smart_units([*corpus, long_doc])
    long_units = [u for u in units if u.doc_id == 58]
    assert len(long_units) >= 2
    assert all("giai đoạn 2022-2024" in u.text for u in long_units)
    assert format_passage_dense_text(corpus[0]).startswith("CLB Lập Trình PTIT — Giới thiệu:")


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
        prompt = str(messages[-1]["content"])
        if "tạo tối đa 3 truy vấn" in prompt:
            return "ProPTIT thành lập năm nào\nMultimedia thành lập năm nào\nGCC thành lập năm nào"
        if "SCOPE:" in prompt and "NORMALIZED:" in prompt:
            if "câu hỏi gốc: phí" in prompt.lower():
                return (
                    "SCOPE: SINGLE\n"
                    "NORMALIZED: Phí hằng tháng của CLB Multimedia PTIT là bao nhiêu?\n"
                    "Q1: phí quỹ CLB Multimedia PTIT\n"
                    "Q2: CLB Multimedia PTIT thu phí"
                )
            return (
                "SCOPE: SINGLE\n"
                "NORMALIZED: CLB Lập Trình PTIT được thành lập ngày nào?\n"
                "Q1: CLB Lập Trình PTIT thành lập ngày nào\n"
                "Q2: giới thiệu CLB Lập Trình PTIT 09/10/2011"
            )
        if "CORE_IDS:" in prompt and "EXTRA_IDS:" in prompt:
            if "câu hỏi gốc: phí" in prompt.lower():
                return "STATUS: UNANSWERABLE\nCORE_IDS: NONE\nEXTRA_IDS: 2"
            return "STATUS: ANSWERABLE\nCORE_IDS: 1\nEXTRA_IDS: 2"
        if "KIỂM TRA:" in prompt and "TRẢ LỜI:" in prompt:
            if "Trạng thái dữ liệu thẩm định: UNANSWERABLE" in prompt:
                return (
                    "KIỂM TRA: Tài liệu không có thông tin về phí của Multimedia.\n"
                    "TRẢ LỜI: Không. Tài liệu hiện có không có thông tin về phí của CLB Multimedia PTIT."
                )
            return (
                "KIỂM TRA: Đoạn 1 ghi thành lập ngày 09/10/2011.\n"
                "TRẢ LỜI: CLB Lập Trình PTIT được thành lập ngày 09/10/2011."
            )
        if "BẮT BUỘC TRẢ VỀ ĐÚNG 2 PHẦN" in prompt or "NHIỆM VỤ:" in str(messages):
            return "ID: 1\nTRẢ LỜI: CLB Lập Trình PTIT được thành lập ngày 09/10/2011."
        return "CLB Lập Trình PTIT được thành lập ngày 09/10/2011."


def test_neorag_pipeline_offline_end_to_end(corpus, tmp_path):
    from conftest import FakeEmbedder
    from neorag.pipelines.base import load_pipeline

    config = SolutionConfig(cache_dir=tmp_path / "cache_neorag")
    pipeline = load_pipeline(
        "neorag.solution:NeoRAGPipeline",
        config=config,
        llm=_OfflineLLM(),
        embedder=FakeEmbedder(),
    )
    assert isinstance(pipeline, NeoRAGPipeline)
    pipeline.prepare()
    pipeline.index(corpus)

    out_ok = pipeline.run("clb lt ptit thanh lap ngay nao?")
    assert out_ok.answer == "CLB Lập Trình PTIT được thành lập ngày 09/10/2011."
    assert out_ok.retrieved_ids[0] == 1
    assert 1 <= len(out_ok.retrieved_ids) <= 10
    assert pipeline.last_debug["status"] == "ANSWERABLE"

    out_unans = pipeline.run("Phí hằng tháng của CLB Multimedia PTIT là bao nhiêu?")
    assert out_unans.answer == "Tài liệu hiện có không có thông tin về phí của CLB Multimedia PTIT."
    assert not out_unans.answer.startswith("Không")
    assert 1 <= len(out_unans.retrieved_ids) <= 10
    assert pipeline.last_debug["status"] == "UNANSWERABLE"


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

    doc_map = {d.id: d for d in corpus}
    covered = pipeline._ensure_club_coverage(
        [1, 2],
        ["CLB Lập Trình PTIT", "CLB Multimedia PTIT", "CLB Nhà Sáng Tạo Game PTIT (GCC)"],
    )
    covered_clubs = [doc_map[i].club for i in covered[:3]]
    assert "CLB Lập Trình PTIT" in covered_clubs
    assert "CLB Multimedia PTIT" in covered_clubs
    assert "CLB Nhà Sáng Tạo Game PTIT (GCC)" in covered_clubs
