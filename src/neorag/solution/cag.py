"""Cache-Augmented Generation (CAG): đưa TOÀN BỘ corpus vào prompt có prompt-caching.

Corpus chỉ có 118 passage (~29K token), nên thay vì chỉ cho LLM xem top-k passage do
retriever chọn, pipeline này cho `claude-haiku-5-5` đọc toàn bộ corpus trong system prompt
(được đánh dấu `cache_control` để gateway/Anthropic cache, lần gọi sau chỉ tốn ~10% giá
input). LLM vừa đóng vai retriever (chọn và xếp hạng id passage cần dùng) vừa trả lời.

Luồng:
    1. Hybrid retriever (dense bge-m3 + BM25, không gọi LLM) xếp hạng toàn bộ passage
       — dùng làm gợi ý cho LLM và để điền phần đuôi của `retrieved_ids`.
    2. Lượt "chọn + trả lời": LLM đọc toàn bộ corpus + few-shot từ tập train, trả về
       danh sách id cần thiết (xếp theo mức quan trọng) và câu trả lời ngắn.
    3. (tùy chọn) Lượt "viết lại": chỉ đưa các passage đã chọn, LLM kiểm tra và viết lại
       câu trả lời bám sát nguyên văn passage.
    4. `retrieved_ids` = id do LLM chọn → bổ sung từ hybrid ranking cho đủ 10.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

import numpy as np

from neorag.data import DATA_DIR, Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.solution.config import SolutionConfig, _env_bool, _env_int
from neorag.solution.corpus import build_chunks
from neorag.solution.graph import KnowledgeGraph
from neorag.solution.index import BM25Index, DenseIndex
from neorag.solution.query import QueryAnalysis, QueryAnalyzer, QueryExpander, SearchQuery
from neorag.solution.retrieve import HybridRetriever
from neorag.solution.vietnamese import ascii_content_words_raw, clean_answer, sanitize_answer

FALLBACK_ANSWER = "Tài liệu hiện có không cung cấp thông tin để trả lời câu hỏi này."

INSTRUCTIONS = """Bạn là trợ lý hỏi đáp chính xác về cộng đồng S2B tại Học viện Công nghệ Bưu chính Viễn thông (PTIT), gồm CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC, CLB đa phương tiện) và CLB Nhà Sáng Tạo Game PTIT (GCC, CLB game). Bạn được cung cấp TOÀN BỘ kho tài liệu bên dưới, mỗi đoạn có dạng [id] CLB — chủ đề, rồi nội dung. Người hỏi thường là sinh viên, có thể viết không dấu, viết tắt (clb, tv, hv, ko, dc, e, a...) hoặc dùng cách gọi khác cho CLB (clb lập trình, clb đa phương tiện/mm/multimedia, clb game/làm game...).

NHIỆM VỤ: với mỗi câu hỏi, trả về đúng hai dòng theo định dạng:
ID: <danh sách id đoạn văn cần dùng, cách nhau bởi dấu phẩy, đoạn quan trọng nhất trước>
TRẢ LỜI: <câu trả lời>

QUY TẮC CHỌN ID:
- Chọn MỌI đoạn chứa dữ kiện dùng trong câu trả lời và chỉ những đoạn đó (thường 1-3 đoạn, tối đa 6). Đoạn trả lời trực tiếp ý chính đứng đầu.
- Câu hỏi nhiều ý, nhiều CLB hoặc so sánh (ví dụ lâu đời nhất, trẻ nhất, so sánh cơ cấu, hoạt động): BẮT BUỘC chọn đủ ít nhất 1 đoạn văn cho MỖI CLB được nhắc tới (đủ cả 3 CLB nếu hỏi cả 3 CLB hoặc hỏi CLB nào lâu đời nhất/trẻ nhất).
- Chú ý đúng CLB được hỏi; không lấy đoạn của CLB khác nếu câu hỏi không nhắc tới.
- Nếu tài liệu không có thông tin trả lời, vẫn ghi id các đoạn liên quan gần nhất mà câu trả lời nhắc tới (nếu hoàn toàn không có đoạn nào liên quan thì để trống).

QUY TẮC TRẢ LỜI:
- Chỉ dùng thông tin trong tài liệu. Tiếng Việt có dấu, viết thành đoạn văn ngắn, không gạch đầu dòng, không markdown, không ghi id hay trích dẫn [id] trong câu trả lời.
- Đi thẳng vào câu trả lời, tuyệt đối không dùng cụm 'Dựa trên tài liệu được cung cấp,', không mở bài, không lặp lại câu hỏi, không kết luận, không khuyên nhủ, không thêm nhận xét suy đoán.
- Với câu hỏi có/không mà tài liệu thực sự có thông tin khẳng định hoặc phủ định thì mở đầu bằng "Có." hoặc "Không."; nếu tài liệu không nêu thông tin được hỏi thì TUYỆT ĐỐI KHÔNG mở đầu bằng "Có." hay "Không.", mà bắt đầu trực tiếp bằng "Tài liệu hiện có không có thông tin về ..." hoặc "Tài liệu không nêu ...".
- Câu hỏi một ý: gói gọn trong 1-2 câu. Câu hỏi nhiều ý/so sánh: tối đa 3-4 câu, trả lời đủ từng ý.
- Gọi đúng tên CLB đầy đủ (ví dụ "CLB Lập Trình PTIT", "CLB Multimedia PTIT", "CLB Nhà Sáng Tạo Game PTIT (GCC)").
- Chép NGUYÊN VĂN cụm từ của tài liệu cho tên riêng, ngày tháng, số liệu, tên ban, tên giải, tên sự kiện, khẩu hiệu; giữ nguyên dấu phẩy và từ nối bên trong cụm (ví dụ 'contest ngắn, dài', 'cả CLB Lập Trình PTIT và CLB IT PTIT', 'chỉ áp dụng cho năm 2026'); không chèn thêm từ hoặc mở ngoặc (như viết thêm ProPTIT) vào giữa cụm từ.
- Nêu đủ tất cả các mục cùng loại mà câu hỏi hỏi (nhiều giải, nhiều mốc, nhiều vòng, nhiều ban, nhiều kênh).
- Với câu hỏi về số năm thành lập/hoạt động: nêu rõ ngày/tháng/năm thành lập VÀ số năm (ví dụ "14 năm") tính đến mốc thời gian được hỏi.
- Với câu hỏi về nhân sự, ban/team phụ trách: nêu rõ tên ban/vị trí chính xác kèm họ tên và nhiệm kỳ (nếu tài liệu có).
- Giữ mốc thời gian/bối cảnh của thông tin khi cần (ví dụ "Theo bài tuyển thành viên năm 2026", "nhiệm kỳ 2025-2026", "đợt tuyển Gen 14"); không trình bày thông tin của một đợt/nhiệm kỳ như quy định cố định.
- Nếu tài liệu không có hoặc thiếu thông tin được hỏi: dùng cấu trúc "Tài liệu hiện có không có thông tin về ..." hoặc "Tài liệu không ghi ...", tuyệt đối không suy đoán hay lấy thông tin của CLB/nhiệm kỳ/đợt khác để trả lời thay. Nếu câu hỏi hỏi về thông tin mà tài liệu chỉ có thông tin tương tự ở CLB/nhiệm kỳ khác, có thể nêu ngắn gọn thông tin đó nhưng phải khẳng định rõ tài liệu không ghi thông tin được hỏi.
- Nếu chỉ có một phần thông tin: trả lời phần có và nói rõ phần nào tài liệu không ghi/không có.
- Nếu các đoạn ghi khác nhau về cùng một dữ kiện: nêu cả hai cách ghi kèm nguồn/thời điểm."""

REWRITE_PROMPT = """CÁC ĐOẠN TÀI LIỆU:
{context}

CÂU HỎI: {query}

CÂU TRẢ LỜI NHÁP:
{draft}

Kiểm tra câu trả lời nháp với các đoạn tài liệu trên và viết lại câu trả lời cuối cùng:
- Giữ đủ mọi dữ kiện trả lời câu hỏi; bỏ mọi chi tiết không được các đoạn trên hỗ trợ và chi tiết ngoài phạm vi câu hỏi.
- Chép nguyên văn cụm từ của tài liệu cho tên riêng, số liệu, ngày tháng, tên ban/giải/sự kiện.
- Câu hỏi có/không mở đầu bằng "Có." hoặc "Không.". Câu hỏi một ý 1-2 câu; nhiều ý tối đa 4 câu.
- Nếu nháp nói tài liệu không có thông tin thì giữ nguyên ý đó.
- Đoạn văn, không markdown, không ghi id. Chỉ trả về câu trả lời cuối cùng."""

_ID_LINE = re.compile(
    r"^\s*\**\s*(?:ID|DOCS?|PASSAGES?|ĐOẠN(?:\s*VĂN)?)\s*\**\s*[:：]\s*(.*)$",
    re.I | re.M,
)
_ANSWER_LINE = re.compile(
    r"^\s*\**\s*(?:TRẢ\s*LỜI|CÂU\s*TRẢ\s*LỜI|ĐÁP\s*ÁN|ANSWER)\s*\**\s*[:：]\s*",
    re.I | re.M,
)
_NUMBERS = re.compile(r"\d+")

# Few-shot cố định từ tập train (bao quát trọn vẹn mọi dạng câu hỏi của cuộc thi).
FEWSHOT_IDS = (
    "train-0001",  # trực tiếp: súc tích, đi thẳng vào đáp án
    "train-0002",  # không dấu/viết tắt: mở đầu bằng Có., giữ nguyên handle/link
    "train-0005",  # diễn đạt khác, câu hỏi có/không: mở đầu bằng Không.
    "train-0003",  # nhiều đoạn: lộ trình đào tạo, giữ đúng số tháng, tên cuộc thi
    "train-0017",  # so sánh 3 CLB: ngày thành lập, tuổi CLB
    "train-0024",  # thời gian: số năm thành lập tính đến mốc thời gian
    "train-0027",  # thiếu dữ liệu: trả lời phần có, nói rõ phần không có
    "train-0019",  # không có đáp án hoàn toàn: ID để trống, từ chối rõ ràng
    "train-0044",  # so sánh 3 CLB toàn diện: chọn đủ ID cả 3 CLB
    "train-0049",  # không có đáp án nhưng có đoạn liên quan: nêu đoạn liên quan và từ chối
)


def _format_doc(doc: Document) -> str:
    return f"[{doc.id}] {doc.club} — {doc.topic}\n{doc.text}"


def load_fewshot(ids: tuple[str, ...], limit: int) -> list[dict]:
    path = DATA_DIR / "train" / "questions.jsonl"
    if limit <= 0 or not path.exists():
        return []
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return [rows[i] for i in ids if i in rows][:limit]


def parse_response(raw: str) -> tuple[list[int], str]:
    ids: list[int] = []
    if match := _ID_LINE.search(raw):
        ids = [int(x) for x in _NUMBERS.findall(match.group(1))]
    answer = raw
    if match := _ANSWER_LINE.search(raw):
        answer = raw[match.end() :]
    elif ids:
        # Không có nhãn TRẢ LỜI: bỏ dòng ID, phần còn lại là câu trả lời.
        answer = _ID_LINE.sub("", raw)
    ans = sanitize_answer(answer)
    return ids, ans


class CAGPipeline(RAGPipeline):
    def __init__(
        self,
        config: SolutionConfig | None = None,
        llm: LLMClient | None = None,
        embedder: Any | None = None,
        **_: Any,
    ):
        base = config or SolutionConfig()
        # Hybrid ranking chỉ làm gợi ý/điền đuôi: không gọi LLM ở bước này.
        self.config = replace(base, use_llm_rerank=False, use_graph_context=False)
        self.config.ensure_dirs()
        self.llm = llm or LLMClient(cache_path=self.config.cache_dir / "llm.sqlite")
        self.embedder = embedder
        self.llm_model = self.config.llm_model
        self.embedding_model = self.config.embedding_model

        self.fewshot_count = _env_int("NEORAG_CAG_FEWSHOT", len(FEWSHOT_IDS))
        self.hint_count = _env_int("NEORAG_CAG_HINTS", 10)
        self.max_llm_ids = _env_int("NEORAG_CAG_MAX_IDS", 6)
        self.use_rewrite = _env_bool("NEORAG_CAG_REWRITE", False)

        self.documents: list[Document] = []
        self.retriever: HybridRetriever | None = None
        self.analyzer = QueryAnalyzer()
        self.expander = QueryExpander(llm=None, config=self.config)
        self.system_prompt = ""
        self.last_debug: dict[str, Any] = {}
        self.debug_by_query: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ index
    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embedder is not None:
            vectors = self.embedder.embed(texts)
        else:
            vectors = self.llm.embed(
                texts,
                model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )
        return np.asarray(vectors, dtype=np.float32)

    def index(self, corpus: list[Document]) -> None:
        self.documents = list(corpus)
        if not self.documents:
            raise ValueError("Corpus rỗng")
        self.doc_by_id = {doc.id: doc for doc in self.documents}

        chunks = build_chunks(self.documents)
        dense = DenseIndex(
            embed=self._embed,
            cache_dir=self.config.cache_dir,
            embedding_model=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        ).fit([chunk.context for chunk in chunks], name="chunks")
        bm25 = BM25Index(
            [f"{doc.club} {doc.topic} {doc.text}" for doc in self.documents],
            tokenizer=ascii_content_words_raw,
        )
        self.retriever = HybridRetriever(
            documents=self.documents,
            chunks=chunks,
            dense=dense,
            bm25=bm25,
            graph=KnowledgeGraph(self.documents, chunks),
            config=self.config,
            llm=None,
        )
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        parts = [INSTRUCTIONS, "KHO TÀI LIỆU:", "\n\n".join(_format_doc(d) for d in self.documents)]
        examples = load_fewshot(FEWSHOT_IDS, self.fewshot_count)
        if examples:
            rows = []
            for ex in examples:
                ids = ", ".join(str(i) for i in ex["relevant_ids"])
                rows.append(f"CÂU HỎI: {ex['query']}\nID: {ids}\nTRẢ LỜI: {ex['answer']}")
            parts.append("VÍ DỤ CÁCH TRẢ LỜI:\n\n" + "\n\n".join(rows))
        return "\n\n".join(parts)

    # -------------------------------------------------------------------- run
    def _system_message(self) -> dict:
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

    def _hybrid_order(self, query: str) -> tuple[list[int], QueryAnalysis]:
        analysis = self.analyzer.analyze(query)
        queries = [SearchQuery(query, 1.0, "original")]
        for text in self.expander._deterministic(query, analysis):
            queries.append(SearchQuery(text, self.config.deterministic_expansion_weight, "club"))
        result = self.retriever.retrieve(analysis, queries)
        return [doc.id for doc in result.documents], analysis

    def _select_and_answer(
        self, query: str, hints: list[int], analysis: QueryAnalysis
    ) -> tuple[list[int], str, str]:
        user = f"CÂU HỎI: {query}"
        if hints and self.hint_count > 0:
            hint_text = ", ".join(str(i) for i in hints[: self.hint_count])
            user += f"\n(Gợi ý tìm kiếm sơ bộ: {hint_text})"
        user += (
            "\n\nBẮT BUỘC TRẢ VỀ ĐÚNG 2 PHẦN THEO ĐỊNH DẠNG:\n"
            "ID: <danh sách id đoạn văn trực tiếp chứa dữ kiện để trả lời, cách nhau dấu phẩy>\n"
            "TRẢ LỜI: <câu trả lời súc tích, trực tiếp>"
        )
        raw = self.llm.chat(
            [self._system_message(), {"role": "user", "content": user}],
            model=self.config.llm_model,
        )
        ids, answer = parse_response(raw)
        valid: list[int] = []
        for doc_id in ids:
            if doc_id in self.doc_by_id and doc_id not in valid:
                valid.append(doc_id)
        return valid[: self.max_llm_ids], answer, raw

    def _rewrite(self, query: str, draft: str, ids: list[int]) -> str:
        if not ids:
            return draft
        context = "\n\n".join(_format_doc(self.doc_by_id[i]) for i in ids)
        prompt = REWRITE_PROMPT.format(context=context, query=query, draft=draft)
        try:
            out = clean_answer(
                self.llm.chat([{"role": "user", "content": prompt}], model=self.config.llm_model)
            )
        except Exception:  # pragma: no cover - phụ thuộc gateway
            return draft
        return out or draft

    def _ensure_club_coverage(
        self,
        doc_ids: list[int],
        clubs: list[str],
    ) -> list[int]:
        if len(clubs) < 2:
            return doc_ids

        club_first: dict[str, int] = {}
        for doc_id in doc_ids:
            doc = self.doc_by_id.get(doc_id)
            if doc and doc.club in clubs and doc.club not in club_first:
                club_first[doc.club] = doc_id
                if len(club_first) == len(clubs):
                    break

        for club in clubs:
            if club not in club_first:
                for doc in self.documents:
                    if doc.club == club:
                        club_first[club] = doc.id
                        break

        head = [club_first[c] for c in clubs if c in club_first]
        head_set = set(head)
        tail = [doc_id for doc_id in doc_ids if doc_id not in head_set]
        return head + tail

    def run(self, query: str) -> RAGOutput:
        if self.retriever is None:
            raise RuntimeError("Pipeline chưa được index. Gọi index(corpus) trước.")

        hybrid_ids, analysis = self._hybrid_order(query)
        raw = ""
        try:
            llm_ids, answer, raw = self._select_and_answer(query, hybrid_ids, analysis)
        except Exception as exc:  # pragma: no cover - phụ thuộc gateway
            llm_ids, answer = [], ""
            raw = f"ERROR {type(exc).__name__}: {exc}"
        if self.use_rewrite and answer:
            answer = self._rewrite(query, answer, llm_ids)
        if not answer:
            answer = FALLBACK_ANSWER

        retrieved: list[int] = []
        for doc_id in [*llm_ids, *hybrid_ids]:
            if doc_id not in retrieved:
                retrieved.append(doc_id)

        retrieved = retrieved[: self.config.max_retrieved] or [self.documents[0].id]

        self.last_debug = {
            "query": query,
            "llm_ids": llm_ids,
            "hybrid_ids": hybrid_ids,
            "raw": raw,
            "retrieved_ids": retrieved,
            "answer": answer,
            "clubs": analysis.clubs,
        }
        self.debug_by_query[query] = self.last_debug
        return RAGOutput(answer=clean_answer(answer, max_chars=2000), retrieved_ids=retrieved)


__all__ = ["CAGPipeline", "FEWSHOT_IDS", "parse_response"]
