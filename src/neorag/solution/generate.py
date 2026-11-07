from __future__ import annotations

from dataclasses import dataclass, field

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.solution.config import SolutionConfig
from neorag.solution.query import QueryAnalysis
from neorag.solution.vietnamese import clean_answer

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp chính xác về cộng đồng S2B, gồm CLB Lập Trình PTIT (ProPTIT), "
    "CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC). "
    "Bạn chỉ trả lời dựa trên tài liệu được cung cấp."
)

ANSWER_PROMPT = """TÀI LIỆU:
{context}

CÂU HỎI: {query}

YÊU CẦU:
- Trả lời trực tiếp bằng tiếng Việt, tối đa 4 câu, viết thành đoạn văn; không dùng gạch đầu dòng, tiêu đề hay markdown.
- Nếu câu hỏi chỉ hỏi một ý, gói câu trả lời trong 1-2 câu; chỉ dùng tối đa 4 câu khi câu hỏi có nhiều ý hoặc so sánh.
- Đi thẳng vào câu trả lời, không mở bài, không lặp lại câu hỏi, không thêm câu kết luận hay giải thích cách tính.
- Không nhắc đến "tài liệu", không bình luận về nguồn hay độ tin cậy, trừ khi câu hỏi hỏi về nguồn.
- Nêu đúng tên riêng, ngày tháng, số liệu; giữ nguyên mốc thời gian của thông tin và dùng lại từ khóa chính của câu hỏi.
- Giữ nguyên cách viết cụm từ quan trọng như trong tài liệu (ngày/tháng/năm, số năm, tên ban, tên giải) để không làm sai lệch dữ kiện.
- Khi nêu một dữ kiện, dùng lại đúng cụm từ của tài liệu cho dữ kiện đó (kể cả trợ từ, từ nối và dấu phẩy bên trong cụm); không diễn giải lại thành cụm khác.
- Nếu câu hỏi hỏi về một loại thông tin (giải thưởng, mốc thời gian, cuộc thi, tên ban, quy định, con số), nêu đủ tất cả các mục cùng loại có trong tài liệu và liên quan đến đối tượng được hỏi, không chỉ mục đầu tiên.
- Nếu tài liệu phân biệt nhiều đối tượng hoặc trường hợp cho cùng một quy định, nêu đủ từng trường hợp.
- Nếu câu hỏi hỏi có ban/team nào phụ trách và tài liệu liệt kê các ban, nêu tên ban phù hợp nhất ngay cả khi tài liệu không dùng đúng từ của câu hỏi.
- Với câu hỏi về số năm hoạt động/thành lập, nêu rõ "X năm" (ví dụ "14 năm") bên cạnh mốc thời gian, không chỉ nói "năm thứ X".
- Nếu câu hỏi có nhiều ý hoặc so sánh, trả lời đủ từng ý trong cùng đoạn văn.
- Nếu không có thông tin để trả lời, chỉ nói rõ "Tài liệu hiện có không cung cấp thông tin về ..."; chỉ nêu thông tin liên quan nếu thực sự giúp ích.
- Nếu tài liệu chỉ có một phần thông tin, trả lời phần có thể và nói rõ phần nào không có; không từ chối toàn bộ nếu vẫn trả lời được một phần.
- Nếu các đoạn mâu thuẫn, nêu ngắn gọn cả hai cách ghi kèm nguồn hoặc thời điểm, không khẳng định một cách là chính thức.
- Chỉ nêu thông tin liên quan trực tiếp đến câu hỏi; không thêm chi tiết ngoài phạm vi câu hỏi. Mỗi ý chỉ nêu dữ kiện trực tiếp trả lời, không thêm ví dụ, bối cảnh hoặc thông tin phụ."""

VERIFY_PROMPT = """TÀI LIỆU:
{context}

CÂU HỎI: {query}

CÂU TRẢ LỜI NHÁP:
{draft}

Hãy kiểm tra câu trả lời nháp với tài liệu và viết lại câu trả lời cuối cùng:
- Chỉ giữ thông tin được tài liệu hỗ trợ; bỏ suy đoán, bình luận ngoài lề và câu thừa.
- Giữ đúng ngày tháng, số liệu, tên riêng; trả lời đủ mọi ý của câu hỏi.
- Giữ nguyên cách viết các cụm từ, số liệu, ngày tháng quan trọng của đáp án nháp nếu tài liệu hỗ trợ; không đổi cách diễn đạt số liệu.
- Bỏ câu dẫn, câu kết luận, phần giải thích cách tính và chi tiết ngoài phạm vi câu hỏi.
- Giữ đủ các mục cùng loại (nhiều giải, nhiều mốc, nhiều trường hợp) mà tài liệu nêu và liên quan trực tiếp; không rút gọn làm mất dữ kiện.
- Giữ nguyên cụm từ dữ kiện như trong tài liệu, không diễn giải lại.
- Nếu có thể gói đủ ý trong 1-2 câu, hãy rút gọn còn 1-2 câu; chỉ giữ tối đa 4 câu cho câu hỏi nhiều ý.
- Nếu câu hỏi không hỏi về mâu thuẫn, không nêu mâu thuẫn.
- Không thêm thông tin mới ngoài tài liệu. Nếu nháp đã đúng và đủ, giữ nguyên hoặc rút gọn.
- Tối đa 4 câu, đoạn văn, không markdown. Chỉ trả về câu trả lời cuối cùng."""

FACT_PROMPT = """TÀI LIỆU:
{context}

CÂU HỎI: {query}

Liệt kê ngắn gọn các dữ kiện trong TÀI LIỆU trả lời câu hỏi. Nếu câu hỏi hỏi về một loại thông tin (ví dụ giải thưởng, mốc thành lập, cuộc thi, tên ban, con số), liệt kê TẤT CẢ dữ kiện cùng loại liên quan đến đối tượng được hỏi. Không liệt kê thông tin khác loại như lịch sử hình thành, sự kiện, số lượng thành viên hay hoạt động khác, trừ khi câu hỏi hỏi đúng thông tin đó. Mỗi dòng một dữ kiện, giữ nguyên tên riêng, số liệu và cụm từ quan trọng như trong tài liệu. Không thêm dữ kiện không có trong tài liệu, không giải thích. Nếu tài liệu không có thông tin trả lời, chỉ ghi: KHÔNG CÓ."""

COMPOSE_PROMPT = """TÀI LIỆU:
{context}

CÂU HỎI: {query}

CÁC DỮ KIỆN LIÊN QUAN ĐÃ TRÍCH:
{facts}

YÊU CẦU:
- Trả lời trực tiếp bằng tiếng Việt, viết thành đoạn văn; không dùng gạch đầu dòng, tiêu đề hay markdown.
- Dùng đầy đủ các dữ kiện trả lời trực tiếp ở trên; nếu câu hỏi có nhiều ý, nhiều mốc hoặc nhiều đối tượng, nêu đủ từng ý trong cùng đoạn. Không thêm dữ kiện nền hay thông tin ngoài phạm vi câu hỏi.
- Gói trong 1-2 câu nếu câu hỏi chỉ hỏi một ý; chỉ dùng tối đa 3 câu khi câu hỏi có nhiều ý hoặc so sánh.
- Giữ nguyên tên riêng, ngày tháng, số liệu, tên giải/ban và cụm từ quan trọng như trong tài liệu; không chèn thêm từ vào giữa tên riêng và số liệu (ví dụ giữ "Giải Ba Chuyên Tin 2022").
- Đi thẳng vào câu trả lời, không mở bài, không lặp lại câu hỏi, không thêm câu kết luận.
- Không nhắc đến "tài liệu", không bình luận về nguồn.
- Nếu dữ kiện ghi "KHÔNG CÓ" hoặc không đủ để trả lời, chỉ nói rõ "Tài liệu hiện có không cung cấp thông tin về ..."."""

VERIFY_COVERAGE_PROMPT = """TÀI LIỆU:
{context}

CÂU HỎI: {query}

CÁC DỮ KIỆN LIÊN QUAN ĐÃ TRÍCH:
{facts}

CÂU TRẢ LỜI NHÁP:
{draft}

Hãy kiểm tra câu trả lời nháp và viết lại câu trả lời cuối cùng:
- Giữ đủ mọi dữ kiện trả lời trực tiếp; không bỏ sót mốc thời gian, giải thưởng, tên ban hay con số mà tài liệu nêu.
- Bỏ thông tin không được tài liệu hỗ trợ, dữ kiện nền, câu dẫn, câu kết luận và chi tiết ngoài phạm vi câu hỏi.
- Gói trong 1-2 câu nếu câu hỏi chỉ hỏi một ý; chỉ dùng tối đa 4 câu khi câu hỏi có nhiều ý.
- Giữ nguyên cụm từ, số liệu, ngày tháng quan trọng như trong tài liệu.
- Tối đa 4 câu, đoạn văn, không markdown. Chỉ trả về câu trả lời cuối cùng."""


@dataclass
class AnswerGenerator:
    llm: LLMClient
    config: SolutionConfig
    last_error: str = field(default="", init=False)

    def _context(self, documents: list[Document], analysis: QueryAnalysis) -> str:
        # Câu hỏi nhiều ý/nhiều CLB cần nhiều passage hơn để trả lời đủ từng ý.
        limit = self.config.context_docs_multi if analysis.is_multi else self.config.context_docs
        rows = []
        for doc in documents[:limit]:
            rows.append(f"[{doc.id}] {doc.club} — {doc.topic}\n{doc.text}")
        return "\n\n".join(rows)

    def _extract_facts(self, query: str, context: str) -> str:
        """Lượt trích dữ kiện: liệt kê đủ các dữ kiện liên quan trước khi viết câu trả lời."""
        prompt = FACT_PROMPT.format(context=context, query=query)
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.llm_model,
            )
        except Exception as exc:  # pragma: no cover - phụ thuộc gateway
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ""

        lines: list[str] = []
        seen: set[str] = set()
        for line in raw.splitlines():
            line = clean_answer(line).strip(" -•*")
            if not line or line.upper() == "KHÔNG CÓ":
                continue
            key = line.lower()
            if key in seen or len(line) < 4:
                continue
            seen.add(key)
            lines.append(line)
        return "\n".join(lines)[:1600]

    def generate(
        self,
        query: str,
        documents: list[Document],
        analysis: QueryAnalysis,
    ) -> str:
        context = self._context(documents, analysis)
        facts = ""
        if self.config.use_fact_extraction:
            facts = self._extract_facts(query, context)
            prompt = COMPOSE_PROMPT.format(context=context, query=query, facts=facts or "KHÔNG CÓ")
        else:
            prompt = ANSWER_PROMPT.format(context=context, query=query)
        try:
            draft = clean_answer(
                self.llm.chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.config.llm_model,
                )
            )
        except Exception as exc:  # pragma: no cover - phụ thuộc gateway
            self.last_error = f"{type(exc).__name__}: {exc}"
            return "Tài liệu hiện có không cung cấp thông tin để trả lời câu hỏi này."

        if not draft:
            draft = "Tài liệu hiện có không cung cấp thông tin để trả lời câu hỏi này."

        if self.config.use_verification:
            verify_prompt = VERIFY_COVERAGE_PROMPT if facts else VERIFY_PROMPT
            try:
                verified = clean_answer(
                    self.llm.chat(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": verify_prompt.format(
                                    context=context,
                                    query=query,
                                    draft=draft,
                                    facts=facts or "KHÔNG CÓ",
                                ),
                            },
                        ],
                        model=self.config.llm_model,
                    )
                )
                if verified:
                    draft = verified
            except Exception as exc:  # pragma: no cover - phụ thuộc gateway
                self.last_error = f"{type(exc).__name__}: {exc}"

        return clean_answer(draft, max_chars=2000)


__all__ = [
    "AnswerGenerator",
    "ANSWER_PROMPT",
    "COMPOSE_PROMPT",
    "FACT_PROMPT",
    "SYSTEM_PROMPT",
    "VERIFY_COVERAGE_PROMPT",
    "VERIFY_PROMPT",
]
