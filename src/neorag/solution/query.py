from __future__ import annotations

import re
from dataclasses import dataclass, field

from neorag.llm import LLMClient
from neorag.solution.config import SolutionConfig
from neorag.solution.vietnamese import (
    MAIN_CLUBS,
    ascii_text,
    content_words,
    detect_clubs,
    detect_topics,
    has_diacritics,
    is_founding_comparison,
    is_multi_club_question,
)

_EXPANSION_PROMPT = """Từ câu hỏi người dùng, tạo tối đa 3 truy vấn tìm kiếm ngắn gọn bằng tiếng Việt để tra cứu trong một corpus về 3 câu lạc bộ: CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC), CLB Nhà Sáng Tạo Game PTIT (GCC). Mỗi dòng một truy vấn, không đánh số, không trả lời câu hỏi. Giữ nguyên tên riêng, số liệu, ngày tháng và từ viết tắt. Nếu câu hỏi so sánh hoặc hỏi về nhiều CLB, tạo truy vấn riêng cho từng CLB liên quan, ghi rõ tên CLB.

CÂU HỎI: {query}"""

_HYDE_PROMPT = """Viết một đoạn văn ngắn (2-4 câu) bằng tiếng Việt chứa các từ khóa, tên riêng, chủ đề và cách diễn đạt có khả năng xuất hiện trong tài liệu trả lời câu hỏi dưới đây về 3 câu lạc bộ: CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC), CLB Nhà Sáng Tạo Game PTIT (GCC). Không cần bịa số liệu hay ngày tháng cụ thể; chỉ mô tả thông tin liên quan. Không trả lời trực tiếp, không giải thích.

CÂU HỎI: {query}"""

_LINE_MARKER = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s*")

@dataclass(frozen=True)
class QueryAnalysis:
    query: str
    clubs: list[str]
    main_clubs: list[str]
    is_multi: bool
    is_founding: bool
    topics: set[str]
    keywords: list[str]
    ascii_query: bool

@dataclass(frozen=True)
class SearchQuery:
    text: str
    weight: float = 1.0
    kind: str = "original"

@dataclass
class QueryExpander:
    llm: LLMClient | None
    config: SolutionConfig
    last_error: str = field(default="", init=False)

    def expand(self, query: str, analysis: QueryAnalysis) -> list[SearchQuery]:
        queries = [SearchQuery(query, 1.0, "original")]
        if self.config.use_query_expansion and self.llm is not None:
            for text in self._llm_expand(query):
                queries.append(SearchQuery(text, 1.0, "llm"))
        for text in self._deterministic(query, analysis):
            queries.append(SearchQuery(text, self.config.deterministic_expansion_weight, "club"))
        if self.config.use_hyde and self.llm is not None and (text := self._llm_hyde(query)):
            queries.append(SearchQuery(text, self.config.hyde_weight, "hyde"))

        deduped: list[SearchQuery] = []
        seen: set[str] = set()
        for item in queries:
            key = ascii_text(item.text)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[: self.config.max_search_queries]

    def _llm_expand(self, query: str) -> list[str]:
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": _EXPANSION_PROMPT.format(query=query)}],
                model=self.config.llm_model,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []

        result: list[str] = []
        for line in raw.splitlines():
            line = _LINE_MARKER.sub("", line).strip().strip('"').strip("'")
            if len(line) < 4 or len(line) > 220:
                continue
            if "câu hỏi" in ascii_text(line) and "?" in line:
                continue
            result.append(line)
            if len(result) >= self.config.expansion_count:
                break
        return result

    def _llm_hyde(self, query: str) -> str:
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
                model=self.config.llm_model,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ""

        text = " ".join(raw.split()).strip().strip('"').strip("'")
        if len(text) < 40:
            return ""
        return text[:700]

    def _deterministic(self, query: str, analysis: QueryAnalysis) -> list[str]:
        result: list[str] = []
        if analysis.is_multi:
            clubs = [c for c in analysis.clubs if c in MAIN_CLUBS] or MAIN_CLUBS
            for club in clubs:
                result.append(f"{club} {query}")
        if analysis.is_founding:
            for club in MAIN_CLUBS:
                result.append(f"{club} thành lập ngày tháng năm nào")
        if "Cơ cấu" in analysis.topics:
            clubs = [c for c in analysis.clubs if c in MAIN_CLUBS]
            if clubs:
                for club in clubs:
                    result.append(f"{club} các ban cơ cấu tổ chức")
            else:
                result.append("các ban cơ cấu tổ chức câu lạc bộ")
        return result

class QueryAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        clubs = detect_clubs(query)
        main = [c for c in clubs if c in MAIN_CLUBS]
        return QueryAnalysis(
            query=query,
            clubs=clubs,
            main_clubs=main,
            is_multi=is_multi_club_question(query),
            is_founding=is_founding_comparison(query),
            topics=detect_topics(query),
            keywords=content_words(query),
            ascii_query=not has_diacritics(query),
        )

__all__ = ["QueryAnalysis", "QueryAnalyzer", "QueryExpander", "SearchQuery"]
