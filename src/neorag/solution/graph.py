"""Knowledge graph nhẹ, không cần LLM.

RAG-Anything dùng graph thực thể – quan hệ để nối các mảnh thông tin rời rạc. Với corpus
văn bản nhỏ của cuộc thi, một graph từ khóa/thực thể có trọng số IDF là đủ và không tốn
budget. Graph được dùng để:
- mở rộng ngữ cảnh ở cuối danh sách retrieval (không làm nhiễu top-5);
- giải thích vì sao hai passage liên quan;
- hỗ trợ câu hỏi nhiều đoạn / nhiều CLB.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from neorag.data import Document
from neorag.solution.corpus import Chunk
from neorag.solution.vietnamese import ascii_text, content_words

_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_URL = re.compile(r"(?:https?://|www\.)[^\s,;)]+|[\w.+-]+@[\w.-]+\.\w+|@[\w.]+")
_ACRONYM = re.compile(r"\b[A-ZĐ][A-ZĐ0-9]{1,7}\b")
_PROPER = re.compile(r"(?<!\w)(?:[A-ZĐ][\wÀ-ỹ]*)(?:\s+(?:[A-ZĐ][\wÀ-ỹ]*|và|&)){1,3}")

_ENTITY_STOP = {
    "clb",
    "cau",
    "lac",
    "bo",
    "ban",
    "va",
    "cua",
    "cho",
    "trong",
    "mot",
    "cac",
    "duoc",
    "voi",
    "de",
    "khi",
    "tu",
    "nay",
    "do",
    "nhu",
    "the",
    "la",
    "co",
    "khong",
    "phai",
}


def extract_entities(text: str) -> set[str]:
    """Trích thực thể có thể so khớp chính xác: ngày, năm, URL, viết tắt, cụm danh từ."""

    entities: set[str] = set()
    for regex in (_DATE, _YEAR, _URL):
        for match in regex.findall(text):
            entities.add(ascii_text(match))
    for match in _ACRONYM.findall(text):
        key = ascii_text(match)
        if len(key) > 1 and key not in _ENTITY_STOP:
            entities.add(key)
    for match in _PROPER.findall(text):
        key = ascii_text(match)
        words = [w for w in key.split() if w not in _ENTITY_STOP]
        if len(words) >= 2 and len(" ".join(words)) >= 5:
            entities.add(" ".join(words))
    return {e for e in entities if e}


def _doc_terms(doc: Document) -> set[str]:
    text = f"{doc.club} {doc.topic} {doc.text}"
    terms = set(content_words(text, ascii_only=True))
    terms.update(extract_entities(text))
    return {t for t in terms if len(t) > 1}


@dataclass(frozen=True)
class GraphEdge:
    doc_id: int
    score: float
    shared: tuple[str, ...]


class KnowledgeGraph:
    """Graph hai tầng: passage ↔ term/entity, kèm cạnh passage ↔ passage."""

    def __init__(self, documents: list[Document], chunks: list[Chunk] | None = None):
        self.documents = documents
        self.chunks = chunks or []
        self.by_id = {d.id: d for d in documents}
        self.terms_by_doc: dict[int, set[str]] = {}
        for doc in documents:
            self.terms_by_doc[doc.id] = _doc_terms(doc)

        df = Counter()
        for terms in self.terms_by_doc.values():
            df.update(terms)
        n_docs = max(1, len(documents))
        # Từ quá phổ biến (ví dụ "ptit", "clb") không tạo ra tín hiệu phân biệt.
        self.term_df = df
        self.term_idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }
        self._max_df = max(3, int(0.60 * n_docs))
        self._index: dict[str, list[int]] = defaultdict(list)
        for doc_id, terms in self.terms_by_doc.items():
            for term in terms:
                if df[term] <= self._max_df:
                    self._index[term].append(doc_id)

        self.neighbors: dict[int, list[GraphEdge]] = {}
        self._build_edges()
        self.entities_by_doc = {
            doc.id: extract_entities(f"{doc.club} {doc.topic} {doc.text}") for doc in documents
        }

    def _build_edges(self, max_neighbors: int = 6) -> None:
        for doc in self.documents:
            scores: dict[int, float] = defaultdict(float)
            shared_terms: dict[int, list[str]] = defaultdict(list)
            for term in self.terms_by_doc[doc.id]:
                if self.term_df.get(term, 0) > self._max_df:
                    continue
                for other_id in self._index.get(term, []):
                    if other_id == doc.id:
                        continue
                    scores[other_id] += self.term_idf.get(term, 0.0)
                    shared_terms[other_id].append(term)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_neighbors]
            self.neighbors[doc.id] = [
                GraphEdge(
                    doc_id=other_id,
                    score=score,
                    shared=tuple(
                        sorted(shared_terms[other_id], key=lambda t: -self.term_idf.get(t, 0))
                    )[:6],
                )
                for other_id, score in ranked
            ]

    def related(self, doc_ids: list[int], limit: int = 2) -> list[tuple[int, float]]:
        """Trả về passage lân cận của một nhóm passage, ưu tiên passage bổ sung thông tin."""

        seed = set(doc_ids)
        aggregate: dict[int, float] = defaultdict(float)
        for doc_id in seed:
            for edge in self.neighbors.get(doc_id, []):
                if edge.doc_id not in seed:
                    aggregate[edge.doc_id] += edge.score
        if not aggregate:
            return []
        seed_topics = {self.by_id[i].topic for i in seed if i in self.by_id}
        seed_clubs = {self.by_id[i].club for i in seed if i in self.by_id}
        ranked = []
        for other_id, score in aggregate.items():
            doc = self.by_id.get(other_id)
            if doc is None:
                continue
            # Ưu tiên nhẹ passage khác chủ đề/CLB để tăng độ phủ ngữ cảnh.
            bonus = 0.08 if doc.topic not in seed_topics else 0.0
            bonus += 0.04 if doc.club not in seed_clubs else 0.0
            ranked.append((other_id, score * (1.0 + bonus)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:limit]

    def shared_terms(self, left_id: int, right_id: int) -> list[str]:
        left = self.terms_by_doc.get(left_id, set())
        right = self.terms_by_doc.get(right_id, set())
        return sorted(left & right, key=lambda t: -self.term_idf.get(t, 0.0))

    def summary(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "terms": len(self.term_df),
            "edges": sum(len(v) for v in self.neighbors.values()),
        }


__all__ = ["GraphEdge", "KnowledgeGraph", "extract_entities"]
