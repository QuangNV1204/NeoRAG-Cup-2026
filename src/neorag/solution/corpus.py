"""Tầng ingestion: biểu diễn nguyên đoạn văn (whole-passage) và đơn vị chi tiết giữ tiêu đề."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from neorag.data import Document
from neorag.solution.vietnamese import club_lexical_aliases

_SENTENCE_END = re.compile(r"(?<=[.!?;…])\s+|\n+")
_FULL_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•+])\s+")


@dataclass(frozen=True)
class Chunk:
    """Một đơn vị retrieval. `context` là văn bản thực sự được embed/index."""

    id: str
    doc_id: int
    text: str
    context: str
    club: str
    topic: str
    position: int
    window: int = 1


def _make_context(club: str, topic: str, text: str) -> str:
    return f"{club} — {topic}. {text}"


def format_passage_dense_text(doc: Document) -> str:
    """Văn bản nguyên đoạn dùng cho Whole-Passage Dense Embedding (bge-m3).

    Mỗi passage trong corpus chỉ dài 1-5 câu (~100-750 ký tự), nên embed nguyên đoạn
    giữ trọn ngữ cảnh xuyên câu (ví dụ lưu ý 'chỉ áp dụng cho đợt tuyển năm 2026' ở cuối đoạn).
    """
    return f"{doc.club} — {doc.topic}: {doc.text.strip()}"


def format_passage_bm25_text(doc: Document) -> str:
    """Văn bản nguyên đoạn dùng cho BM25, gắn thêm tên gọi phổ biến của CLB."""
    return f"{club_lexical_aliases(doc.club)} — {doc.topic}: {doc.text.strip()}"


def split_sentences(text: str) -> list[str]:
    """Tách câu theo dấu kết thúc và dấu chấm phẩy."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        sentences.extend(part.strip() for part in _SENTENCE_END.split(line) if part.strip())
    return [sentence for sentence in sentences if len(sentence) > 1]


def _split_full_sentences(text: str) -> list[str]:
    """Tách theo câu hoàn chỉnh (.!?), giữ nguyên dấu chấm phẩy bên trong câu ngắn."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        sentences.extend(part.strip() for part in _FULL_SENTENCE_END.split(line) if part.strip())
    return [sentence for sentence in sentences if len(sentence) > 1]


def _split_long_semicolon_sentence(sentence: str, max_chars: int = 260) -> list[str]:
    """Tách câu liệt kê dài bằng `;` nhưng giữ phần dẫn trước dấu `:` (như giai đoạn/nhiệm kỳ)."""
    if len(sentence) <= max_chars or ";" not in sentence:
        return [sentence]

    header = ""
    body = sentence
    if ":" in sentence:
        prefix, rest = sentence.split(":", 1)
        if 8 <= len(prefix.strip()) <= 160 and ";" in rest:
            header = prefix.strip()
            body = rest.strip()

    items = [item.strip(" .") for item in body.split(";") if item.strip(" .")]
    if len(items) <= 1:
        return [sentence]

    units: list[str] = []
    step = 2
    for start in range(0, len(items), step):
        group = "; ".join(items[start : start + step])
        unit = f"{header}: {group}." if header else f"{group}."
        units.append(unit)
    return units


def build_smart_units(documents: list[Document], max_sentence_chars: int = 260) -> list[Chunk]:
    """Tạo các đơn vị con giữ nguyên ngữ cảnh tiêu đề để bổ trợ cho Whole-Passage Embedding.

    - Đoạn ngắn 1 câu: giữ nguyên 1 đơn vị duy nhất.
    - Đoạn nhiều câu: tạo đơn vị từng câu + cửa sổ 2 câu liền kề.
    - Câu liệt kê dài (giải thưởng, Ban chủ nhiệm, nội quy...): tách theo cặp mệnh đề `;`
      nhưng luôn giữ tiền tố giai đoạn/nhiệm kỳ trước dấu `:`.
    """
    units: list[Chunk] = []
    for doc in documents:
        full_sentences = _split_full_sentences(doc.text)
        if not full_sentences:
            full_sentences = [doc.text.strip()]

        expanded: list[str] = []
        for sent in full_sentences:
            expanded.extend(_split_long_semicolon_sentence(sent, max_chars=max_sentence_chars))

        seen: set[str] = set()
        pos = 0
        for idx, text in enumerate(expanded):
            candidates = [(1, text)]
            if idx + 1 < len(expanded):
                pair = f"{text} {expanded[idx + 1]}"
                if len(pair) <= 750:
                    candidates.append((2, pair))
            for window, item in candidates:
                clean = item.strip()
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                units.append(
                    Chunk(
                        id=f"u:{doc.id}:{pos}:{window}",
                        doc_id=doc.id,
                        text=clean,
                        context=_make_context(doc.club, doc.topic, clean),
                        club=doc.club,
                        topic=doc.topic,
                        position=pos,
                        window=window,
                    )
                )
                pos += 1
    return units


def build_chunks(documents: list[Document], max_window_chars: int = 1200) -> list[Chunk]:
    """Tạo chunk một câu và cửa sổ hai câu liền kề (giữ tương thích ngược)."""
    chunks: list[Chunk] = []
    for doc in documents:
        sentences = split_sentences(doc.text)
        if not sentences:
            sentences = [doc.text.strip()]
        seen: set[str] = set()
        for position, sentence in enumerate(sentences):
            candidates = [(1, sentence)]
            if position + 1 < len(sentences):
                pair = f"{sentence} {sentences[position + 1]}"
                if len(pair) <= max_window_chars:
                    candidates.append((2, pair))
            for window, text in candidates:
                text = text.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                chunks.append(
                    Chunk(
                        id=f"{doc.id}:{position}:{window}",
                        doc_id=doc.id,
                        text=text,
                        context=_make_context(doc.club, doc.topic, text),
                        club=doc.club,
                        topic=doc.topic,
                        position=position,
                        window=window,
                    )
                )
    return chunks


def fingerprint(documents: list[Document], namespace: str = "") -> str:
    """Hash ổn định của corpus để dùng làm khóa cache embedding."""
    payload = [[d.id, d.club, d.topic, d.text] for d in sorted(documents, key=lambda item: item.id)]
    raw = json.dumps([namespace, payload], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def group_chunks_by_doc(chunks: list[Chunk]) -> dict[int, list[Chunk]]:
    grouped: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


__all__ = [
    "Chunk",
    "build_chunks",
    "build_smart_units",
    "fingerprint",
    "format_passage_bm25_text",
    "format_passage_dense_text",
    "group_chunks_by_doc",
]
