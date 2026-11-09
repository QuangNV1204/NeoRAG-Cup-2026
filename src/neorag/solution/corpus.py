from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from neorag.data import Document

_SENTENCE_END = re.compile(r"(?<=[.!?;…])\s+|\n+")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•+])\s+")

@dataclass(frozen=True)
class Chunk:

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

def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        sentences.extend(part.strip() for part in _SENTENCE_END.split(line) if part.strip())
    return [sentence for sentence in sentences if len(sentence) > 1]

def build_chunks(documents: list[Document], max_window_chars: int = 1200) -> list[Chunk]:
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
    payload = [[d.id, d.club, d.topic, d.text] for d in sorted(documents, key=lambda item: item.id)]
    raw = json.dumps([namespace, payload], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def group_chunks_by_doc(chunks: list[Chunk]) -> dict[int, list[Chunk]]:
    grouped: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped

__all__ = ["Chunk", "build_chunks", "fingerprint", "group_chunks_by_doc"]
