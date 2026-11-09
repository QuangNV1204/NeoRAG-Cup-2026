from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.solution.config import SolutionConfig
from neorag.solution.corpus import Chunk
from neorag.solution.graph import KnowledgeGraph
from neorag.solution.index import BM25Index, DenseIndex, normalize01
from neorag.solution.query import QueryAnalysis, SearchQuery
from neorag.solution.vietnamese import MAIN_CLUBS, ascii_text, topic_labels

_RERANK_PROMPT = """Bạn là bộ xếp hạng passage cho hệ thống RAG.

CÂU HỎI: {query}

CÁC PASSAGE ỨNG VIÊN:
{candidates}

Hãy chọn tối đa {limit} passage thực sự cần nhất để trả lời câu hỏi, xếp từ quan trọng nhất đến ít quan trọng nhất. Nếu câu hỏi có nhiều ý hoặc nhắc nhiều CLB, phải chọn đủ passage cho từng ý/CLB. Chỉ trả về ID, mỗi dòng một ID, không giải thích."""

_RERANK_LINE = re.compile(
    r"^\s*(?:[-*•+]\s*)?(?:\d+\s*[.)]\s*)?(?:ID\s*[:.]?\s*)?\[?(\d+)\]?[.)]?\s*$",
    re.I,
)
_RERANK_CLEAN = re.compile(r"^[\s\d,\[\]\-]+$")
_RERANK_NUMBERS = re.compile(r"\d+")

@dataclass
class RetrievalResult:
    documents: list[Document]
    scores: dict[int, float]
    search_queries: list[SearchQuery]
    analysis: QueryAnalysis
    graph_ids: list[int]

class HybridRetriever:

    def __init__(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        dense: DenseIndex,
        bm25: BM25Index,
        graph: KnowledgeGraph,
        config: SolutionConfig,
        llm: LLMClient | None = None,
    ):
        self.documents = documents
        self.chunks = chunks
        self.dense = dense
        self.bm25 = bm25
        self.graph = graph
        self.config = config
        self.llm = llm
        self.doc_positions = {doc.id: i for i, doc in enumerate(documents)}
        self.chunk_groups: dict[int, list[int]] = {}
        for i, chunk in enumerate(chunks):
            self.chunk_groups.setdefault(chunk.doc_id, []).append(i)

    def _dense_doc_scores(self, query_text: str) -> np.ndarray:
        query_vector = self.dense.embed_query(query_text)
        chunk_scores = self.dense.scores(query_vector)
        doc_scores = np.full(len(self.documents), -1e9, dtype=np.float32)
        for doc_id, indices in self.chunk_groups.items():
            position = self.doc_positions.get(doc_id)
            if position is None:
                continue
            values = chunk_scores[np.asarray(indices)]
            if values.size == 0:
                continue
            if values.size == 1:
                doc_scores[position] = float(values[0])
            else:
                top = np.partition(values, -2)[-2:]
                doc_scores[position] = float(top.mean())
        return doc_scores

    def _score_search_query(self, search_query: SearchQuery) -> np.ndarray:
        dense_scores = normalize01(self._dense_doc_scores(search_query.text))
        if search_query.kind == "hyde":
            return dense_scores * search_query.weight
        sparse_scores = normalize01(self.bm25.scores(search_query.text))
        weights = self.config.dense_weight + self.config.sparse_weight
        fused = (
            self.config.dense_weight * dense_scores + self.config.sparse_weight * sparse_scores
        ) / max(weights, 1e-9)
        return fused * search_query.weight

    def _apply_coverage(
        self,
        order: list[int],
        scores: np.ndarray,
        analysis: QueryAnalysis,
    ) -> list[int]:
        markers = ("ca 3", "3 clb", "ba clb", "lau doi", "tre nhat")
        explicit_coverage = len(analysis.main_clubs) >= 2 or any(
            marker in ascii_text(analysis.query) for marker in markers
        )
        if not analysis.is_multi or not explicit_coverage:
            return order

        coverage_clubs = [c for c in analysis.main_clubs if c in MAIN_CLUBS]
        if not coverage_clubs:
            coverage_clubs = list(MAIN_CLUBS)
        selected: list[int] = []
        for club in coverage_clubs:
            for index in order:
                if self.documents[index].club == club:
                    selected.append(index)
                    break
        selected = sorted(set(selected), key=lambda i: (-float(scores[i]), i))
        selected_set = set(selected)
        return selected + [i for i in order if i not in selected_set]

    def _ordered_indices(self, scores: np.ndarray, analysis: QueryAnalysis) -> list[int]:
        order = [int(i) for i in np.argsort(-scores)]
        return self._apply_coverage(order, scores, analysis)

    def _llm_rerank(
        self,
        query: str,
        order: list[int],
    ) -> list[int]:
        if not self.config.use_llm_rerank or self.llm is None or len(order) <= 3:
            return order

        candidates = order[: self.config.rerank_candidates]
        candidate_set = set(candidates)
        rows = []
        for index in candidates:
            doc = self.documents[index]
            text = " ".join(doc.text.split())
            if len(text) > 500:
                text = text[:500].rsplit(" ", 1)[0] + "…"
            rows.append(f"[{doc.id}] {doc.club} — {doc.topic}\n{text}")

        limit = min(5, len(candidates))
        prompt = _RERANK_PROMPT.format(
            query=query,
            candidates="\n\n".join(rows),
            limit=limit,
        )
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}], model=self.config.llm_model)
        except Exception:
            return order

        selected: list[int] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _RERANK_LINE.match(line)
            if match:
                ids = [int(match.group(1))]
            elif _RERANK_CLEAN.match(line):
                ids = [int(value) for value in _RERANK_NUMBERS.findall(line)]
            else:
                continue
            for doc_id in ids:
                position = self.doc_positions.get(doc_id)
                if position in candidate_set and position not in selected:
                    selected.append(position)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

        if len(selected) < min(3, len(candidates)):
            return order
        selected_set = set(selected)
        return selected + [i for i in order if i not in selected_set]

    def retrieve(
        self,
        analysis: QueryAnalysis,
        search_queries: list[SearchQuery],
    ) -> RetrievalResult:
        if not search_queries:
            raise ValueError("search_queries rỗng")

        score_matrix = np.stack([self._score_search_query(q) for q in search_queries])
        scores = score_matrix.max(axis=0)
        if self.config.query_fusion_mean_weight:
            scores = scores + self.config.query_fusion_mean_weight * score_matrix.mean(axis=0)

        if self.config.club_boost and analysis.clubs:
            for i, doc in enumerate(self.documents):
                if doc.club in analysis.clubs:
                    scores[i] += self.config.club_boost

        labels = topic_labels(analysis.topics)
        if self.config.topic_boost and labels:
            for i, doc in enumerate(self.documents):
                if doc.topic in labels:
                    scores[i] += self.config.topic_boost

        order = self._ordered_indices(scores, analysis)
        order = self._llm_rerank(analysis.query, order)
        order = self._apply_coverage(order, scores, analysis)
        primary = order[: self.config.context_docs]
        selected = list(primary)
        selected_set = set(selected)

        graph_ids: list[int] = []
        if self.config.use_graph_context and len(selected) < self.config.max_retrieved:
            seed_ids = [self.documents[i].id for i in primary]
            related = self.graph.related(seed_ids, limit=self.config.graph_context_docs)
            for doc_id, _ in related:
                position = self.doc_positions.get(doc_id)
                if position is None or position in selected_set:
                    continue
                graph_ids.append(doc_id)
                selected.append(position)
                selected_set.add(position)
                if len(graph_ids) >= self.config.graph_context_docs:
                    break

        for index in order:
            if len(selected) >= self.config.max_retrieved:
                break
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)

        selected = selected[: self.config.max_retrieved]
        documents = [self.documents[i] for i in selected]
        score_map = {self.documents[i].id: float(scores[i]) for i in selected}
        return RetrievalResult(
            documents=documents,
            scores=score_map,
            search_queries=search_queries,
            analysis=analysis,
            graph_ids=graph_ids,
        )

__all__ = ["HybridRetriever", "RetrievalResult"]
