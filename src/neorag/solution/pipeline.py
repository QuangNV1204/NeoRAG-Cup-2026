from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.solution.cag import CAGPipeline
from neorag.solution.config import SolutionConfig
from neorag.solution.corpus import build_chunks
from neorag.solution.focused import NeoRAGPipeline
from neorag.solution.generate import AnswerGenerator
from neorag.solution.graph import KnowledgeGraph
from neorag.solution.index import BM25Index, DenseIndex
from neorag.solution.query import QueryAnalyzer, QueryExpander
from neorag.solution.retrieve import HybridRetriever
from neorag.solution.vietnamese import ascii_content_words_raw


class RAGAnythingPipeline(RAGPipeline):
    def __init__(
        self,
        config: SolutionConfig | None = None,
        llm: LLMClient | None = None,
        embedder: Any | None = None,
        **_: Any,
    ):
        self.config = config or SolutionConfig()
        self.config.ensure_dirs()
        self.llm = llm or LLMClient(cache_path=self.config.cache_dir / "llm.sqlite")
        self.embedder = embedder
        self.llm_model = self.config.llm_model
        self.embedding_model = self.config.embedding_model

        self.documents: list[Document] = []
        self.chunks = []
        self.graph: KnowledgeGraph | None = None
        self.retriever: HybridRetriever | None = None
        self.expander: QueryExpander | None = None
        self.generator: AnswerGenerator | None = None
        self.analyzer = QueryAnalyzer()
        self.last_debug: dict[str, Any] = {}

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

    def prepare(self) -> None:
        self.config.ensure_dirs()

    def index(self, corpus: list[Document]) -> None:
        self.documents = list(corpus)
        if not self.documents:
            raise ValueError("Corpus rỗng")

        self.chunks = build_chunks(self.documents)
        dense = DenseIndex(
            embed=self._embed,
            cache_dir=self.config.cache_dir,
            embedding_model=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        ).fit([chunk.context for chunk in self.chunks], name="chunks")
        bm25 = BM25Index(
            [f"{doc.club} {doc.topic} {doc.text}" for doc in self.documents],
            tokenizer=ascii_content_words_raw,
        )
        self.graph = KnowledgeGraph(self.documents, self.chunks)
        self.retriever = HybridRetriever(
            documents=self.documents,
            chunks=self.chunks,
            dense=dense,
            bm25=bm25,
            graph=self.graph,
            config=self.config,
            llm=self.llm,
        )
        self.expander = QueryExpander(self.llm, self.config)
        self.generator = AnswerGenerator(self.llm, self.config)

    def run(self, query: str) -> RAGOutput:
        if self.retriever is None or self.expander is None or self.generator is None:
            raise RuntimeError("Pipeline chưa được index. Gọi index(corpus) trước.")

        analysis = self.analyzer.analyze(query)
        search_queries = self.expander.expand(query, analysis)
        retrieval = self.retriever.retrieve(analysis, search_queries)
        answer = self.generator.generate(query, retrieval.documents, analysis)

        retrieved_ids = [doc.id for doc in retrieval.documents][: self.config.max_retrieved]
        if not retrieved_ids:
            retrieved_ids = [self.documents[0].id]

        self.last_debug = {
            "query": query,
            "analysis": {
                "clubs": analysis.clubs,
                "is_multi": analysis.is_multi,
                "is_founding": analysis.is_founding,
                "topics": sorted(analysis.topics),
                "ascii_query": analysis.ascii_query,
            },
            "search_queries": [asdict(item) for item in search_queries],
            "retrieved_ids": retrieved_ids,
            "scores": retrieval.scores,
            "graph_ids": retrieval.graph_ids,
            "answer": answer,
        }
        return RAGOutput(answer=answer, retrieved_ids=retrieved_ids)


__all__ = ["CAGPipeline", "NeoRAGPipeline", "RAGAnythingPipeline"]
