from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default

def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default

@dataclass(frozen=True)
class SolutionConfig:

    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-5-5"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3"))
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("NEORAG_CACHE_DIR", ".cache/solution"))
    )

    context_docs: int = field(default_factory=lambda: _env_int("NEORAG_CONTEXT_DOCS", 6))
    context_docs_multi: int = field(
        default_factory=lambda: _env_int("NEORAG_CONTEXT_DOCS_MULTI", 10)
    )
    max_retrieved: int = field(default_factory=lambda: _env_int("NEORAG_MAX_RETRIEVED", 10))

    use_query_expansion: bool = field(default_factory=lambda: _env_bool("NEORAG_EXPANSION", True))
    expansion_count: int = field(default_factory=lambda: _env_int("NEORAG_EXPANSION_COUNT", 3))
    deterministic_expansion_weight: float = field(
        default_factory=lambda: _env_float("NEORAG_DET_EXPANSION_WEIGHT", 1.0)
    )
    use_hyde: bool = field(default_factory=lambda: _env_bool("NEORAG_HYDE", True))
    hyde_weight: float = field(default_factory=lambda: _env_float("NEORAG_HYDE_WEIGHT", 0.9))
    max_search_queries: int = field(
        default_factory=lambda: _env_int("NEORAG_MAX_SEARCH_QUERIES", 12)
    )

    dense_weight: float = field(default_factory=lambda: _env_float("NEORAG_DENSE_WEIGHT", 0.65))
    sparse_weight: float = field(default_factory=lambda: _env_float("NEORAG_SPARSE_WEIGHT", 0.35))
    club_boost: float = field(default_factory=lambda: _env_float("NEORAG_CLUB_BOOST", 0.05))
    topic_boost: float = field(default_factory=lambda: _env_float("NEORAG_TOPIC_BOOST", 0.0))
    query_fusion_mean_weight: float = field(
        default_factory=lambda: _env_float("NEORAG_QUERY_MEAN_WEIGHT", 0.08)
    )

    use_llm_rerank: bool = field(default_factory=lambda: _env_bool("NEORAG_LLM_RERANK", True))
    rerank_candidates: int = field(default_factory=lambda: _env_int("NEORAG_RERANK_CANDIDATES", 20))

    use_graph_context: bool = field(default_factory=lambda: _env_bool("NEORAG_GRAPH_CONTEXT", True))
    graph_context_docs: int = field(default_factory=lambda: _env_int("NEORAG_GRAPH_DOCS", 2))

    use_verification: bool = field(default_factory=lambda: _env_bool("NEORAG_VERIFY", True))
    use_fact_extraction: bool = field(
        default_factory=lambda: _env_bool("NEORAG_FACT_EXTRACTION", False)
    )

    embedding_batch_size: int = field(default_factory=lambda: _env_int("NEORAG_EMBED_BATCH", 32))

    def __post_init__(self) -> None:
        if not isinstance(self.cache_dir, Path):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "embeddings").mkdir(parents=True, exist_ok=True)
