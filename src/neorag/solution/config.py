"""Cấu hình cho pipeline giải pháp.

Mọi thông số có thể chỉnh bằng biến môi trường để không phải sửa code khi thử nghiệm.
"""

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
    """Tham số của pipeline. Giá trị mặc định đã được tinh chỉnh trên tập train."""

    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-5-5"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3"))
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("NEORAG_CACHE_DIR", ".cache/solution"))
    )

    # Số đoạn đưa vào prompt và số id tối đa trả về để chấm retrieval.
    context_docs: int = field(default_factory=lambda: _env_int("NEORAG_CONTEXT_DOCS", 6))
    # Câu hỏi nhiều ý/nhiều CLB cần nhiều ngữ cảnh hơn để trả lời đủ từng ý.
    context_docs_multi: int = field(
        default_factory=lambda: _env_int("NEORAG_CONTEXT_DOCS_MULTI", 10)
    )
    max_retrieved: int = field(default_factory=lambda: _env_int("NEORAG_MAX_RETRIEVED", 10))

    # Truy vấn: mở rộng bằng LLM + tách truy vấn theo CLB cho câu hỏi nhiều CLB.
    use_query_expansion: bool = field(default_factory=lambda: _env_bool("NEORAG_EXPANSION", True))
    expansion_count: int = field(default_factory=lambda: _env_int("NEORAG_EXPANSION_COUNT", 3))
    deterministic_expansion_weight: float = field(
        default_factory=lambda: _env_float("NEORAG_DET_EXPANSION_WEIGHT", 1.0)
    )
    # Query2doc/HyDE: sinh thêm một đoạn văn giả định để tăng recall của dense.
    use_hyde: bool = field(default_factory=lambda: _env_bool("NEORAG_HYDE", True))
    hyde_weight: float = field(default_factory=lambda: _env_float("NEORAG_HYDE_WEIGHT", 0.9))
    max_search_queries: int = field(
        default_factory=lambda: _env_int("NEORAG_MAX_SEARCH_QUERIES", 12)
    )

    # Truy xuất hybrid: dense (bge-m3) + BM25 trên dạng bỏ dấu.
    dense_weight: float = field(default_factory=lambda: _env_float("NEORAG_DENSE_WEIGHT", 0.65))
    sparse_weight: float = field(default_factory=lambda: _env_float("NEORAG_SPARSE_WEIGHT", 0.35))
    # Cộng điểm cho passage thuộc CLB được nhận diện trong câu hỏi.
    club_boost: float = field(default_factory=lambda: _env_float("NEORAG_CLUB_BOOST", 0.05))
    # Cộng điểm cho passage có topic khớp với chủ đề nhận diện từ câu hỏi.
    topic_boost: float = field(default_factory=lambda: _env_float("NEORAG_TOPIC_BOOST", 0.0))
    # Fuse nhiều truy vấn: max + hệ số * trung bình. Giúp câu hỏi nhiều ý không bị
    # một truy vấn quá mạnh lấn át hoàn toàn các truy vấn còn lại.
    query_fusion_mean_weight: float = field(
        default_factory=lambda: _env_float("NEORAG_QUERY_MEAN_WEIGHT", 0.08)
    )

    # Rerank bằng LLM: xem lại top ứng viên và đưa các passage thực sự cần lên trước.
    use_llm_rerank: bool = field(default_factory=lambda: _env_bool("NEORAG_LLM_RERANK", True))
    rerank_candidates: int = field(default_factory=lambda: _env_int("NEORAG_RERANK_CANDIDATES", 20))

    # Knowledge graph: dùng để mở rộng ngữ cảnh ở cuối danh sách (không ảnh hưởng top-5).
    use_graph_context: bool = field(default_factory=lambda: _env_bool("NEORAG_GRAPH_CONTEXT", True))
    graph_context_docs: int = field(default_factory=lambda: _env_int("NEORAG_GRAPH_DOCS", 2))

    # Sinh câu trả lời: một lượt trả lời + một lượt tự kiểm tra.
    use_verification: bool = field(default_factory=lambda: _env_bool("NEORAG_VERIFY", True))
    # Kiến trúc trích dữ kiện trước rồi mới viết câu trả lời (giúp phủ đủ ý).
    use_fact_extraction: bool = field(
        default_factory=lambda: _env_bool("NEORAG_FACT_EXTRACTION", False)
    )

    # Cache embedding dùng chung giữa các lần chạy.
    embedding_batch_size: int = field(default_factory=lambda: _env_int("NEORAG_EMBED_BATCH", 32))

    def __post_init__(self) -> None:
        if not isinstance(self.cache_dir, Path):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "embeddings").mkdir(parents=True, exist_ok=True)
