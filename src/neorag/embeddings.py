from __future__ import annotations

import logging
import os
import shutil
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

_load_lock = threading.Lock()

def is_downloaded(model_name: str) -> bool:
    if Path(model_name).exists():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception:
        return False

def gpu_unused(device: str) -> bool:
    return device == "cpu" and shutil.which("nvidia-smi") is not None

class _OnlyErrors(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR

def _silence_libraries() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
        _OnlyErrors()
    )
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass

@lru_cache(maxsize=4)
def _load(model_name: str):
    _silence_libraries()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=is_downloaded(model_name))

class LocalEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._lock = threading.Lock()

    def load(self):
        with _load_lock:
            return _load(self.model_name)

    @property
    def device(self) -> str:
        return str(self.load().device)

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self.load()
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)
