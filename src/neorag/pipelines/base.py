from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from neorag.data import Document

@dataclass
class RAGOutput:
    answer: str
    retrieved_ids: list[int]

class RAGPipeline(ABC):
    def prepare(self) -> None:
        return None

    @abstractmethod
    def index(self, corpus: list[Document]) -> None:
        pass

    @abstractmethod
    def run(self, query: str) -> RAGOutput:
        pass

def load_pipeline(spec: str, **kwargs) -> RAGPipeline:
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"Pipeline phải có dạng 'module.path:ClassName', nhận được: {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(cls, RAGPipeline):
        raise TypeError(f"{spec} phải kế thừa neorag.pipelines.RAGPipeline")
    return cls(**kwargs)
