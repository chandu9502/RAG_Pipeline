import os
import sys
from typing import Optional

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from .data_loader import load_all_documents
    from .embedding import EmbeddingPipeline
    from .vectorstore import FaissVectorStore
    from .search import RAGSearch
    from .evaluation import (
        correctness,
        relevance,
        groundedness,
        retrieval_relevance,
        run_evaluation_langsmith,
        run_evaluation_local,
    )
except ImportError:
    from data_loader import load_all_documents
    from embedding import EmbeddingPipeline
    from vectorstore import FaissVectorStore
    from search import RAGSearch
    from evaluation import (
        correctness,
        relevance,
        groundedness,
        retrieval_relevance,
        run_evaluation_langsmith,
        run_evaluation_local,
    )


class RAGPipeline:
   
    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-20b",
    ):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self._search_engine: Optional[RAGSearch] = None

    def build_index(self):
        docs = load_all_documents(self.data_dir)
        store = FaissVectorStore(self.persist_dir, self.embedding_model)
        store.build_from_documents(docs)
        return store

    def get_search_engine(self) -> RAGSearch:
        if self._search_engine is None:
            self._search_engine = RAGSearch(
                persist_dir=self.persist_dir,
                embedding_model=self.embedding_model,
                llm_model=self.llm_model,
            )
        return self._search_engine

    def query(self, question: str, top_k: int = 5, score_threshold: float = 0.45) -> str:
        search = self.get_search_engine()
        return search.search_and_summarize(question, top_k=top_k, score_threshold=score_threshold)

    def search_and_retrieve(self, question: str, top_k: int = 5, score_threshold: float = 0.45) -> dict:
        search = self.get_search_engine()
        return search.search_and_retrieve(question, top_k=top_k, score_threshold=score_threshold)

    def evaluate(self, mode: str = "local"):
        if mode == "langsmith":
            return run_evaluation_langsmith()
        return run_evaluation_local()


__all__ = [
    "load_all_documents",
    "EmbeddingPipeline",
    "FaissVectorStore",
    "RAGSearch",
    "RAGPipeline",
    "correctness",
    "relevance",
    "groundedness",
    "retrieval_relevance",
    "run_evaluation_langsmith",
    "run_evaluation_local",
]
