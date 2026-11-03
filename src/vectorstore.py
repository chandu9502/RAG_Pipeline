import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Optional

from embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        if not os.path.exists(persist_dir):
            src_persist = os.path.join(os.path.dirname(__file__), persist_dir)
            if os.path.exists(src_persist):
                persist_dir = src_persist
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.emb_pipe = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def build_from_documents(self, documents: List[Any]):
        print(f"Building vector store from {len(documents)} raw documents...")
        chunks = self.emb_pipe.chunk_documents(documents)
        embeddings = self.emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3, score_threshold: Optional[float] = None):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue
            similarity = 1.0 - (float(dist) / 2.0)
            if score_threshold is not None and similarity < score_threshold:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": float(dist), "score": similarity, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 3, score_threshold: Optional[float] = None):
        print(f"Querying vector store for: '{query_text}'")
        query_emb = self.emb_pipe.model.encode([query_text], normalize_embeddings=True).astype('float32')
        return self.search(query_emb, top_k=top_k, score_threshold=score_threshold)

if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is qubit?", top_k=3, score_threshold=0.45))
