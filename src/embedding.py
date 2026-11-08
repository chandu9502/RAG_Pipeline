import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from typing import List
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        print(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded successfully.")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            print("No documents provided to chunk.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(
        self,
        chunks: List[Document],
        batch_size: int = 32
    ) -> np.ndarray:
        if not chunks:
            print("No chunks provided to embed.")
            return np.empty((0,))

        texts = [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks locally...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True 
        )

        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()

    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)

    if len(embeddings) > 0:
        print("First embedding vector dimension:", embeddings[0].shape)
        print("Sample vector values (first 5 elements):", embeddings[0][:5])
    else:
        print("No embeddings were generated.")
