import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-20b"):
        if not os.path.exists(persist_dir):
            src_persist = os.path.join(os.path.dirname(os.path.abspath(__file__)), persist_dir)
            if os.path.exists(src_persist):
                persist_dir = src_persist

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model, temperature=0.1)
        print(f"Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5, score_threshold: float = 0.45) -> str:
        results = self.vectorstore.query(query, top_k=top_k, score_threshold=score_threshold)
        texts = [r["metadata"].get("text", "") for r in results if r and r.get("metadata")]
        context = "\n\n".join(texts).strip()
        if not context:
            return "No relevant documents found."
        prompt = (
            f"You are a helpful assistant. Answer the query using ONLY the provided context.\n"
            f"If the context does not contain enough relevant information to answer the question, state: "
            f"'I could not find relevant information in the documents to answer this question.'\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query}\n\n"
            f"Answer:"
        )
        response = self.llm.invoke([prompt])
        return response.content

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is no cloning theorem?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
