import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src import RAGPipeline


def main():
    pipeline = RAGPipeline(
        data_dir="data",
        persist_dir="src/faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="openai/gpt-oss-20b"
    )

    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:]).strip()
        if query_text == "--build":
            print("Building vector store index from documents in data/...")
            pipeline.build_index()
            print("Index successfully built.")
            return

        print(f"Query: {query_text}")
        answer = pipeline.query(query_text, top_k=3, score_threshold=0.45)
        print("\nAnswer:")
        print(answer)
        return

    if not sys.stdin.isatty():
        sample_query = "What is qubit?"
        print(f"Running sample query: '{sample_query}'")
        answer = pipeline.query(sample_query, top_k=3, score_threshold=0.45)
        print("\nAnswer:")
        print(answer)
        return

    print("=" * 60)
    print("RAG Pipeline Interactive Assistant")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 60)

    while True:
        try:
            query = input("\nEnter query: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break

            answer = pipeline.query(query, top_k=3, score_threshold=0.45)
            print("\nAnswer:")
            print(answer)
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()

