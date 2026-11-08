import os
import sys
import time
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

load_dotenv()

langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGSMith_API_KEY")
if langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key

langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
os.environ["LANGSMITH_ENDPOINT"] = langsmith_endpoint
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")

from langsmith import Client, traceable
import pandas as pd

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from .search import RAGSearch
except ImportError:
    from search import RAGSearch


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]


correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


def get_grader_llm(schema: Any):
   
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        eval_model = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-20b")
        return ChatGroq(groq_api_key=groq_key, model_name=eval_model, temperature=0).with_structured_output(
            schema
        )

    raise ValueError("GROQ_API_KEY found in environment.")


def _get_val(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return bool(obj.get(key, False))
    return bool(getattr(obj, key, False))


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    answers = f"QUESTION: {inputs['question']}\nGROUND TRUTH ANSWER: {reference_outputs['answer']}\nSTUDENT ANSWER: {outputs['answer']}"
    grader = get_grader_llm(CorrectnessGrade)
    grade = grader.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": answers},
    ])
    return _get_val(grade, "correct")


def relevance(inputs: dict, outputs: dict) -> bool:
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grader = get_grader_llm(RelevanceGrade)
    grade = grader.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return _get_val(grade, "relevant")


def groundedness(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents", [])
    doc_string = "\n\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs)
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grader = get_grader_llm(GroundedGrade)
    grade = grader.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": answer},
    ])
    return _get_val(grade, "grounded")


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents", [])
    doc_string = "\n\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs)
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    grader = get_grader_llm(RetrievalRelevanceGrade)
    grade = grader.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return _get_val(grade, "relevant")


_rag_search: Optional[RAGSearch] = None

def get_rag_search(persist_dir: str = "src/faiss_store") -> RAGSearch:
    global _rag_search
    if _rag_search is None:
        _rag_search = RAGSearch(persist_dir=persist_dir)
    return _rag_search


@traceable(name="quantum_rag_bot")
def rag_bot(question: str, top_k: int = 3, score_threshold: float = 0.45) -> dict:
    search_engine = get_rag_search()
    return search_engine.search_and_retrieve(question, top_k=top_k, score_threshold=score_threshold)


def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])


QUANTUM_BENCHMARK_EXAMPLES = [
    {
        "inputs": {"question": "What is a qubit in quantum computing?"},
        "outputs": {
            "answer": "A qubit is the fundamental unit of quantum information, analogous to a classical bit, which can exist in a superposition of states |0⟩ and |1⟩ in a two-dimensional Hilbert space."
        },
    },
    {
        "inputs": {"question": "What is the no-cloning theorem?"},
        "outputs": {
            "answer": "The no-cloning theorem states that it is impossible to create an identical copy of an arbitrary unknown quantum state."
        },
    },
    {
        "inputs": {"question": "How does an n-qubit quantum system scale in Hilbert space size?"},
        "outputs": {
            "answer": "An n-qubit quantum system has access to a Hilbert space of dimension 2^n, enabling exponential state representations compared to classical bits."
        },
    },
    {
        "inputs": {"question": "What are Variational Quantum Eigensolvers (VQE) used for?"},
        "outputs": {
            "answer": "Variational Quantum Eigensolvers (VQE) are hybrid quantum-classical algorithms that use parameterized quantum circuits and classical optimization to find the ground state energy of quantum systems."
        },
    },
    {
        "inputs": {"question": "What is quantum decoherence?"},
        "outputs": {
            "answer": "Quantum decoherence is the decay of quantum superposition into classical states caused by environmental interaction and noise."
        },
    },
]


def get_or_create_dataset(client: Client, dataset_name: str = "Quantum RAG Evaluation") -> Any:
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        return datasets[0]

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Quantum Computing benchmark evaluation dataset for modular RAG pipeline."
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=QUANTUM_BENCHMARK_EXAMPLES,
    )
    return dataset


def run_evaluation_langsmith(
    dataset_name: str = "Quantum RAG Evaluation",
    experiment_prefix: str = "quantum-rag-eval",
) -> pd.DataFrame:
    client = Client(api_url=os.getenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com"))
    get_or_create_dataset(client, dataset_name=dataset_name)

    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix=experiment_prefix,
        metadata={"pipeline": "Modular FAISS + Groq RAG", "top_k": 3},
    )

    df = experiment_results.to_pandas()
    print_evaluation_summary(df)
    return df


def run_evaluation_local(examples: Optional[List[Dict]] = None) -> pd.DataFrame:
    if examples is None:
        examples = QUANTUM_BENCHMARK_EXAMPLES

    rows = []
    for ex in examples:
        q = ex["inputs"]["question"]
        ref_ans = ex["outputs"]["answer"]

        out = rag_bot(q)
        ans = out["answer"]
        docs = out.get("documents", [])

        try:
            score_correct = correctness(ex["inputs"], out, ex["outputs"])
        except Exception:
            score_correct = None

        try:
            score_rel = relevance(ex["inputs"], out)
        except Exception:
            score_rel = None

        try:
            score_ground = groundedness(ex["inputs"], out)
        except Exception:
            score_ground = None

        try:
            score_ret_rel = retrieval_relevance(ex["inputs"], out)
        except Exception:
            score_ret_rel = None

        rows.append({
            "question": q,
            "answer": ans,
            "reference_answer": ref_ans,
            "retrieved_chunks_count": len(docs),
            "correctness": score_correct,
            "relevance": score_rel,
            "groundedness": score_ground,
            "retrieval_relevance": score_ret_rel,
        })

    df = pd.DataFrame(rows)
    print_evaluation_summary(df)
    return df


def print_evaluation_summary(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("RAG EVALUATION METRICS REPORT")
    print("=" * 75)

    metric_cols = [
        c for c in [
            "correctness", "relevance", "groundedness", "retrieval_relevance",
            "feedback.correctness", "feedback.relevance", "feedback.groundedness", "feedback.retrieval_relevance"
        ] if c in df.columns
    ]
    
    for col in metric_cols:
        clean_name = col.replace("feedback.", "").replace("_", " ").title()
        valid = df[col].dropna()
        if len(valid) > 0:
            true_count = sum(1 for v in valid if v is True or str(v).lower() == "true")
            pct = (true_count / len(valid)) * 100
            print(f"- {clean_name:25s}: {pct:6.2f}% ({true_count}/{len(valid)} passed)")
        else:
            print(f"- {clean_name:25s}: N/A")

    print("-" * 75)

    q_col = "inputs.question" if "inputs.question" in df.columns else ("question" if "question" in df.columns else None)
    a_col = "outputs.answer" if "outputs.answer" in df.columns else ("answer" if "answer" in df.columns else None)

    display_cols = [c for c in [q_col, a_col] if c is not None] + metric_cols
    display_cols = list(dict.fromkeys(display_cols))
    if display_cols:
        display_df = df[display_cols].copy()
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda v: "None" if pd.isna(v) else ((str(v)[:90] + "...") if len(str(v)) > 90 else str(v))
            )
        print(display_df.to_string(index=False))

    print("=" * 75 + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode == "langsmith":
        run_evaluation_langsmith()
    else:
        run_evaluation_local()
