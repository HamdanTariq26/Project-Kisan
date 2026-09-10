import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from .config import LOCAL_FAISS_PATH, get_huggingface_secret


secret_value = get_huggingface_secret()

embedding_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device": "cuda",
        "trust_remote_code": True,
        "model_kwargs": {
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
        },
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 4,
    },
)

vector_db = FAISS.load_local(
    folder_path=LOCAL_FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)

print("Loaded existing FAISS index from disk successfully...!")


reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    trust_remote_code=True,
    token=secret_value,
)


def retrieve_and_rerank(
    query: str,
    candidate_k: int = 20,
    final_k: int = 5,
    min_rerank_score: float = 0.5,
):
    """
    Retrieve candidate documents using FAISS, rerank them with a CrossEncoder,
    and discard results that are not sufficiently relevant.
    """

    candidates = vector_db.similarity_search_with_score(
        query,
        k=candidate_k,
    )

    if not candidates:
        return []

    pairs = [
        [query, doc.page_content]
        for doc, _ in candidates
    ]

    reranker_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidates, reranker_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    relevant_docs = [
        doc
        for (doc, _faiss_score), rerank_score in ranked_results
        if rerank_score >= min_rerank_score
    ]

    return relevant_docs[:final_k]


def rag_search(query):
    docs = retrieve_and_rerank(
        query,
        candidate_k=20,
        final_k=5,
        min_rerank_score=0.5,
    )

    return {
        "status": "rag_results_found" if docs else "no_results",
        "results": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get(
                    "page",
                    doc.metadata.get("page_number", "N/A"),
                ),
            }
            for doc in docs
        ],
    }
