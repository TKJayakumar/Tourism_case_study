from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma


class RetrieveManager:
    """
    Member 3 – Hybrid Retrieval (BM25 Sparse + Chroma Dense via EnsembleRetriever).
    Returns top hybrid candidates so Member 4 (`MeridianReranker`) can perform
    cross-encoder reranking and date-aware policy filtering.
    """

    def query_hybrid(
        self,
        query: str,
        documents: list[Document],
        vector_store: Chroma,
        n_results: int = 10,
        candidate_results: int = 10,
    ) -> list[Document]:
        if not documents:
            return []

        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": candidate_results},
        )

        bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=candidate_results,
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5],
        )

        results = ensemble_retriever.invoke(query)
        return results[:n_results]