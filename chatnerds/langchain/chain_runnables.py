from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.runnables import chain, Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores.chroma import Chroma
from chatnerds.stores.store_factory import StoreFactory

DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@chain
def retrieve_relevant_documents_runnable(
    questions: List[str] | str, retriever: VectorStoreRetriever, **kwargs
) -> Runnable:
    if isinstance(questions, str):
        questions = [questions]

    retrieved_documents: List[Document] = []
    for i in questions:
        results = retriever.get_relevant_documents(i)
        retrieved_documents.extend(results)

    # Remove duplicates
    unique_ids = set()
    unique_documents = [
        doc
        for doc in retrieved_documents
        if doc.page_content not in unique_ids
        and (unique_ids.add(doc.page_content) or True)
    ]

    return unique_documents


# Cross Encoding happens in here
@chain
def rerank_documents_runnable(
    input: Dict,
    use_cross_encoding_rerank: bool = True,
    model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    **kwargs,
) -> Runnable:

    question: str = input.get("question", None)
    documents: List[Document] = input.get("documents", [])

    if not use_cross_encoding_rerank:
        return documents

    if len(documents) == 0:
        return []

    pairs = []
    for doc in documents:
        pairs.append([question, doc.page_content])

    # Cross Encoder Scoring
    cross_encoder = CrossEncoder(model_name, **kwargs)
    scores = cross_encoder.predict(pairs)

    # Add score to metadata
    for x in range(len(scores)):
        documents[x].metadata["score"] = scores[x]

    # Rerank the documents
    sorted_documents = sorted(
        documents, key=lambda x: x.metadata["score"], reverse=True
    )

    return sorted_documents


@chain
def get_parent_documents_runnable(
    documents: List[Document],
    store: Chroma,
    n_combined_documents: int,
    **kwargs,
) -> Runnable:
    if len(documents) == 0:
        return []

    result_documents = []
    for doc in documents:
        source = doc.metadata.get("source", None)
        start_index = doc.metadata.get("start_index", None)

        if not source or not start_index:
            continue

        try:
            start_index = int(start_index)
        except ValueError:
            continue

        sibblings_data = store.get(
            include=["metadatas", "documents"], where={"source": source}
        )

        sibbling_documents = []
        for i in range(len(sibblings_data["metadatas"])):
            sibbling_documents.append(
                {
                    "id": sibblings_data["ids"][i],
                    "page_content": sibblings_data["documents"][i],
                    "metadata": sibblings_data["metadatas"][i],
                }
            )

        sibbling_documents = sorted(
            sibbling_documents, key=lambda x: int(x["metadata"].get("start_index", 0))
        )

        for sibling_i, sibling in enumerate(sibbling_documents):
            sibling_start_index = sibling["metadata"].get("start_index", None)
            sibling_start_index = (
                int(sibling_start_index) if sibling_start_index else None
            )

            if sibling_start_index == start_index:
                prev_index = max(0, sibling_i - 1)
                next_index = min(len(sibbling_documents) - 1, sibling_i + 1)

                parent_page_content = ""
                for i in range(prev_index, next_index + 1):
                    parent_page_content = (
                        parent_page_content
                        + "\n"
                        + sibbling_documents[i]["page_content"]
                    )

                result_documents.append(
                    Document(
                        page_content=parent_page_content,
                        metadata=sibling["metadata"],
                    )
                )

                if len(result_documents) == n_combined_documents:
                    return result_documents
                else:
                    break

    return result_documents


@chain
def get_source_documents_runnable(
    documents: List[Document],
    store_factory: StoreFactory,
    n_combined_documents: int,
    **kwargs,
) -> Runnable:

    if len(documents) == 0:
        return []

    unique_result_sources = set()
    result_documents = []
    for doc in documents:
        source = doc.metadata.get("source", None)
        if not source:
            continue
        if source in unique_result_sources:
            continue

        with store_factory.get_status_store() as status_store:
            studied_document_data = status_store.get_studied_document(source)

        if studied_document_data:
            result_documents.append(
                Document(
                    page_content=studied_document_data["page_content"],
                    metadata=studied_document_data["metadata"],
                )
            )
            unique_result_sources.add(source)

            if len(result_documents) == n_combined_documents:
                return result_documents

    return result_documents


@chain
def combine_documents_runnable(documents: List[Document]) -> Runnable:
    if len(documents) == 0:
        return ""

    page_contents = [document.page_content for document in documents]
    return "\n\n".join(page_contents)


# @chain
# def reciprocal_rank_fusion(results: list[list], k=20) -> Runnable:
#     fused_scores = {}
#     for docs in results:
#         # Assumes the docs are returned in sorted order of relevance
#         for rank, doc in enumerate(docs):
#             doc_str = dumps(doc)
#             if doc_str not in fused_scores:
#                 fused_scores[doc_str] = 0
#             fused_scores[doc_str]
#             fused_scores[doc_str] += 1 / (rank + k)

#     reranked_results = [
#         (loads(doc), score)
#         for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#     ]
#     return reranked_results
