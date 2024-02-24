from typing import Dict, List
import logging
from langchain_core.documents import Document
from langchain_core.runnables import chain, Runnable
from langchain_core.load import dumps, loads
from langchain.llms.base import LLM as LLMBase
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder
from langchain_core.output_parsers import StrOutputParser
from chatnerds.langchain.prompt_factory import PromptFactory
from chatnerds.constants import DEFAULT_QUERY_EXPANSION_PROMPT


# Generate similar queries from original query using LLM
# Source: https://levelup.gitconnected.com/3-query-expansion-methods-implemented-using-langchain-to-improve-your-rag-81078c1330cd
@chain
def query_expansion_runnable(
    original_query: Dict | str,
    llm: LLMBase,
    prompt_type: str = None,
    n_expanded_queries: int = 0,
    **kwargs,
) -> Runnable:
    if isinstance(original_query, str):
        query = original_query
    elif isinstance(original_query, Dict) and "question" in original_query:
        query = original_query["question"]
    else:
        raise ValueError(
            f"Invalid input for query_expansion_runnable: {original_query}"
        )

    if not n_expanded_queries or n_expanded_queries < 1:
        return [query]

    query_expansion_system_prompt: str = DEFAULT_QUERY_EXPANSION_PROMPT.format(
        n_expanded_queries=n_expanded_queries
    )

    query_expansion_prompt = PromptFactory().get_query_expansion_prompt(
        llm=llm, system_prompt=query_expansion_system_prompt, prompt_type=prompt_type
    )

    rag_chain = query_expansion_prompt | llm | StrOutputParser()

    question_string = rag_chain.invoke(
        {"question": query},
        # config={'callbacks': [ConsoleCallbackHandler()]}
    )

    lines_list = question_string.splitlines()
    queries = []
    queries = [query] + [
        line.strip("1234567890. ")
        for line in lines_list
        if line.endswith("?") and line != query
    ]

    logging.debug(f"query_expansion_runnable: Alternative queries:\n{queries}")

    return queries


# Use only retriever configuration. Get the best k documents from the input queries
@chain
def retrieve_best_documents_runnable_v2(
    queries: List[str], retriever: VectorStoreRetriever
) -> Runnable:
    retriever_k = retriever.search_kwargs.get("k", 4)

    retrieved_documents_by_query: List[List[Document]] = []
    for i in queries:
        # retriever.search_kwargs["k"] = 10 * retriever_k  # increase k for better results
        results = retriever.get_relevant_documents(i)
        # retriever.search_kwargs["k"] = retriever_k  # reset k

        retrieved_documents_by_query.append(results)

    retrieved_documents: List[Document] = []
    while len(retrieved_documents) < retriever_k:
        for query_retrieved_documents in retrieved_documents_by_query:
            retrieved_documents.append(query_retrieved_documents.pop(0))

    # Remove duplicates
    unique_ids = set()
    unique_documents = [
        doc
        for doc in retrieved_documents
        if doc.page_content not in unique_ids
        and (unique_ids.add(doc.page_content) or True)
    ]

    print(
        f"retrieve_best_documents_runnable_v2: There are {len(retrieved_documents)} documents retrieved."
    )
    print(
        f"retrieve_best_documents_runnable_v2: There are {len(unique_documents)} UNIQUE documents retrieved."
    )

    # return top k
    # return unique_documents[:retriever_k]
    return unique_documents


# Cross Encoding happens in here
@chain
def retrieve_best_documents_runnable_v1(
    queries: List[str], retriever: VectorStoreRetriever
) -> Runnable:
    retriever_k = retriever.search_kwargs.get("k", 4)

    retrieved_documents: List[Document] = []
    for i in queries:
        retriever.search_kwargs["k"] = 20 * retriever_k  # increase k for better results
        results = retriever.get_relevant_documents(i)
        retriever.search_kwargs["k"] = retriever_k  # reset k

        retrieved_documents.extend(results)

    # Remove duplicates
    unique_ids = set()
    unique_documents = [
        doc
        for doc in retrieved_documents
        if doc.page_content not in unique_ids
        and (unique_ids.add(doc.page_content) or True)
    ]

    print(
        f"retrieve_best_documents_runnable_v1: There are {len(retrieved_documents)} documents retrieved."
    )
    print(
        f"retrieve_best_documents_runnable_v1: There are {len(unique_documents)} UNIQUE documents retrieved."
    )

    pairs = []
    for doc in unique_documents:
        pairs.append([queries[0], doc.page_content])

    # Cross Encoder Scoring
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = cross_encoder.predict(pairs)
    logging.debug(f"retrieve_best_documents_runnable_v1: len(scores): {len(scores)}")

    # Add score to metadata
    for x in range(len(scores)):
        unique_documents[x].metadata["score"] = scores[x]

    # Rerank the documents
    sorted_documents = sorted(
        unique_documents, key=lambda x: x.metadata["score"], reverse=True
    )

    # return top k
    return sorted_documents[:retriever_k]


@chain
def combine_documents_for_context_runnable(docs: List[Document]) -> Runnable:
    doc_page_contents = [doc.page_content for doc in docs]
    return "\n\n".join(doc_page_contents)


@chain
def reciprocal_rank_fusion(results: list[list], k=20) -> Runnable:
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    logging.debug(f"reciprocal_rank_fusion:results: \n{reranked_results}")
    return reranked_results
