from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.runnables import chain, Runnable
from langchain_core.load import dumps, loads
from langchain.llms.base import LLM as LLMBase
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.chroma import Chroma
from chatnerds.langchain.prompt_factory import PromptFactory
from chatnerds.constants import DEFAULT_QUERY_EXPANSION_PROMPT


# Generate similar questions from original query using LLM
# Source: https://levelup.gitconnected.com/3-query-expansion-methods-implemented-using-langchain-to-improve-your-rag-81078c1330cd
@chain
def question_expansion_runnable(
    original_question: Dict | str,
    llm: LLMBase,
    prompt_type: str = None,
    n_expanded_questions: int = 0,
    **kwargs,
) -> Runnable:
    if isinstance(original_question, str):
        question = original_question
    elif isinstance(original_question, Dict) and "question" in original_question:
        question = original_question["question"]
    else:
        raise ValueError(
            f"Invalid input for question_expansion_runnable: {original_question}"
        )

    if not n_expanded_questions or n_expanded_questions < 1:
        return [question]

    query_expansion_system_prompt: str = DEFAULT_QUERY_EXPANSION_PROMPT.format(
        n_expanded_questions=n_expanded_questions
    )

    query_expansion_prompt = PromptFactory().get_query_expansion_prompt(
        llm=llm, system_prompt=query_expansion_system_prompt, prompt_type=prompt_type
    )

    rag_chain = query_expansion_prompt | llm | StrOutputParser()

    question_string = rag_chain.invoke(
        {"question": question},
        # config={'callbacks': [ConsoleCallbackHandler()]}
    )

    lines_list = question_string.splitlines()
    questions = []
    questions = [question] + [
        line.lstrip("1234567890. ")
        for line in lines_list
        if line.endswith("?") and line != question
    ]

    return questions


@chain
def retrieve_relevant_documents_runnable(
    questions: List[str] | str, retriever: VectorStoreRetriever
) -> Runnable:
    if isinstance(questions, str):
        questions = [questions]

    retriever_k = retriever.search_kwargs.get("k", 4)

    retrieved_documents: List[Document] = []
    for i in questions:
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

    return unique_documents


# Cross Encoding happens in here
@chain
def rerank_documents_runnable(input: Dict) -> Runnable:

    question: str = input.get("question", None)
    documents: List[Document] = input.get("documents", [])

    if len(documents) == 0:
        return []

    pairs = []
    for doc in documents:
        pairs.append([question, doc.page_content])

    # Cross Encoder Scoring
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
    documents: List[Document], parents_db_client: Chroma, retriever_k: int
) -> Runnable:

    if len(documents) == 0:
        return []

    # Get parent documents
    parent_ids = []
    for doc in documents:
        parent_id = doc.metadata.get("parent_id", None)
        if parent_id and parent_id not in parent_ids:
            parent_ids.append(parent_id)

        if len(parent_ids) == retriever_k:
            break

    parent_result = parents_db_client.get(ids=parent_ids)

    result_documents = []
    for index in range(len(parent_result["documents"])):
        result_documents.append(
            Document(
                page_content=parent_result["documents"][index],
                metadata=parent_result["metadatas"][index],
            )
        )

    return result_documents[:retriever_k]


@chain
def combine_documents_runnable(documents: List[Document]) -> Runnable:
    if len(documents) == 0:
        return ""

    page_contents = [document.page_content for document in documents]
    return "\n\n".join(page_contents)


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
    return reranked_results
