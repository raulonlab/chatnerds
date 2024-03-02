from typing import Any, Dict
import logging
from operator import itemgetter
from langchain.llms.base import LLM as LLMBase
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.base import Chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableParallel
from chatnerds.langchain.document_embeddings import DocumentEmbeddings
from chatnerds.langchain.chroma_database import (
    ChromaDatabase,
    DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
)
from chatnerds.langchain.llm_factory import LLMFactory
from chatnerds.langchain.chain_runnables import (
    question_expansion_runnable,
    retrieve_relevant_documents_runnable,
    rerank_documents_runnable,
    get_parent_documents_runnable,
    combine_documents_runnable,
)
from chatnerds.langchain.prompt_factory import PromptFactory
from chatnerds.constants import DEFAULT_CHAT_SYSTEM_PROMPT


class ChainFactory:
    config: Dict[str, Any] = {}
    llm: LLMBase = None
    retriever: VectorStoreRetriever = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # Source: https://levelup.gitconnected.com/3-query-expansion-methods-implemented-using-langchain-to-improve-your-rag-81078c1330cd
    def get_rag_fusion_chain(self) -> Chain:
        embeddings = DocumentEmbeddings(config=self.config).get_embedding_function()

        chroma_database = ChromaDatabase(
            embeddings=embeddings, config=self.config["chroma"]
        )
        parent_database = ChromaDatabase(
            collection_name=DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
            embeddings=embeddings,
            config=self.config["chroma"],
        )
        self.retriever = chroma_database.client.as_retriever(**self.config["retriever"])
        retriever_k = self.retriever.search_kwargs.get("k", 4)

        self.llm, prompt_type = LLMFactory(config=self.config).get_llm()

        system_prompt: str = self.config.get(
            "system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT
        )
        system_prompt = system_prompt.replace("\n", " ").strip(". ")

        qa_prompt = PromptFactory(self.config).get_rag_prompt(
            llm=self.llm, system_prompt=system_prompt, prompt_type=prompt_type
        )

        n_expanded_questions: int = self.config["chain"].get("n_expanded_questions", 0)

        retrieve_relevant_documents = RunnableParallel(
            documents={
                "question": RunnablePassthrough(),
                "documents": question_expansion_runnable.bind(
                    llm=self.llm,
                    prompt_type=prompt_type,
                    n_expanded_questions=n_expanded_questions,
                )
                | retrieve_relevant_documents_runnable.bind(
                    retriever=self.retriever,
                ),
            }
            | rerank_documents_runnable
            | get_parent_documents_runnable.bind(
                parents_db_client=parent_database.client, retriever_k=retriever_k
            ),
            question=RunnablePassthrough(),
        )

        combine_documents_in_context = RunnableParallel(
            context=itemgetter("documents") | combine_documents_runnable,
            question=itemgetter("question"),
            documents=itemgetter("documents"),
        )

        get_results = RunnableParallel(
            result=qa_prompt | self.llm | StrOutputParser(),
            source_documents=itemgetter("documents"),
        )

        chain = retrieve_relevant_documents | combine_documents_in_context | get_results

        return chain

    def get_prompt_test_chain(self) -> Chain:
        embeddings = DocumentEmbeddings(config=self.config).get_embedding_function()

        chroma_database = ChromaDatabase(
            embeddings=embeddings, config=self.config["chroma"]
        )
        self.retriever = chroma_database.client.as_retriever(**self.config["retriever"])
        # get llm model
        self.llm, prompt_type = LLMFactory(config=self.config).get_llm()

        system_prompt: str = self.config.get(
            "system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT
        )
        system_prompt = system_prompt.replace("\n", " ").strip(". ")

        qa_prompt = PromptFactory(self.config).get_rag_prompt(
            llm=self.llm, system_prompt=system_prompt, prompt_type=prompt_type
        )

        chain = (
            {
                "context": lambda x: "This is a test context",
                "question": RunnablePassthrough(),
            }
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        logging.debug(
            f"get_prompt_test_chain: input_schema:\n{chain.input_schema.schema()}"
        )
        logging.debug(
            f"get_prompt_test_chain: chain.get_prompts():\n{chain.get_prompts()}"
        )

        return chain

    def get_retrieval_qa_chain(self) -> Chain:
        embeddings = DocumentEmbeddings(config=self.config).get_embedding_function()

        chroma_database = ChromaDatabase(
            embeddings=embeddings, config=self.config["chroma"]
        )
        retriever = chroma_database.client.as_retriever(**self.config["retriever"])
        llm = LLMFactory(config=self.config).get_llm()

        system_prompt = self.config.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT)

        template = (
            system_prompt
            + """
            Context:
            {context}

            Question: {question}"""
        )

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": False,
                "prompt": prompt,
                # "memory": ConversationBufferMemory(
                #     memory_key="history",
                #     input_key="question"),
            },
        )

        return qa
