from typing import Any, Dict
import logging
from operator import itemgetter
from langchain.llms.base import LLM as LLMBase
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableParallel
from chatnerds.langchain.document_embeddings import DocumentEmbeddings
from chatnerds.langchain.chroma_database import ChromaDatabase
from chatnerds.langchain.llm_factory import LLMFactory
from chatnerds.langchain.chain_runnables import (
    query_expansion_runnable,
    retrieve_best_documents_runnable_v1,
    combine_documents_for_context_runnable,
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
    def get_rag_fusion_chain(self) -> Any:
        embeddings = DocumentEmbeddings(config=self.config).get_embedding_function()

        chroma_database = ChromaDatabase(
            embeddings=embeddings, config=self.config["chroma"]
        )
        self.retriever = chroma_database.client.as_retriever(**self.config["retriever"])
        self.llm, prompt_type = LLMFactory(config=self.config).get_llm()

        system_prompt: str = self.config.get(
            "system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT
        )
        system_prompt = system_prompt.replace("\n", " ").strip(". ")

        qa_prompt = PromptFactory(self.config).get_rag_prompt(
            llm=self.llm, system_prompt=system_prompt, prompt_type=prompt_type
        )

        n_expanded_queries: int = self.config["chain"].get("n_expanded_queries", 0)

        # https://python.langchain.com/docs/expression_language/cookbook/retrieval
        retrieved_documents = {
            "documents": query_expansion_runnable.bind(
                llm=self.llm,
                prompt_type=prompt_type,
                n_expanded_queries=n_expanded_queries,
            )
            | retrieve_best_documents_runnable_v1.bind(retriever=self.retriever),
            "question": RunnablePassthrough(),
        }

        prompt_inputs = {
            "context": itemgetter("documents") | combine_documents_for_context_runnable,
            "question": itemgetter("question"),
        }

        results = RunnableParallel(
            {
                "result": prompt_inputs | qa_prompt | self.llm | StrOutputParser(),
                "source_documents": itemgetter("documents"),
            }
        )

        chain = retrieved_documents | results

        return chain

    def get_prompt_test_chain(self) -> Any:
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

    def get_retrieval_qa_chain(self) -> Any:
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
