from typing import Any, Callable, Dict, Optional
from rich import print
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import ConfigurableField
from langchain.chains import RetrievalQA
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.load import dumps, loads
from langchain_core.runnables import RunnableParallel
from langchain.runnables.hub import HubRunnable
from ..document_embeddings import DocumentEmbeddings
from ..chroma_database import ChromaDatabase
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.base import BaseCallbackHandler
from .llm_factory import LLMFactory

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question.
"""

# Resources:
# https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#using-in-a-chain
# https://python.langchain.com/docs/guides/local_llms#prompts
# https://python.langchain.com/docs/expression_language/cookbook/retrieval#with-memory-and-returning-source-documents

# More Resources:
# https://towardsdatascience.com/llms-for-everyone-running-the-llama-13b-model-and-langchain-in-google-colab-68d88021cf0b
# https://towardsdatascience.com/llms-for-everyone-running-langchain-and-a-mistralai-7b-model-in-google-colab-246ca94d7c4d

# More more Resources:
# https://medium.com/ai-insights-cobet/rag-and-parent-document-retrievers-making-sense-of-complex-contexts-with-code-5bd5c3474a8a
#   - Code: https://github.com/azharlabs/large-models/blob/main/notebooks/RAG/RagFusion_with_PaLM_2_Langchain.ipynb

class ChainFactory:
    config: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
    

    def get_rag_fusion_chain(self) -> Any:
        embeddings = DocumentEmbeddings(config=self.config).get_embeddings()

        chroma_database = ChromaDatabase(embeddings=embeddings, config=self.config["chroma"])
        retriever = chroma_database.client.as_retriever(**self.config["retriever"])
        # get llm model
        llm = LLMFactory(config=self.config).get_llm()

        generate_queries_prompt = ChatPromptTemplate(
              input_variables=['original_query'],
              messages=[
                  SystemMessagePromptTemplate(
                      prompt=PromptTemplate(input_variables=[],
                                            template='You are a helpful assistant that generates multiple search queries based on a single input query.')),
                  HumanMessagePromptTemplate(
                      prompt=PromptTemplate(input_variables=['original_query'],
                                            template='Generate 4 search queries separated by ; related to the query: {question}'))
                  ]
              )
        
        generate_queries_chain = (
            generate_queries_prompt | llm | StrOutputParser() | (lambda str: str.split(";")) | (lambda lst: list(filter(str.strip, lst)))
        )

        ragfusion_chain = generate_queries_chain | retriever.map() | self.reciprocal_rank_fusion

        system_prompt = self.config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        print(f"System Prompt: {system_prompt}")

        template = system_prompt + """
        Context:
        {context}

        Question:
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        full_rag_fusion_chain = (
            {
                "context": ragfusion_chain,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            # | StrOutputParser()
        )

        return full_rag_fusion_chain

    
    def get_rag_chain(self) -> Any:
        embeddings = DocumentEmbeddings(config=self.config).get_embeddings()

        chroma_database = ChromaDatabase(embeddings=embeddings, config=self.config["chroma"])
        retriever = chroma_database.client.as_retriever(**self.config["retriever"])
        # get llm model
        llm = LLMFactory(config=self.config).get_llm()

        # get the prompt template and memory if set by the user.
        system_prompt = self.config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        prompt_type = self.config.get("prompt_type", None)
        use_history = self.config.get("use_history", False)
        
        prompt = self.get_prompt_template(system_prompt=system_prompt, prompt_type=prompt_type, history=use_history)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    def get_prompt_template_from_hub(self):
        prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
            owner_repo_commit=ConfigurableField(
                id="hub_commit",
                name="Hub Commit",
                description="The Hub commit to pull from",
            )
        )
        return prompt.with_config(configurable={"hub_commit": "rlm/rag-prompt-llama"})


    @staticmethod
    def get_prompt_template(system_prompt=DEFAULT_SYSTEM_PROMPT, prompt_type=None, history=False):
        if prompt_type == "llama":
            # https://smith.langchain.com/hub/rlm/rag-prompt-llama
            prompt_template = (
                "[INST]<<SYS>> "
                + system_prompt
                + " <</SYS>>" 
                + "Question: {question}\n"
                + "Context: {context}\n"
                + "Answer: [/INST]"
            )

            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        
        elif prompt_type == "mistral":
            # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            prompt_template = (
                "<s>"
                + "[INST] "
                + system_prompt
                + " [/INST] "
                + "</s>"
                + "[INST] "
                + "Question: {question}\n"
                + "Context: {context}\n"
                + "Answer: [/INST]"
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            
        else:
            prompt_template = (
                system_prompt
                + "/n"
                + "Question: {question}\n"
                + "Context: {context}\n"
                + "Answer:"
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        return prompt

        # memory = ConversationBufferMemory(input_key="question", memory_key="history") # , return_messages=True

        # return (
        #     prompt,
        #     memory,
        # )

    @staticmethod
    def reciprocal_rank_fusion(results: list[list], k=20):
        fused_scores = {}
        for docs in results:
            # Assumes the docs are returned in sorted order of relevance
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        print("----------------------------------\n")
        print(f"Reranked Results: \n{reranked_results}")
        print("----------------------------------\n")
        return reranked_results
