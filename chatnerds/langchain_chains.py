"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""
from typing import Any, Callable, Dict, Optional
from rich import print
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from .document_embeddings import DocumentEmbeddings
from .chroma_database import ChromaDatabase
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.base import BaseCallbackHandler
from .llms.llm_factory import LLMFactory

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""


def get_prompt_template(system_prompt=DEFAULT_SYSTEM_PROMPT, prompt_type=None, history=False):
    if prompt_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    elif prompt_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history") # , return_messages=True

    return (
        prompt,
        memory,
    )


def get_retrieval_qa(
    config: Dict[str, Any],
    *,
    callback: Optional[Callable[[str], None]] = None,
) -> RetrievalQA:
    embeddings = DocumentEmbeddings(config=config).get_embeddings()

    chroma_database = ChromaDatabase(embeddings=embeddings, config=config["chroma"])
    retriever = chroma_database.client.as_retriever(**config["retriever"])

    # get the prompt template and memory if set by the user.
    system_prompt = config.get("system_prompt", None)
    prompt_type = config.get("prompt_type", None)
    use_history = config.get("use_history", False)
    
    chain_type_kwargs = {}
    if system_prompt:
        prompt, memory = get_prompt_template(system_prompt=system_prompt, prompt_type=prompt_type, history=use_history)
        chain_type_kwargs["prompt"] = prompt
        if use_history:
            chain_type_kwargs["memory"] = memory
    
    # print("get_retrieval_qa ******")
    # print("system_prompt:", system_prompt)
    # print("prompt_type:", prompt_type)
    # print("use_history:", use_history)
    # print("chain_type_kwargs:", chain_type_kwargs)

    # get llm model
    llm = LLMFactory(config, callback).get_llm()

    class CallbackHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            callback(token)

    callbacks = [CallbackHandler()] if callback else None

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        callbacks=callbacks,
        chain_type_kwargs=chain_type_kwargs,
    )

    return qa
