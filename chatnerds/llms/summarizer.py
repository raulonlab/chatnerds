from typing import Any, Dict, List
import logging
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM as LLMBase
from langchain.chains.summarize import load_summarize_chain
from .llm_factory import LLMFactory


class Summarizer:
    config: Dict[str, Any] = {}
    llm: LLMBase = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = LLMFactory(config).get_llm()
    

    def summarize_documents(self, docs: List[Document], split: bool = False, source_title: str = None) -> str:
        if (split):
            text_splitter = RecursiveCharacterTextSplitter()    # separators=["\n\n", "\n", ".", " "], chunk_size=1800, chunk_overlap=200
            docs = text_splitter.split_documents(docs)

        if (self.config.get("summarize_chain_type") == "map_reduce"):
            summary = self._summarize_docs_with_mapreduce(self.llm, docs, source_title=source_title)
        elif (self.config.get("summarize_chain_type") == "refine"):
            summary = self._summarize_docs_with_refine(self.llm, docs, source_title=source_title)
        else:
            summary = self._summarize_docs_with_stuff(self.llm, docs, source_title=source_title)
        
        # print("summarize_chain_type: ", self.config.get("summarize_chain_type"))
        # print("summary:\n", summary)
        
        return summary


    def summarize_text(self, text: str, source_title: str = None) -> str:
        return self.summarize_documents(docs=[Document(page_content=text)], split=True, source_title=source_title)


    # Sure, here's the list of main topics and themes:
    #     first date tips, self-confidence, body language, respect, attraction, comfort, spark, relationship maintenance.
    @staticmethod
    def _summarize_docs_with_mapreduce(llm: LLMBase, docs: List[Document], source_title: str = None) -> str:
        intro = "The following is a set of documents"

        if (source_title):
            intro = f"{intro} with the title '{source_title}'"

        map_template = intro + """

        {text}

        Based on this list of docs, please write a concise summary. 
        Helpful Answer:"""

        combine_template = """The following is a set of summaries:

        {text}

        Take these and distill it into a final, consolidated list of the main topics and themes. 
        Return that list as a comma separated list. 
        Helpful Answer:"""

        map_prompt = PromptTemplate.from_template(map_template)
        combine_prompt = PromptTemplate.from_template(combine_template)

        # Get your chain ready to use
        chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
            ) # verbose=True optional to see what is getting sent to the LLM
        
        output = chain.run(docs)

        return output


    @staticmethod
    def _summarize_docs_with_refine(llm: LLMBase, docs: List[Document], source_title: str = None) -> str:
        initial_template = """
        Extract the most relevant themes from the following:

        "{text}"

        THEMES:"""

        refine_template = """
        Your job is to extract the most relevant themes
        We have provided an existing list of themes up to a certain point: {existing_answer}
        We have the opportunity to refine the existing list(only if needed) with some more context below.
        ------------
        {text}
        ------------
        Given the new context, refine the original list
        If the context isn't useful, return the original list and ONLY the original list.
        Return that list as a comma separated list.

        LIST:"""

        initial_prompt = PromptTemplate.from_template(initial_template)
        refine_prompt = PromptTemplate.from_template(refine_template)

        # Get your chain ready to use
        chain = load_summarize_chain(
            llm=llm,
            chain_type='refine',
            question_prompt=initial_prompt,
            refine_prompt=refine_prompt,
            verbose=False
            ) # verbose=True optional to see what is getting sent to the LLM
        
        output = chain.run(docs)

        return output


    @staticmethod
    def _summarize_docs_with_stuff(llm: LLMBase, docs: List[Document], source_title: str = None) -> str:
        template = """
        Write a concise summary of the following in GERMAN:

        "{text}"

        CONCISE SUMMARY IN GERMAN:
        """

        prompt = PromptTemplate.from_template(template)

        # Get your chain ready to use
        chain = load_summarize_chain(
            llm=llm,
            chain_type='stuff',
            prompt=prompt,
            verbose=False
            ) # verbose=True optional to see what is getting sent to the LLM
        
        output = chain.run(docs)

        return output

