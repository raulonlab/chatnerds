import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM as LLMBase
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableLambda
from chatnerds.langchain.prompt_factory import PromptFactory

DEFAULT_MAX_TOKENS = 128_000
MAX_DOCUMENTS_TO_SUMMARIZE = 10


class Summarizer:
    config: Dict[str, Any] = {}
    summarize_chain: Chain = None  # BaseCombineDocumentsChain

    def __init__(
        self,
        config: Dict[str, Any],
        llm: LLMBase,
        prompt_type: Optional[str] = None,
    ):

        self.config = config or {}
        summarize_chain_type = self.config.get("summarize", {}).get(
            "summarize_chain_type", None
        )

        if summarize_chain_type == "map_reduce":
            self.summarize_chain = self._get_mapreduce_summarize_chain(
                llm, prompt_type=prompt_type
            )
        elif summarize_chain_type == "refine":
            self.summarize_chain = self._get_refine_summarize_chain(
                llm, prompt_type=prompt_type
            )
        else:
            self.summarize_chain = self._get_stuff_summarize_chain(
                llm, prompt_type=prompt_type
            )

    def get_chain(
        self, input_documents_key: Optional[str] = None, return_output_text: bool = True
    ) -> Chain:
        chain = self.summarize_chain

        if input_documents_key:
            chain = (
                RunnableLambda(lambda input: input.get(input_documents_key, [])) | chain
            )

        if return_output_text:
            chain = chain | RunnableLambda(
                lambda output: output.get("output_text", None)
            )

        return chain

    def summarize_documents(self, documents: List[Document]) -> str:
        # Split documents in chunks of size max_tokens or DEFAULT_MAX_TOKENS
        max_tokens = self.config.get("summarize", {}).get(
            "max_tokens", DEFAULT_MAX_TOKENS
        )

        try:
            max_tokens = int(max_tokens)
            if not 0 < max_tokens <= DEFAULT_MAX_TOKENS:
                max_tokens = DEFAULT_MAX_TOKENS
        except ValueError:
            max_tokens = DEFAULT_MAX_TOKENS

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "], chunk_size=max_tokens, chunk_overlap=0
        )  # separators=["\n\n", "\n", ".", " "], chunk_size=max_tokens, chunk_overlap=0
        splitted_documents = text_splitter.split_documents(documents)

        if len(splitted_documents) > MAX_DOCUMENTS_TO_SUMMARIZE:
            logging.warning(
                f"Number of documents to summarize cut to max allowed {MAX_DOCUMENTS_TO_SUMMARIZE} (received {len(splitted_documents)})"
            )

            splitted_documents = splitted_documents[:MAX_DOCUMENTS_TO_SUMMARIZE]

        summary = self.summarize_chain.invoke(splitted_documents)

        if isinstance(summary, str):
            return summary
        elif isinstance(summary, dict) and "output_text" in summary:
            return summary["output_text"]
        else:
            return str(summary)

    def summarize_text(self, text: str) -> str:
        return self.summarize_documents([Document(page_content=text)])

    def _get_mapreduce_summarize_chain(
        self, llm: LLMBase, prompt_type: str = None
    ) -> Chain:
        summary_system_prompt = self.config["prompts"].get(
            "summary_system_prompt", None
        )
        summary_human_prompt: str = self.config["prompts"].get(
            "summary_human_prompt", None
        )
        combine_system_prompt = self.config["prompts"].get(
            "combine_system_prompt", None
        )
        combine_human_prompt: str = self.config["prompts"].get(
            "combine_human_prompt", None
        )

        if summary_system_prompt:
            map_prompt = PromptFactory.build_prompt_template(
                system_prompt=summary_system_prompt,
                human_prompt=summary_human_prompt or "{text}",
                prompt_type=prompt_type,
                input_variables=["text"],
            )
        else:
            intro = "The following is a set of documents"

            map_template = (
                intro
                + """

            {text}

            Based on this list of docs, please write a concise summary. 
            Helpful Answer:"""
            )

            map_prompt = PromptTemplate.from_template(map_template)

        if combine_system_prompt:
            combine_prompt = PromptFactory.build_prompt_template(
                system_prompt=combine_system_prompt,
                human_prompt=combine_human_prompt or "{text}",
                prompt_type=prompt_type,
                input_variables=["text"],
            )

        else:
            combine_template = """The following is a set of summaries:

            {text}

            Take these and distill it into a final, consolidated list of the main topics and themes. 
            Return that list as a comma separated list. 
            Helpful Answer:"""

            combine_prompt = PromptTemplate.from_template(combine_template)

        # Get your chain ready to use
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            map_reduce_document_variable_name="text",
            combine_document_variable_name="text",
            verbose=False,
            # token_max=3000,
            return_intermediate_steps=False,
        )  # verbose=True optional to see what is getting sent to the LLM

        return chain

    def _get_refine_summarize_chain(
        self, llm: LLMBase, prompt_type: str = None
    ) -> Chain:
        summary_system_prompt = self.config["prompts"].get(
            "summary_system_prompt", None
        )
        summary_human_prompt: str = self.config["prompts"].get(
            "summary_human_prompt", None
        )
        refine_system_prompt = self.config["prompts"].get("refine_system_prompt", None)
        refine_human_prompt: str = self.config["prompts"].get(
            "refine_human_prompt", None
        )

        if summary_system_prompt:
            initial_prompt = PromptFactory.build_prompt_template(
                system_prompt=summary_system_prompt,
                human_prompt=summary_human_prompt or "{text}",
                prompt_type=prompt_type,
                input_variables=["text"],
            )
        else:
            initial_template = """
            Extract the most relevant themes from the following:

            "{text}"

            THEMES:"""

            initial_prompt = PromptTemplate.from_template(initial_template)

        if refine_system_prompt:
            refine_prompt = PromptFactory.build_prompt_template(
                system_prompt=refine_system_prompt,
                human_prompt=refine_human_prompt or "{text}",
                prompt_type=prompt_type,
                input_variables=["text"],
            )
        else:
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

            refine_prompt = PromptTemplate.from_template(refine_template)

        # Get your chain ready to use
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=initial_prompt,
            refine_prompt=refine_prompt,
            document_variable_name="text",
            initial_response_name="existing_answer",
            verbose=False,
            return_intermediate_steps=False,
        )  # verbose=True optional to see what is getting sent to the LLM

        return chain

    def _get_stuff_summarize_chain(
        self, llm: LLMBase, prompt_type: str = None
    ) -> Chain:
        summary_system_prompt = self.config["prompts"].get(
            "summary_system_prompt", None
        )
        summary_human_prompt: str = self.config["prompts"].get(
            "summary_human_prompt", None
        )

        if summary_system_prompt:
            summarize_prompt = PromptFactory.build_prompt_template(
                system_prompt=summary_system_prompt,
                human_prompt=summary_human_prompt or "{text}",
                prompt_type=prompt_type,
                input_variables=["text"],
            )
        else:
            template = """
            Write a concise summary of the following text:

            "{text}"

            CONCISE SUMMARY OF TEXT:
            """

            summarize_prompt = PromptTemplate.from_template(template)

        # Get your chain ready to use
        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=summarize_prompt,
            verbose=True,
            document_variable_name="text",
            return_intermediate_steps=False,
        )  # verbose=True optional to see what is getting sent to the LLM

        return chain
