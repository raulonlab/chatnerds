import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM as LLMBase
from langchain.chains.base import Chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from chatnerds.langchain.prompt_factory import PromptFactory


DEFAULT_MAX_TOKENS = 128_000
MAX_DOCUMENTS_TO_PROCESS = 10

SCHEMA = {
    "properties": {
        "tags": {
            "type": "list",
            # "enum": [1, 2, 3, 4, 5],
            "description": "list of tags or keywords that classify the content of the document. The list is separated by commas.",
        },
        "category": {
            "type": "string",
            "description": "One word that describes the main category of the document",
        },
    },
    "required": ["tags", "category"],
}


DEFAULT_FIND_TAGS_PROMPT = "You are a classification and tagging assistant \
and your task is to find a list with the 5 most relevant tags of a textt. \
{format_instructions}. \
\
\n\Text:\n\
\
{text}"


class Tagger:
    config: Dict[str, Any] = {}
    chain: Chain = None  # BaseCombineDocumentsChain

    def __init__(
        self,
        config: Dict[str, Any],
        llm: LLMBase,
        prompt: str,
        prompt_type: Optional[str] = None,
        n_tags: int = 5,
    ):

        self.config = config or {}
        self.chain = self._get_csv_tag_chain(
            llm, prompt=prompt, prompt_type=prompt_type, n_tags=n_tags
        )

    def tag_documents(self, documents: List[Document]) -> str:
        # Split documents in chunks of size max_tokens or DEFAULT_MAX_TOKENS
        max_tokens = self.config.get("tag", {}).get("max_tokens", DEFAULT_MAX_TOKENS)

        try:
            max_tokens = int(max_tokens)
            if not 0 < max_tokens <= DEFAULT_MAX_TOKENS:
                max_tokens = DEFAULT_MAX_TOKENS
        except ValueError:
            max_tokens = DEFAULT_MAX_TOKENS

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "], chunk_size=max_tokens, chunk_overlap=0
        )

        splitted_documents = text_splitter.split_documents(documents)

        if len(splitted_documents) > MAX_DOCUMENTS_TO_PROCESS:
            logging.warning(
                f"Number of documents to tag cut to max allowed {MAX_DOCUMENTS_TO_PROCESS} (received {len(splitted_documents)})"
            )

            splitted_documents = splitted_documents[:MAX_DOCUMENTS_TO_PROCESS]

        page_contents = [doc.page_content for doc in splitted_documents]

        callbacks = [ConsoleCallbackHandler()]
        tags = self.chain.invoke(
            {"text": "\n".join(page_contents)}, config={"callbacks": callbacks}
        )

        return tags

    def tag_text(self, text: str) -> str:
        return self.tag_documents([Document(page_content=text)])

    def _get_csv_tag_chain(
        self,
        llm: LLMBase,
        prompt: str,
        prompt_type: str = None,
        n_tags: int = 5,
        format_instructions: str = None,
    ) -> Chain:
        output_parser = CommaSeparatedListOutputParser()

        if not format_instructions:
            format_instructions = output_parser.get_format_instructions()

        prompt = PromptFactory().format_prompt_template(
            llm=llm,
            prompt=prompt,
            prompt_type=prompt_type,
            input_variables=["text"],
            partial_variables={
                "n_tags": n_tags,
                "format_instructions": format_instructions,
            },
        )

        chain = prompt | llm | output_parser

        return chain
