import re
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM as LLMBase
from langchain.chains.base import Chain
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.tracers.stdout import ConsoleCallbackHandler
import hdbscan
import pandas as pd
from chatnerds.langchain.prompt_factory import PromptFactory


DEFAULT_MAX_TOKENS = 128_000
MAX_DOCUMENTS_TO_PROCESS = 10

REPLACE_INPUT_APOSTROPHE_CHARS = ['"', "“", "”", "’"]
REPLACE_INPUT_SPACE_CHARS = ["\n", "\r", "\t"]
REPLACE_INPUT_DOT_CHARS = [":", ";"]
REMOVE_INPUT_CHARS = ["*"]
STRIP_OUTPUT_CHARS = [" ", "\n", "\r", "\t", "“", "”", "’", '"', "."]

DEFAULT_N_TAGS = 5

# OPENAI_SCHEMA = {
#     "properties": {
#         "tags": {
#             "type": "list",
#             # "enum": [1, 2, 3, 4, 5],
#             "description": "list of tags or keywords that classify the content of the document. The list is separated by commas.",
#         },
#         "category": {
#             "type": "string",
#             "description": "One word that describes the main category of the document",
#         },
#     },
#     "required": ["tags", "category"],
# }


class Tagger:
    config: Dict[str, Any] = {}
    chain: Chain = None  # BaseCombineDocumentsChain

    def __init__(
        self,
        config: Dict[str, Any],
        llm: LLMBase,
        prompt_type: Optional[str] = None,
    ):

        self.config = config or {}
        self.llm = llm
        self.prompt_type = prompt_type

    def find_tags(
        self, texts: List[str], n_tags: Optional[int] = DEFAULT_N_TAGS
    ) -> List[List[str]]:
        chain = self.get_tagging_chain(n_tags=n_tags)

        callbacks = [ConsoleCallbackHandler()]  # [ConsoleCallbackHandler()]

        texts_tags = []
        for text in texts:
            tags = chain.invoke({"text": text}, config={"callbacks": callbacks})
            if isinstance(tags, dict) and "text" in tags:
                tags = tags["text"]
            if not isinstance(tags, list):
                tags = [tags]

            texts_tags.append(tags)

        return texts_tags

    # https://dylancastillo.co/clustering-documents-with-openai-langchain-hdbscan/
    def find_clusterized_tags(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        n_tags: Optional[int] = DEFAULT_N_TAGS,
        combine_texts_separator: Optional[str] = "/n",
    ) -> List[List[str]] | None:
        min_cluster_size = max(3, (len(texts) // 100) + 3)

        # min_samples=6, cluster_selection_method="leaf"
        hdb = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=min_cluster_size).fit(
            embeddings
        )  # min_samples=3,

        df = pd.DataFrame(
            {
                "text": texts,
                "cluster": hdb.labels_,
            }
        )
        df = df.query("cluster != -1")  # Remove documents that are not in a cluster

        tagging_chain = self.get_tagging_chain(n_tags=n_tags)

        callbacks = [ConsoleCallbackHandler()]  # [ConsoleCallbackHandler()]

        clusters_tags = []
        for c in df.cluster.unique():
            cluster_articles = [
                article["text"]
                for article in df.query(f"cluster == {c}").to_dict(orient="records")
            ]

            if len(cluster_articles) > 100:
                cluster_articles = cluster_articles[:100]

            cluster_articles_str = self._format_input_text(
                cluster_articles, separator=combine_texts_separator
            )

            tags = tagging_chain.invoke(
                {"text": cluster_articles_str}, config={"callbacks": callbacks}
            )
            if isinstance(tags, dict) and "text" in tags:
                tags = tags["text"]
            if not isinstance(tags, list):
                tags = [tags]

            clusters_tags.append(tags)

        return clusters_tags

    def get_tagging_chain(self, n_tags: Optional[int] = DEFAULT_N_TAGS) -> Chain:

        config_prompts = self.config.get("prompts", {})
        find_tags_of_cluster_prompt = config_prompts.get("find_tags_prompt", None)

        prompt = PromptFactory.build_prompt_template(
            human_prompt=find_tags_of_cluster_prompt or "{text}",
            prompt_type=self.prompt_type,
            input_variables=["text"],
            partial_variables={
                "n_tags": n_tags,
                # "format_instructions": format_instructions,
            },
        )

        chain = (
            prompt
            | self.llm
            | CommaSeparatedListOutputParser()
            | self._strip_output_list
        )

        return chain

    @staticmethod
    def _format_input_text(
        texts_list: List[str], separator: Optional[str] = "/n"
    ) -> str:
        space_chars = ""
        for i in REPLACE_INPUT_SPACE_CHARS:
            space_chars += i

        apostrophe_chars = ""
        for i in REPLACE_INPUT_APOSTROPHE_CHARS:
            apostrophe_chars += i

        dot_chars = ""
        for i in REPLACE_INPUT_DOT_CHARS:
            dot_chars += i

        remove_chars = ""
        for i in REMOVE_INPUT_CHARS:
            remove_chars += i

        texts_str = ""
        for text in texts_list:
            text = re.sub(rf"[{remove_chars}]", "", text)
            text = re.sub(rf"[{space_chars}]", " ", text)
            text = re.sub(rf"[{apostrophe_chars}]", "'", text)
            text = re.sub(rf"[{dot_chars}]", ".", text)
            text = text.rstrip(separator)

            texts_str += f"{text}{separator}"

        return texts_str.rstrip(separator)

    @staticmethod
    def _strip_output_list(texts_list: List[str]) -> List[str]:

        if not texts_list or len(texts_list) == 0:
            return []

        strip_chars = ""
        for i in STRIP_OUTPUT_CHARS:
            strip_chars += i

        return [str(text).strip(strip_chars) for text in texts_list]
