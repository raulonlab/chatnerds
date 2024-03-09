from typing import Any, Dict
from langchain.prompts import PromptTemplate
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.llms.base import LLM as LLMBase
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate


class PromptFactory:
    config: Dict[str, Any]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else {}

    def get_rag_prompt(
        self, llm: LLMBase, system_prompt=None, prompt_type=None, history=False
    ):
        prompt_selector = ConditionalPromptSelector(
            default_prompt=self._get_rag_prompt_generic(
                system_prompt=system_prompt, history=history
            ),
            conditionals=[
                (
                    lambda llm: isinstance(llm, LlamaCpp),
                    self._get_rag_prompt_by_type(
                        system_prompt=system_prompt, prompt_type=prompt_type
                    ),
                )
            ],
        )

        prompt = prompt_selector.get_prompt(llm)

        return prompt

    def _get_rag_prompt_generic(self, system_prompt=None, history=False):
        prompt_messages = []
        if system_prompt:
            prompt_messages.append(("system", system_prompt))
        prompt_messages.append(
            ("human", "Context: {context}\nQuestion: {question}\nAnswer:")
        )

        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        return prompt

    def _get_rag_prompt_by_type(
        self, system_prompt=None, prompt_type=None, history=False
    ):
        prompt_parts = []

        if (
            prompt_type == "llama"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-llama
            prompt_parts.append("[INST]")
            if system_prompt:
                prompt_parts.append(
                    "<<SYS>> {system_prompt}<</SYS>>".format(
                        system_prompt=system_prompt
                    )
                )
            prompt_parts.append(
                "\nContext: {context}\nQuestion: {question} \nAnswer:[/INST]"
            )

        elif (
            prompt_type == "mistral"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            if system_prompt:
                prompt_parts.append(
                    "<s> [INST] {system_prompt} [/INST] </s>\n".format(
                        system_prompt=system_prompt
                    )
                )
            prompt_parts.append(
                "[INST] Context: {context}\nQuestion: {question}\nAnswer:[/INST]"
            )

        else:  # https://smith.langchain.com/hub/rlm/rag-prompt
            if system_prompt:
                prompt_parts.append(system_prompt)
            prompt_parts.append("\nContext: {context} \nQuestion: {question} \nAnswer:")

        prompt_template = "".join(prompt_parts)
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        return prompt

    def get_question_answer_prompt(
        self,
        llm: LLMBase,
        system_prompt: str = None,
        prompt_type: str = None,
    ):
        prompt_selector = ConditionalPromptSelector(
            default_prompt=self._get_question_answer_prompt_generic(
                system_prompt=system_prompt
            ),
            conditionals=[
                (
                    lambda llm: isinstance(llm, LlamaCpp),
                    self._get_question_answer_prompt_by_type(
                        system_prompt=system_prompt, prompt_type=prompt_type
                    ),
                )
            ],
        )

        prompt = prompt_selector.get_prompt(llm)

        return prompt

    def _get_question_answer_prompt_generic(self, system_prompt=None):
        prompt_messages = []
        if system_prompt:
            prompt_messages.append(("system", system_prompt))
        prompt_messages.append(("human", "Question: {question}\nAnswer:"))

        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        return prompt

    def _get_question_answer_prompt_by_type(self, system_prompt=None, prompt_type=None):
        prompt_parts = []

        if (
            prompt_type == "llama"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-llama
            prompt_parts.append("[INST]")
            if system_prompt:
                prompt_parts.append(
                    "<<SYS>> {system_prompt}<</SYS>>".format(
                        system_prompt=system_prompt
                    )
                )
            prompt_parts.append("\nQuestion: {question}\nAnswer:[/INST]")

        elif (
            prompt_type == "mistral"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            if system_prompt:
                prompt_parts.append(
                    "<s> [INST] {system_prompt} [/INST] </s>\n".format(
                        system_prompt=system_prompt
                    )
                )
            prompt_parts.append("[INST] Question: {question}\nAnswer:[/INST]")

        else:  # https://smith.langchain.com/hub/rlm/rag-prompt
            if system_prompt:
                prompt_parts.append(system_prompt)
            prompt_parts.append("\nQuestion: {question}\nAnswer:")

        prompt_template = "".join(prompt_parts)
        prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

        return prompt
