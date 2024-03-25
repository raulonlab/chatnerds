from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.llms.base import LLM as LLMBase
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate


class PromptFactory:

    def get_rag_prompt(
        self, llm: LLMBase, prompt_type=None, system_prompt=None, history=False
    ):
        prompt_selector = ConditionalPromptSelector(
            default_prompt=self._get_rag_prompt_generic(
                system_prompt=system_prompt, history=history
            ),
            conditionals=[
                (
                    lambda llm: isinstance(llm, LlamaCpp),
                    self._get_rag_prompt_by_type(
                        prompt_type=prompt_type, system_prompt=system_prompt
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
        prompt_messages.append(("human", "Context: {context}"))
        prompt_messages.append(("human", "Question: {question}"))

        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        return prompt

    def _get_rag_prompt_by_type(
        self, prompt_type=None, system_prompt=None, history=False
    ):
        prompt_parts = []

        if (
            prompt_type == "llama"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-llama
            prompt_parts.append("[INST]")
            if system_prompt:
                prompt_parts.append(
                    "<<SYS>> {system_prompt}<</SYS>>\n".format(
                        system_prompt=system_prompt.strip(" \n")
                    )
                )
            prompt_parts.append(
                "Context: {context}\nQuestion: {question} \nAnswer:[/INST]"
            )

        elif (
            prompt_type == "mistral"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            if system_prompt:
                prompt_parts.append(
                    "<s> [INST] {system_prompt} [/INST] </s>\n".format(
                        system_prompt=system_prompt.strip(" \n")
                    )
                )
            prompt_parts.append(
                "[INST] Context: {context}\nQuestion: {question}\nAnswer:[/INST]"
            )

        else:  # https://smith.langchain.com/hub/rlm/rag-prompt
            if system_prompt:
                prompt_parts.append(system_prompt.strip(" \n") + "\n")
            prompt_parts.append("Context: {context} \nQuestion: {question} \nAnswer:")

        prompt_template = "".join(prompt_parts)
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        return prompt

    def get_system_human_prompt(
        self,
        llm: LLMBase,
        prompt_type: str = None,
        system_prompt: str = None,
        human_prompt: str = None,
        append_human_variable: str = None,
    ):
        prompt_selector = ConditionalPromptSelector(
            default_prompt=self._get_system_human_prompt_generic(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                append_human_variable=append_human_variable,
            ),
            conditionals=[
                (
                    lambda llm: isinstance(llm, LlamaCpp),
                    self._get_system_human_prompt_by_type(
                        prompt_type=prompt_type,
                        system_prompt=system_prompt,
                        human_prompt=human_prompt,
                        append_human_variable=append_human_variable,
                    ),
                )
            ],
        )

        prompt = prompt_selector.get_prompt(llm)

        return prompt

    def _get_system_human_prompt_generic(
        self,
        system_prompt: str = None,
        human_prompt: str = "",
        append_human_variable: str = None,
    ):

        prompt_messages = []
        if system_prompt:
            prompt_messages.append(("system", system_prompt))

        human_message = human_prompt or ""
        if append_human_variable:
            human_message = (
                human_prompt.strip(" \n") + "\n{" + append_human_variable + "}"
            )
        prompt_messages.append(("human", human_message))

        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        return prompt

    def _get_system_human_prompt_by_type(
        self,
        prompt_type=None,
        system_prompt=None,
        human_prompt: str = "",
        append_human_variable: str = None,
    ):

        prompt_parts = []

        if (
            prompt_type == "llama"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-llama
            prompt_parts.append("[INST]")
            if system_prompt:
                prompt_parts.append(
                    "<<SYS>> {system_prompt}<</SYS>>".format(
                        system_prompt=system_prompt.strip(" \n")
                    )
                )
            human_message = human_prompt or ""
            if append_human_variable:
                human_message = (
                    human_message.strip(" \n") + "\n{" + append_human_variable + "}"
                )
            prompt_parts.append(human_message + "[/INST]")

        elif (
            prompt_type == "mistral"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            if system_prompt:
                prompt_parts.append(
                    "<s> [INST] {system_prompt} [/INST] </s>\n".format(
                        system_prompt=system_prompt.strip(" \n")
                    )
                )
            human_message = human_prompt or ""
            if append_human_variable:
                human_message = (
                    human_message.strip(" \n") + "\n{" + append_human_variable + "}"
                )
            prompt_parts.append("[INST] " + human_message + "[/INST]")

        else:  # https://smith.langchain.com/hub/rlm/rag-prompt
            if system_prompt:
                prompt_parts.append(system_prompt.strip(" \n") + "\n")

            human_message = human_prompt or ""
            if append_human_variable:
                human_message = (
                    human_message.strip(" \n") + "\n{" + append_human_variable + "}"
                )
            prompt_parts.append(human_message)

        prompt_template = "".join(prompt_parts)
        prompt = PromptTemplate(
            input_variables=[append_human_variable] if append_human_variable else [],
            template=prompt_template,
        )

        return prompt

    def format_prompt_template(
        self,
        llm: LLMBase,
        prompt: str,
        prompt_type: Optional[str] = None,
        **kwargs,
    ):
        prompt_selector = ConditionalPromptSelector(
            default_prompt=PromptTemplate(
                template=prompt,
                **kwargs,
            ),
            conditionals=[
                (
                    lambda llm: isinstance(llm, LlamaCpp),
                    self._format_prompt_template_by_type(
                        prompt=prompt,
                        prompt_type=prompt_type,
                        **kwargs,
                    ),
                )
            ],
        )

        prompt = prompt_selector.get_prompt(llm)

        return prompt

    def _format_prompt_template_by_type(
        self,
        prompt: str,
        prompt_type: Optional[str] = None,
        **kwargs,
    ):
        prompt_parts = []

        if (
            prompt_type == "llama"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-llama
            prompt_parts.append("[INST]")
            prompt_parts.append(prompt.strip(" \n"))
            prompt_parts.append("[/INST]")
        elif (
            prompt_type == "mistral"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            prompt_parts.append("<s>[INST] ")
            prompt_parts.append(prompt.strip(" \n"))
            prompt_parts.append(" [/INST]")
        else:  # https://smith.langchain.com/hub/rlm/rag-prompt
            prompt_parts.append(prompt.strip(" \n"))

        prompt_template = "".join(prompt_parts)
        prompt = PromptTemplate(template=prompt_template, **kwargs)

        return prompt
