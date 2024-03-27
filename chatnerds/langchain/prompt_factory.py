from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.prompts import ChatPromptTemplate


class PromptFactory:

    @staticmethod
    def build_prompt_template(
        human_prompt: str,
        system_prompt: Optional[str] = None,
        prompt_type: Optional[str] = None,
        **kwargs,
    ):
        prompt_selector = ConditionalPromptSelector(
            default_prompt=PromptFactory._build_from_messages(
                human_prompt=human_prompt,
                system_prompt=system_prompt,
                **kwargs,
            ),
            conditionals=[
                (
                    lambda prompt_type: prompt_type,
                    PromptFactory._build_apply_syntax(
                        human_prompt=human_prompt,
                        system_prompt=system_prompt,
                        prompt_type=prompt_type,
                        **kwargs,
                    ),
                )
            ],
        )

        prompt = prompt_selector.get_prompt(prompt_type)

        return prompt

    @staticmethod
    def _build_from_messages(
        human_prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        prompt_messages = []
        if system_prompt:
            prompt_messages.append(("system", system_prompt))

        prompt_messages.append(("human", human_prompt))

        prompt_template = ChatPromptTemplate.from_messages(prompt_messages)
        if "partial_variables" in kwargs:
            prompt_template = prompt_template.partial(**kwargs["partial_variables"])

        return prompt_template

    @staticmethod
    def _build_apply_syntax(
        human_prompt: str,
        system_prompt: Optional[str] = None,
        prompt_type: Optional[str] = None,
        **kwargs,
    ):

        prompt_parts = []

        if (
            prompt_type == "llama"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-llama
            prompt_parts.append("[INST]")
            if system_prompt:
                prompt_parts.append(
                    "<<SYS>> {system_prompt} <</SYS>>".format(
                        system_prompt=system_prompt.strip(" \n")
                    )
                )

            prompt_parts.append("[INST] " + human_prompt + " [/INST]")

        elif (
            prompt_type == "mistral"
        ):  # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            if system_prompt:
                prompt_parts.append(
                    "<s>[INST] {system_prompt} [/INST]</s>".format(
                        system_prompt=system_prompt.strip(" \n")
                    )
                )

            prompt_parts.append("[INST] " + human_prompt + " [/INST]")

        else:  # https://smith.langchain.com/hub/rlm/rag-prompt
            if system_prompt:
                prompt_parts.append(system_prompt.strip(". \n") + ". ")

            prompt_parts.append(human_prompt)

        template = "".join(prompt_parts)
        prompt_template = PromptTemplate(
            template=template,
            **kwargs,
        )

        return prompt_template
