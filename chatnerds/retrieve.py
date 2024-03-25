import os
from typing import Optional, Dict
from rich import print
from rich.markup import escape
from rich.panel import Panel
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from chatnerds.langchain.chain_factory import ChainFactory
from chatnerds.config import Config


_global_config = Config.environment_instance()


def retrieve(query: str = None, with_summary: Optional[bool] = False) -> None:
    nerd_config = _global_config.get_nerd_config()

    retrieve_chain = ChainFactory(nerd_config).get_retrieve_chain(
        with_summary=with_summary
    )

    callbacks = []
    if _global_config.VERBOSE > 1:
        callbacks.append(ConsoleCallbackHandler())

    output = retrieve_chain.invoke(query, config={"callbacks": callbacks})
    if isinstance(output, Dict):
        documents = output.get("documents", [])
        summary = output.get("summary", "")
    else:
        documents = output
        summary = ""

    for doc in documents:
        artist = doc.metadata.get("artist", "")
        if not artist:
            artist = doc.metadata.get("album", "")

        title = doc.metadata.get("title", "")
        link = doc.metadata.get("comment", "")

        sanatized_source = os.path.relpath(
            doc.metadata.get("source", ""), _global_config.get_nerd_base_path()
        )
        if not sanatized_source:
            sanatized_source = "(Source not found)"

        sanatized_title = ""
        if artist:
            sanatized_title = sanatized_title + f"{artist} - "
        if title:
            sanatized_title = sanatized_title + f"{title}"
        if link:
            sanatized_title = f"[link={link}]{sanatized_title or link}[/link]"

        print(
            Panel(
                f"[bright_blue]...{escape(sanatized_source)}[/bright_blue]\n\n{escape(doc.page_content)}",
                title=sanatized_title,
                title_align="left",
            )
        )

    if with_summary:
        print()
        print(
            Panel(
                f"{summary}",
                title="Summary",
                title_align="left",
            )
        )
