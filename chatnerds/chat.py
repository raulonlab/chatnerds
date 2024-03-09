from typing import Dict, Optional
from rich import print
from rich.markup import escape
from rich.panel import Panel
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from chatnerds.langchain.chain_factory import ChainFactory
from chatnerds.tools.chat_logger import ChatLogger
from chatnerds.config import Config


_global_config = Config.environment_instance()


def chat(query: Optional[str] = None) -> None:
    nerd_config = _global_config.get_nerd_config()

    chat_chain = ChainFactory(nerd_config).get_chat_chain()
    chat_logger = ChatLogger()

    interactive = not query
    print()
    if interactive:
        print("Type your query below and press Enter.")
        print("Type 'exit' or 'quit' or 'q' to exit the application.\n")

    while True:
        print("[bold]Q: ", end="", flush=True)
        if interactive:
            query = input()
        else:
            print(escape(query))
        print()
        if query.strip() in ["exit", "quit", "q"]:
            print("Exiting...\n")
            break
        print("[bold]A:", end="", flush=True)

        callbacks = []
        if _global_config.VERBOSE > 1:
            callbacks.append(ConsoleCallbackHandler())

        res = chat_chain.invoke(query, config={"callbacks": callbacks})

        if isinstance(res, Dict) and "result" in res:
            print()
            if "source_documents" in res:
                for doc in res["source_documents"]:
                    source, content = doc.metadata["source"], doc.page_content
                    print(
                        Panel(
                            f"[bright_blue]{escape(source)}[/bright_blue]\n\n{escape(content)}"
                        )
                    )

                print()

            chat_logger.log(query, res["result"], res.get("source_documents", []))
        else:
            chat_logger.log(query, res)

        if not interactive:
            break
