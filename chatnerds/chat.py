from typing import Any, Dict, Optional
from rich import print
from rich.markup import escape
from rich.panel import Panel
from datetime import datetime
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from chatnerds.langchain.chain_factory import ChainFactory
from chatnerds.config import Config

_global_config = Config.environment_instance()

_QA_LOG_PATH = "./chatnerds_qa.log"


def _write_qa_log(text: str) -> None:
    with open(_QA_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(text)


def _handle_answer(text: str) -> None:
    escaped_text = escape(text)
    print(f"[bright_cyan]{escaped_text}", end="", flush=True)
    _write_qa_log(escaped_text)


def _get_log_header(config: Dict[str, Any]) -> str:
    now_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    log = "\n\n----------------------------------\n"
    log += f"Created at:      {now_string}\n"

    log += "----------------------------------\n"

    return log


def chat(query: Optional[str] = None) -> None:
    config = _global_config.get_nerd_config()

    chat_chain = ChainFactory(config).get_rag_fusion_chain()

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

        # raul: save Q & A to log file
        log_query = f"Q:\n{escape(query)}\nA:\n"
        _write_qa_log(f"{_get_log_header(config)}{log_query}")

        # print("get_graph:")
        # chat_chain.get_graph().print_ascii()

        callbacks = []
        if _global_config.VERBOSE > 1:
            callbacks.append(ConsoleCallbackHandler())

        res = chat_chain.invoke(query, config={"callbacks": callbacks})

        if isinstance(res, Dict) and "result" in res:
            _handle_answer(res["result"])

            print()
            if "source_documents" in res:
                log_sources = "\nSources:\n"
                for doc in res["source_documents"]:
                    source, content = doc.metadata["source"], doc.page_content
                    log_sources += f"- {source}\n"
                    print(
                        Panel(
                            f"[bright_blue]{escape(source)}[/bright_blue]\n\n{escape(content)}"
                        )
                    )

                _write_qa_log(log_sources)
                print()
        else:
            # _handle_answer("No response result found!")
            _handle_answer(res)

        if not interactive:
            break
