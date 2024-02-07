from typing import Any, Callable, Dict, Optional
from rich import print
from rich.markup import escape
from rich.panel import Panel
from datetime import datetime
from .langchain_chains import get_retrieval_qa

QA_LOG_PATH = "./chatnerds_qa.log"


def write_qa_log(text: str) -> None:
    with open(QA_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(text)


def handle_answer(text: str) -> None:
    escaped_text = escape(text)
    print(f"[bright_cyan]{escaped_text}", end="", flush=True)
    write_qa_log(escaped_text)


def get_log_header(config: Dict[str, Any]) -> str:
    now_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    log = "\n\n----------------------------------\n"
    log += f"Created at:      {now_string}\n"
    
    log += "----------------------------------\n"
    
    return log

def chat(config: Dict[str, Any], query: Optional[str] = None) -> None:
    qa = get_retrieval_qa(config, callback=handle_answer)
    # qa = get_retrieval_qa(config)

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
        write_qa_log(f"{get_log_header(config)}{log_query}")

        res = qa(query)

        if "result" in res:
            handle_answer(res["result"])
        else:
            handle_answer("No response result found!")

        print()
        log_sources = "\nSources:\n"
        for doc in res["source_documents"]:
            source, content = doc.metadata["source"], doc.page_content
            log_sources += f"- {source}\n"
            print(
                Panel(
                    f"[bright_blue]{escape(source)}[/bright_blue]\n\n{escape(content)}"
                )
            )

        write_qa_log(log_sources)
        print()

        if not interactive:
            break

