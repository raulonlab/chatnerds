import os
from typing import Optional
from typing_extensions import Annotated
import typer
import logging
from chatnerds.config import Config
from chatnerds.logging_setup import setup as setup_logging
from chatnerds.cli import cli_nerds, cli_sources, cli_tools, cli_utils
from chatnerds import LogColors
from chatnerds import chat, study
from chatnerds.langchain.document_embeddings import DocumentEmbeddings
from chatnerds.langchain.chroma_database import ChromaDatabase

# load config
_global_config = Config.environment_instance()

_log_terminal_level = logging.DEBUG
if _global_config.VERBOSE == 0:
    _log_terminal_level = logging.WARNING
elif _global_config.VERBOSE == 1:
    _log_terminal_level = logging.INFO

setup_logging(
    log_terminal_level=_log_terminal_level,
    log_file_level=_global_config.LOG_FILE_LEVEL,
    log_file_path=_global_config.LOG_FILE_PATH,
)

app = typer.Typer(cls=cli_utils.OrderCommands)

# Import commands from other modules
# app.add_typer(cli_nerds.app, name="nerd")
# app.add_typer(cli_sources.app, name="sources")
app.registered_commands += cli_nerds.app.registered_commands
app.registered_commands += cli_sources.app.registered_commands


@app.command("study", help="Add source documents to embeddings DB")
def study_command(
    directory_filter: cli_utils.DirectoryFilterArgument = None,
    source: cli_utils.SourceOption = None,
):
    if not directory_filter and not source:
        typer.echo(
            "No directory or source specified. Please specify one of them. See chatnerds add --help."
        )
        raise typer.Abort()

    cli_utils.validate_confirm_active_nerd()

    try:
        study(directory_filter=directory_filter, source=source)
    except Exception as e:
        logging.error(
            f"Error studying sources. Try to load the sources separately with the option --source. See chatnerds add --help."
        )
        logging.error(e)
    except SystemExit:
        raise typer.Abort()


@app.command(
    "chat", help="Start a chat session with the documents added in the embeddings DB"
)
def chat_command(
    query: Annotated[
        Optional[str],
        typer.Argument(
            help="The query to use for retrieval. If not specified, runs in interactive mode."
        ),
    ] = None,
):
    cli_utils.validate_confirm_active_nerd()

    chat(query=query)


@app.command("db", help="Print summary of the embeddings DB")
def db_command():
    cli_utils.validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    embeddings = DocumentEmbeddings(config=nerd_config).get_embedding_function()

    database = ChromaDatabase(embeddings=embeddings, config=nerd_config["chroma"])

    collection = database.client.get(include=["metadatas"])

    # Get (distinct) documents loaded in the db
    distinct_documents = []
    for metadata in collection["metadatas"]:
        metadata["source"] = os.path.basename(metadata["source"])
        if metadata["source"] not in map(lambda x: x["source"], distinct_documents):
            distinct_documents.append(metadata)

    # Print summary
    print("Chroma DB information:")
    print(
        f"DB Path:       {LogColors.BOLD}{nerd_config['chroma']['persist_directory']}{LogColors.ENDC}"
    )
    print(f"Documents ({len(distinct_documents)}):")
    for document in distinct_documents:
        print(f"  - {document['source']}")
        for key, value in document.items():
            if key != "source":
                print(f"      - {key}: {value}")


@app.command("config", help="Print runtime configuration")
def config_command():
    print(str(_global_config))


app.add_typer(
    cli_tools.app,
    name="tools",
    help="Other tools",
    short_help="Other tools",
    epilog="* These tools work independently of your active nerd environment.",
)
