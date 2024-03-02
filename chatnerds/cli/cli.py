from typing import Optional
from typing_extensions import Annotated
import typer
import logging
from chatnerds.config import Config
from chatnerds.logging_setup import setup as setup_logging
from chatnerds.cli import cli_nerds, cli_sources, cli_tools, cli_utils, cli_db


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

    from chatnerds.study import study

    try:
        study(directory_filter=directory_filter, source=source)
    except Exception as e:
        logging.error(
            "Error studying sources. Try to load the sources separately with the option --source. See chatnerds add --help."
        )
        raise e
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

    from chatnerds.chat import chat

    chat(query=query)


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

app.add_typer(
    cli_db.app,
    name="db",
    help="Database operations",
    short_help="Database operations",
    epilog="* These commands require an active nerd environment.",
)
