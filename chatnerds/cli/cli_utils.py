from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import typer
from typer.core import TyperGroup
from click import Context
from tqdm import tqdm
from chatnerds.enums import SourceEnum, LogColors
from chatnerds.config import Config


_global_config = Config.environment_instance()


SourceOption = Annotated[
    Optional[SourceEnum],
    typer.Option(
        "--source",
        "-s",
        case_sensitive=False,
        help="The source to be processed: books, youtube, podcasts. If not specified, process all sources.",
    ),
]


DirectoryFilterArgument = Annotated[
    Optional[Path],
    typer.Argument(
        help="The relative path of the directory to be processed. Optional."
    ),
]


UrlFilterArgument = Annotated[
    str,
    typer.Argument(help="Url of Youtube video"),
]


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


class TqdmHolder:
    _tqdm: Optional[tqdm] = None
    _init_kwargs: dict = {}

    def __init__(self, **kwargs):
        self._init_kwargs = kwargs

    def start(self, *args, **kwargs):
        if self._tqdm:
            self._tqdm.close()

        # move positional argument "total" to kwargs
        if args and len(args) > 0:
            kwargs["total"] = int(args[0])

        self._tqdm = tqdm(**{**self._init_kwargs, **kwargs})

    def update(self, *args, **kwargs):
        if not self._tqdm:
            return

        self._tqdm.update(*args, **kwargs)

    def close(self):
        if not self._tqdm:
            return

        self._tqdm.close()
        self._tqdm = None


# validate and prompt to confirm the active nerd
def validate_confirm_active_nerd(skip_confirmation: bool = True):
    if not _global_config.get_active_nerd():
        typer.echo(
            "No active nerd set. Please set an active nerd first. See chatnerds nerd --help. "
        )
        raise typer.Abort()

    active_nerd = _global_config.get_active_nerd()

    if skip_confirmation:
        typer.echo(f"Using nerd {LogColors.BOLDNERD}{active_nerd}{LogColors.ENDC}...")
    else:
        prompt_response = prompt_active_nerd(
            active_nerd=active_nerd,
            nerd_base_path=_global_config.get_nerd_base_path(),
        )
        if not prompt_response:
            raise typer.Abort()


def prompt_active_nerd(active_nerd: str, nerd_base_path: Path):
    if not nerd_base_path.exists():
        typer.echo(
            f"The active nerd {LogColors.BOLDNERD}{active_nerd}{LogColors.ENDC} does not exist. Please create it first."
        )
        raise typer.Abort()
    else:
        prompt_text = (
            f"The active nerd is {LogColors.BOLDNERD}{active_nerd}{LogColors.ENDC}"
        )
        return typer.confirm(
            f"{prompt_text}... do you want to continue?", default=True, abort=False
        )


def grep_match(pattern: str, *args):
    """
    Returns True if a pattern matches any of the args values (in a case-insensitive manner)
    """
    if not pattern:
        return True

    pattern = pattern.lower()
    for arg in args:
        if pattern in str(arg).lower():
            return True
    return False
