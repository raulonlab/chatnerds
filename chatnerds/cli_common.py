from enum import Enum
import os
from pathlib import Path
from typing import Optional, List
from typing_extensions import Annotated
import typer
from . import utils

class SourceEnum(str, Enum):
    books = "books"
    youtube = "youtube"
    podcasts = "podcasts"

SOURCE_PATHS = {
    SourceEnum.books: "source_documents",
    SourceEnum.youtube: "downloads/youtube",
    SourceEnum.podcasts: "downloads/podcasts",
}

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
    typer.Argument(help="The relative path of the directory to be processed. Optional."),
]

def prompt_active_nerd(active_nerd: str, nerd_base_path: Path):
    if not nerd_base_path.exists():
        typer.echo(f"The active nerd {utils.LogColors.BOLDNERD}{active_nerd}{utils.LogColors.ENDC} does not exist. Please create it first.")
        raise typer.Abort()
    else:
        prompt_text = f"The active nerd is {utils.LogColors.BOLDNERD}{active_nerd}{utils.LogColors.ENDC}"
        return typer.confirm(f"{prompt_text}... do you want to continue?", default=True, abort=False)

def get_source_directory_paths(directory_filter: Path = None, source: SourceEnum = None, base_path: Path = ".") -> List[Path]:
    filtered_directories = get_filtered_directories(directory_filter=directory_filter, base_path=base_path)
    if len(filtered_directories) > 0:
        return filtered_directories
    
    source_directories = get_source_directory(source=source, base_path=base_path)
    if len(source_directories) > 0:
        return source_directories
    
    # Not found
    return []

def get_filtered_directories(directory_filter: Path = None, base_path: Path = ".") -> List[Path]:
    if not directory_filter:
        return []
    
    if directory_filter.exists():
        return [directory_filter]
    
    directory_filter_from_base_path = Path(base_path, directory_filter)
    if directory_filter_from_base_path.exists():
        return [directory_filter_from_base_path]
    
    # Search for directory
    filtered_directories = []
    for dirpath, dirs, _ in os.walk(base_path):
        for dir in dirs:
            full_dir_path = Path(dirpath, dir)
            # find directory_filter at the end of full_dir_path
            if str(full_dir_path).endswith(str(directory_filter)):
                filtered_directories.append(full_dir_path)

    return filtered_directories

def get_source_directory(source: SourceEnum = None, base_path: Path = ".") -> List[Path]:
    source_directories = []
    if not source:
        source_directories.extend([
            Path(base_path, SOURCE_PATHS[SourceEnum.books.value]), 
            Path(base_path, SOURCE_PATHS[SourceEnum.youtube.value]), 
            Path(base_path, SOURCE_PATHS[SourceEnum.podcasts.value]), 
        ])
    else:
        source_directories.append(Path(base_path, SOURCE_PATHS[source.value]))

    return source_directories

