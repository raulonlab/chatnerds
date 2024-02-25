import os
from pathlib import Path
from typing import List
from chatnerds import SourceEnum, SOURCE_PATHS


def get_source_directory_paths(
    directory_filter: Path = None, source: SourceEnum = None, base_path: Path = "."
) -> List[Path]:
    filtered_directories = get_filtered_directories(
        directory_filter=directory_filter, base_path=base_path
    )
    if len(filtered_directories) > 0:
        return filtered_directories

    source_directories = get_source_directory(source=source, base_path=base_path)
    if len(source_directories) > 0:
        return source_directories

    # Not found
    return []


def get_filtered_directories(
    directory_filter: Path = None, base_path: Path = "."
) -> List[Path]:
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


def get_source_directory(
    source: SourceEnum = None, base_path: Path = "."
) -> List[Path]:
    source_directories = []
    if not source:
        source_directories.extend(
            [
                Path(base_path, SOURCE_PATHS[SourceEnum.books.value]),
                Path(base_path, SOURCE_PATHS[SourceEnum.youtube.value]),
                Path(base_path, SOURCE_PATHS[SourceEnum.podcasts.value]),
            ]
        )
    else:
        source_directories.append(Path(base_path, SOURCE_PATHS[source.value]))

    return source_directories
