import os
import sys
import importlib.util
import time
from pathlib import Path
from typing import List
from chatnerds.enums import SourceEnum, SOURCE_PATHS


class TimeTaken:
    """
    Records the duration of a task in debug mode and prints it to the console.
    """

    def __init__(self, title: str, callback: callable = None):
        self.title = title
        self.callback = callback
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        diff = time.time() - self.start
        if self.callback:
            self.callback(f"TimeTaken - {self.title}: {diff:.4f} seconds")
        else:
            print(f"TimeTaken - {self.title}: {diff:.4f} seconds")


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


# borrowed from: https://stackoverflow.com/a/1051266/656011
def check_for_package(package):
    if package in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(package)) is not None:
        try:
            module = importlib.util.module_from_spec(spec)

            sys.modules[package] = module
            spec.loader.exec_module(module)

            return True
        except ImportError:
            return False
    else:
        return False


# Yield successive n-sized chunks from l.
def divide_list_in_chunks(input_list: list, chunk_size: int):
    # looping till length chunk_size
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


def process_memory_limit(limit):
    import resource as rs

    soft, hard = rs.getrlimit(rs.RLIMIT_AS)
    rs.setrlimit(rs.RLIMIT_AS, (limit, hard))
