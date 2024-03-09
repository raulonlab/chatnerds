import os
import sys
import importlib.util
import glob
import shutil
import time
from pathlib import Path
from typing import List


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


def get_filtered_directories(
    directory_filter: str | None = None, base_path: Path = "."
) -> List[Path]:

    if not directory_filter or directory_filter.lower() in str(base_path).lower():
        return [base_path]

    # Search for subdirectories matching directory_filter
    filtered_directories = []
    for dirpath, dirs, _ in os.walk(base_path):
        for dir in dirs:
            full_dir_path = Path(dirpath, dir)

            if directory_filter.lower() in str(full_dir_path).lower():
                filtered_directories.append(full_dir_path)

                # skip subdirectories
                dirs.clear()

    return filtered_directories


def copy_files_between_directories(
    glob_search: str, src_dir: Path | str, dst_dir: Path | str
):
    src_dir = str(src_dir)
    dst_dir = str(dst_dir)

    for src_file in glob.glob(glob_search, recursive=True, root_dir=src_dir):
        os.makedirs(os.path.join(dst_dir, os.path.dirname(src_file)), exist_ok=True)
        shutil.copy(os.path.join(src_dir, src_file), os.path.join(dst_dir, src_file))


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
