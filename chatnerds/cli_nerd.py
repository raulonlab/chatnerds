import os
import shutil
from pathlib import Path
import logging
from typing import Dict
import typer
from .config import Config

YOUTUBE_SOURCES_FILE_NAME = "youtube.sources"
PODCAST_SOURCES_FILE_NAME = "podcast.sources"
CHATDOCS_YAML_FILE_NAME = "chatdocs.yml"

YOUTUBE_SOURCES_INITIAL_CONTENT = """
# playlist urls
# https://www.youtube.com/playlist?list=...

# video urls
# https://www.youtube.com/watch?v=...
"""
PODCAST_SOURCES_INITIAL_CONTENT = """
# podcast feed urls
# https://feeds.buzzsprout.com/...
# https://feeds.simplecast.com/...
"""

CHATDOCS_YAML_INITIAL_CONTENT = """
# Override chatdocs configuration. Default: https://github.com/marella/chatdocs/blob/main/chatdocs/data/chatdocs.yml

download: true
"""

global_config = Config.environment_instance()

app = typer.Typer()

@app.command("create")
def create_nerd(nerd_name: str):
    logging.debug(f"Creating nerd '{nerd_name}'")
    nerd_path = Path(global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if nerd_path.exists():
        logging.error(f"Nerd '{nerd_name}' already exists")
        return
    # nerd directory
    nerd_path.mkdir(parents=True, exist_ok=True)

    # nerd subdirectories
    Path(nerd_path, "downloads").mkdir(parents=True, exist_ok=True)
    Path(nerd_path, "downloads", "youtube").mkdir(parents=True, exist_ok=True)
    Path(nerd_path, "downloads", "podcasts").mkdir(parents=True, exist_ok=True)
    Path(nerd_path, "source_documents").mkdir(parents=True, exist_ok=True)

    # nerd sources files
    with open(Path(nerd_path, YOUTUBE_SOURCES_FILE_NAME), "w", encoding="utf-8") as file:
        file.write(YOUTUBE_SOURCES_INITIAL_CONTENT)
    with open(Path(nerd_path, PODCAST_SOURCES_FILE_NAME), "w", encoding="utf-8") as file:
        file.write(PODCAST_SOURCES_INITIAL_CONTENT)
    with open(Path(nerd_path, CHATDOCS_YAML_FILE_NAME), "w", encoding="utf-8") as file:
        file.write(CHATDOCS_YAML_INITIAL_CONTENT)

    logging.info(f"Nerd '{nerd_name}' created")

@app.command("delete")
def delete_nerd(nerd_name: str):
    logging.debug(f"Deleting nerd '{nerd_name}'")
    nerd_path = Path(global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if not nerd_path.exists():
        logging.error(f"Nerd '{nerd_name}' does not exist")
        return
    shutil.rmtree(nerd_path)
    logging.info(f"Nerd '{nerd_name}' deleted")

@app.command("rename")
def rename_nerd(nerd_name: str, new_nerd_name: str):
    logging.debug(f"Renaming nerd '{nerd_name}' to '{new_nerd_name}'")
    nerd_path = Path(global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if not nerd_path.exists():
        logging.error(f"Nerd '{nerd_name}' does not exist")
        return
    new_nerd_path = Path(global_config.NERDS_DIRECTORY_PATH, new_nerd_name)
    if new_nerd_path.exists():
        logging.error(f"Nerd '{new_nerd_name}' already exists")
        return
    nerd_path.rename(new_nerd_path)

    if global_config.get_active_nerd() == nerd_name:
        global_config.activate_nerd(new_nerd_name)

    logging.info(f"Nerd '{nerd_name}' renamed to '{new_nerd_name}'")

@app.command("activate")
def activate_nerd(nerd_name: str):
    logging.debug(f"Activating nerd '{nerd_name}'")
    nerd_path = Path(global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if not nerd_path.exists():
        logging.error(f"Nerd '{nerd_name}' does not exist")
        return
    
    global_config.activate_nerd(nerd_name)
    logging.info(f"Nerd '{nerd_name}' activated")

@app.command("list")
def list_nerds():
    logging.debug("Listing nerds")
    nerds = os.listdir(global_config.NERDS_DIRECTORY_PATH)
    for nerd in nerds:
        print(f"[ ] {nerd}") if nerd != global_config.get_active_nerd() else print(f"[x] {nerd}")



