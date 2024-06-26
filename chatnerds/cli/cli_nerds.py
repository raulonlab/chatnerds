import os
import re
import shutil
from pathlib import Path
import logging
import typer
from chatnerds.config import Config


YOUTUBE_SOURCES_INITIAL_CONTENT = """
# video urls
# https://www.youtube.com/watch?v=...

# playlist urls
# https://www.youtube.com/playlist?list=...

# channel urls
# https://www.youtube.com/@...
"""
PODCAST_SOURCES_INITIAL_CONTENT = """
# podcast feed urls
# https://feeds.buzzsprout.com/...
# https://feeds.simplecast.com/...
"""

NERD_CONFIG_INITIAL_CONTENT = """
# Override default nerd config
# See default config file in https://github.com/raulonlab/chatnerds/blob/main/chatnerds/config.yml,

"""

NERD_CONFIG_MODELS_INITIAL_CONTENT = """
# Add custom model presets available to your nerd
# See initial presets in https://github.com/raulonlab/chatnerds/blob/main/chatnerds/config.models.yml,
# or run "chatnerds config models" to see the complete list.

"""

NERD_CONFIG_PROMPTS_INITIAL_CONTENT = """
# Change the prompts used by the nerd.
# See default prompts in https://github.com/raulonlab/chatnerds/blob/main/chatnerds/config.prompts.yml,
# or run "chatnerds config prompts" to see the active ones.

"""

_global_config = Config.environment_instance()
app = typer.Typer()


@app.command(
    "init",
    help="Create and initialize a new nerd with the given name",
)
def init_nerd(nerd_name: str):
    if not validate_nerd_name(nerd_name):
        logging.error(
            f"Invalid nerd name '{nerd_name}'. Nerd name must start with alphanumeric and underscore characters"
        )
        return

    logging.debug(f"Initializing nerd '{nerd_name}'")
    nerd_path = Path(_global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if nerd_path.exists():
        logging.warning(f"Nerd directory '{nerd_name}' already exists... skipping")
        return
    # nerd directory
    nerd_path.mkdir(parents=True, exist_ok=True)

    # nerd store subdirectory
    nerd_store_path = Path(nerd_path, Config._NERD_STORE_DIRECTORYNAME)
    nerd_store_path.mkdir(parents=True, exist_ok=True)

    # nerd subdirectories
    Path(nerd_path, "downloads").mkdir(parents=True, exist_ok=True)
    Path(nerd_path, "downloads", "youtube").mkdir(parents=True, exist_ok=True)
    Path(nerd_path, "downloads", "podcasts").mkdir(parents=True, exist_ok=True)
    Path(nerd_path, "source_documents").mkdir(parents=True, exist_ok=True)

    # nerd sources files
    with open(
        Path(nerd_path, Config._YOUTUBE_SOURCES_FILENAME), "w", encoding="utf-8"
    ) as file:
        file.write(YOUTUBE_SOURCES_INITIAL_CONTENT)
    with open(
        Path(nerd_path, Config._PODCAST_SOURCES_FILENAME), "w", encoding="utf-8"
    ) as file:
        file.write(PODCAST_SOURCES_INITIAL_CONTENT)
    with open(
        Path(nerd_path, Config._NERD_CONFIG_FILENAME), "w", encoding="utf-8"
    ) as file:
        file.write(NERD_CONFIG_INITIAL_CONTENT)
    with open(
        Path(nerd_path, Config._NERD_CONFIG_MODELS_FILENAME), "w", encoding="utf-8"
    ) as file:
        file.write(NERD_CONFIG_MODELS_INITIAL_CONTENT)
    with open(
        Path(nerd_path, Config._NERD_CONFIG_PROMPTS_FILENAME), "w", encoding="utf-8"
    ) as file:
        file.write(NERD_CONFIG_PROMPTS_INITIAL_CONTENT)

    logging.info(f"Nerd '{nerd_name}' created")

    # Activate nerd if no active nerd
    if not _global_config.get_active_nerd():
        activate_nerd(nerd_name)


@app.command("remove", help="Remove an existing nerd")
def remove_nerd(nerd_name: str):
    logging.debug(f"Deleting nerd '{nerd_name}'")
    nerd_path = Path(_global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if not nerd_path or not nerd_path.exists():
        logging.error(f"Nerd '{nerd_name}' does not exist")
        return

    # Prompt for confirmation
    confirmation_response = typer.confirm(
        f"Are you sure you want to remove the nerd files in '{nerd_path}'?",
        default=False,
        abort=False,
    )
    if not confirmation_response:
        raise typer.Abort()

    shutil.rmtree(nerd_path)
    logging.info(f"Nerd '{nerd_name}' deleted")

    if _global_config.get_active_nerd() == nerd_name:
        _global_config.activate_nerd("")  # deactivate nerd
        logging.info("Nerd deactivated")


@app.command("rename", help="Rename an existing nerd")
def rename_nerd(nerd_name: str, new_nerd_name: str):
    if not validate_nerd_name(new_nerd_name):
        logging.error(
            f"Invalid nerd name '{new_nerd_name}'. Nerd name must start with alphanumeric and underscore characters"
        )
        return

    logging.debug(f"Renaming nerd '{nerd_name}' to '{new_nerd_name}'")
    nerd_path = Path(_global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if not nerd_path.exists():
        logging.error(f"Nerd '{nerd_name}' does not exist")
        return
    new_nerd_path = Path(_global_config.NERDS_DIRECTORY_PATH, new_nerd_name)
    if new_nerd_path.exists():
        logging.error(f"Nerd '{new_nerd_name}' already exists")
        return
    nerd_path.rename(new_nerd_path)

    # Update active nerd name if necessary
    if _global_config.get_active_nerd() == nerd_name:
        _global_config.activate_nerd(new_nerd_name)

    logging.info(f"Nerd '{nerd_name}' renamed to '{new_nerd_name}'")


@app.command(
    "activate",
    help="Activate a nerd. All subsequent commands like chat, study, etc. will use the active nerd configuration and source documents.",
)
def activate_nerd(nerd_name: str):
    logging.debug(f"Activating nerd '{nerd_name}'")
    nerd_path = Path(_global_config.NERDS_DIRECTORY_PATH, nerd_name)
    if not nerd_path.exists():
        logging.error(f"Nerd '{nerd_name}' does not exist")
        return

    _global_config.activate_nerd(nerd_name)
    logging.info(f"Nerd '{nerd_name}' activated")


@app.command("list", help="List all available nerds")
def list_nerds():
    logging.debug("Listing nerds")
    nerds = os.listdir(_global_config.NERDS_DIRECTORY_PATH)
    for nerd in nerds:
        if not validate_nerd_name(nerd):
            continue

        (
            print(f"[ ] {nerd}")
            if nerd != _global_config.get_active_nerd()
            else print(f"[x] {nerd}")
        )


def validate_nerd_name(nerd_name: str):
    return bool(re.match(r"\w", nerd_name))
