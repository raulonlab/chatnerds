# Config consts
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, ClassVar
from dotenv import load_dotenv
from deepmerge import always_merger
import yaml
from dataclasses import dataclass, fields

_RUNTIME_DOTENV_PATH = ".chatnerds.env"

load_dotenv(dotenv_path=".env", override=True)  # load default .env file
load_dotenv(
    dotenv_path=_RUNTIME_DOTENV_PATH, override=True
)  # load runtime .env file (save env vars between runs)


@dataclass
class Config(object):
    _DB_NAME: ClassVar[str] = "db"  # Name of the database directory
    _NERD_CONFIG_FILENAME: ClassVar[str] = "config.yml"  # Name of the nerd config file
    _YOUTUBE_SOURCES_FILENAME: ClassVar[str] = (
        "youtube.sources"  # Name of the youtube sources file
    )
    _PODCAST_SOURCES_FILENAME: ClassVar[str] = (
        "podcast.sources"  # Name of the podcast sources file
    )
    _environment_instance: ClassVar = None  # singleton instance with environment values
    _default_instance: ClassVar = None  # singleton instance with class default values

    ACTIVE_NERD: Union[str, None] = None  # (Default: None) Name of the active nerd
    NERDS_DIRECTORY_PATH: str = "nerds"  # (Default: "nerds") Path to nerds directory
    LOG_FILE_LEVEL: Optional[str] = (
        None  # (Default: None) Logging level for the log file. Values: INFO, WARNING, ERROR, CRITICAL, NOTSET. If None, disable logging to file
    )
    LOG_FILE_PATH: str = (
        "logs/chatnerds.log"  # (Default: "logs/chatnerds.log") Path to log file
    )
    VERBOSE: int = (
        1  # (Default: 1) Amount of logs written to stdout (0: none, 1: medium, 2: full)
    )
    OPENAI_API_KEY: str = ""  # (Default: "") OpenAI API key
    WHISPER_TRANSCRIPTION_MODEL_NAME: str = (
        "base"  # (Default: "base") Name of the model to use for transcribing audios: tiny, base, small, medium, large
    )
    TRANSCRIPT_ADD_SUMMARY: bool = (
        False  # (Default: False) Include a summary of the transcription in the output file
    )
    YOUTUBE_GROUP_BY_AUTHOR: bool = (
        True  # (Default: True) Group downloaded videos by channel
    )
    YOUTUBE_SLEEP_SECONDS_BETWEEN_DOWNLOADS: int = (
        3  # (Default: 3) Number of seconds to sleep between downloads
    )
    YOUTUBE_ADD_DATE_PREFIX: bool = (
        True  # (Default: True) Prefix all episodes with an ISO8602 formatted date of when they were published. Useful to ensure chronological ordering
    )
    YOUTUBE_SLUGIFY_PATHS: bool = (
        True  # (Default: True) Clean all folders and filename of potentially weird characters that might cause trouble with one or another target filesystem
    )
    YOUTUBE_MAXIMUM_EPISODE_COUNT: int = (
        30  # (Default: 30) Only download the given number of episodes per youtube channel. Useful if you don't really need the entire backlog. Set 0 to disable limit
    )
    PODCAST_UPDATE_ARCHIVE: bool = (
        True  # (Default: True) Force the archiver to only update the feeds with newly added episodes. As soon as the first old episode found in the download directory, further downloading is interrupted
    )
    PODCAST_ADD_DATE_PREFIX: bool = (
        True  # (Default: True) Prefix all episodes with an ISO8602 formatted date of when they were published. Useful to ensure chronological ordering
    )
    PODCAST_SLUGIFY_PATHS: bool = (
        True  # (Default: True) Clean all folders and filename of potentially weird characters that might cause trouble with one or another target filesystem
    )
    PODCAST_GROUP_BY_AUTHOR: bool = (
        True  # (Default: True) Create a subdirectory for each feed (named with their titles) and put the episodes in there
    )
    PODCAST_MAXIMUM_EPISODE_COUNT: int = (
        30  # (Default: 30) Only download the given number of episodes per podcast feed. Useful if you don't really need the entire backlog. Set 0 to disable limit
    )
    PODCAST_SHOW_PROGRESS_BAR: bool = (
        True  # (Default: True) Show a progress bar while downloading
    )
    TEST: str = ""  # (Default: "") Test config value

    def __init__(self, config: dict = None):
        # ref: https://alexandra-zaharia.github.io/posts/python-configuration-and-dataclasses/
        environment_config = Config.environment_instance()

        # initialise with environment values
        for field in fields(environment_config):
            setattr(self, field.name, getattr(environment_config, field.name))

        # initialise with config received
        for config_key, config_value in config.items() if config else []:
            if hasattr(self, config_key):
                setattr(self, config_key, config_value)

    def __str__(self):
        response = "Config:\n"
        for field in fields(self):
            if field.name.startswith("_"):
                continue
            response += f"  {field.name}: {getattr(self, field.name)}\n"
        return response

    @classmethod
    def default_instance(cls):
        if cls._default_instance is None:
            cls._default_instance = cls.__new__(cls)

        return cls._default_instance

    @classmethod
    def environment_instance(cls):
        if cls._environment_instance is None:
            cls._environment_instance = cls.__new__(cls)

            # Try to load config from environment variables
            for field in fields(cls):
                setattr(
                    cls._environment_instance,
                    field.name,
                    os.environ.get(
                        field.name, getattr(cls._environment_instance, field.name)
                    ),
                )

            # fix types
            cls._environment_instance.VERBOSE = int(cls._environment_instance.VERBOSE)
            cls._environment_instance.TRANSCRIPT_ADD_SUMMARY = bool(
                cls._environment_instance.TRANSCRIPT_ADD_SUMMARY
            )
            cls._environment_instance.YOUTUBE_GROUP_BY_AUTHOR = bool(
                cls._environment_instance.YOUTUBE_GROUP_BY_AUTHOR
            )
            cls._environment_instance.YOUTUBE_SLEEP_SECONDS_BETWEEN_DOWNLOADS = int(
                cls._environment_instance.YOUTUBE_SLEEP_SECONDS_BETWEEN_DOWNLOADS
            )
            cls._environment_instance.YOUTUBE_ADD_DATE_PREFIX = bool(
                cls._environment_instance.YOUTUBE_ADD_DATE_PREFIX
            )
            cls._environment_instance.YOUTUBE_SLUGIFY_PATHS = bool(
                cls._environment_instance.YOUTUBE_SLUGIFY_PATHS
            )
            cls._environment_instance.YOUTUBE_MAXIMUM_EPISODE_COUNT = int(
                cls._environment_instance.YOUTUBE_MAXIMUM_EPISODE_COUNT
            )
            cls._environment_instance.PODCAST_ADD_DATE_PREFIX = bool(
                cls._environment_instance.PODCAST_ADD_DATE_PREFIX
            )
            cls._environment_instance.PODCAST_MAXIMUM_EPISODE_COUNT = int(
                cls._environment_instance.PODCAST_MAXIMUM_EPISODE_COUNT
            )
            cls._environment_instance.PODCAST_SHOW_PROGRESS_BAR = bool(
                cls._environment_instance.PODCAST_SHOW_PROGRESS_BAR
            )
            cls._environment_instance.PODCAST_UPDATE_ARCHIVE = bool(
                cls._environment_instance.PODCAST_UPDATE_ARCHIVE
            )
            cls._environment_instance.PODCAST_SLUGIFY_PATHS = bool(
                cls._environment_instance.PODCAST_SLUGIFY_PATHS
            )
            cls._environment_instance.PODCAST_GROUP_BY_AUTHOR = bool(
                cls._environment_instance.PODCAST_GROUP_BY_AUTHOR
            )

        return cls._environment_instance

    def bootstrap(self):
        config = Config.environment_instance()
        nerds_path = Path(config.NERDS_DIRECTORY_PATH)
        if not nerds_path.exists():
            nerds_path.mkdir(parents=True, exist_ok=True)

    def get_nerd_config(self, nerd_name: Optional[str] = None):
        config = Config.environment_instance()

        nerd_name = nerd_name or config.get_active_nerd() or None
        if not nerd_name:
            raise ValueError("No nerd name provided")

        default_config = self.read_nerd_config(Path(__file__).parent)
        active_nerd_config = self.read_nerd_config(
            config.get_nerd_base_path(nerd_name=nerd_name)
        )

        merged_nerd_config = self.merge_config(default_config, active_nerd_config)
        merged_nerd_config["chroma"]["persist_directory"] = str(
            Path(config.get_nerd_base_path(nerd_name=nerd_name), self._DB_NAME)
        )

        return merged_nerd_config

    def get_nerd_base_path(self, nerd_name: Optional[str] = None) -> Union[str, Path]:
        config = Config.environment_instance()
        nerd_name = nerd_name or config.get_active_nerd() or None
        if not nerd_name:
            raise ValueError("No nerd name provided")

        return Path(config.NERDS_DIRECTORY_PATH, nerd_name)

    def get_nerd_youtube_sources(self, nerd_name: Optional[str] = None) -> List[str]:
        config = Config.environment_instance()
        nerd_name = nerd_name or config.get_active_nerd() or None
        if not nerd_name:
            raise ValueError("No nerd name provided")

        nerd_youtube_sources_path = Path(
            config.NERDS_DIRECTORY_PATH, nerd_name, self._YOUTUBE_SOURCES_FILENAME
        )

        youtube_playlists: List[str] = []
        if os.path.exists(nerd_youtube_sources_path):
            with open(nerd_youtube_sources_path) as file_handler:
                for line in file_handler:
                    line = self.strip_source_url(line)
                    if not line or line in youtube_playlists:
                        continue

                    youtube_playlists.append(line)
        return youtube_playlists

    def get_nerd_podcast_sources(self, nerd_name: Optional[str] = None) -> List[str]:
        config = Config.environment_instance()
        nerd_name = nerd_name or config.get_active_nerd() or None
        if not nerd_name:
            raise ValueError("No nerd name provided")

        nerd_podcast_sources_path = Path(
            config.NERDS_DIRECTORY_PATH, nerd_name, self._PODCAST_SOURCES_FILENAME
        )

        podcast_feeds: List[str] = []
        if os.path.exists(nerd_podcast_sources_path):
            with open(nerd_podcast_sources_path) as file_handler:
                for line in file_handler:
                    line = self.strip_source_url(line)
                    if not line or line in podcast_feeds:
                        continue

                    podcast_feeds.append(line)
        return podcast_feeds

    def activate_nerd(self, nerd_name: str | None = None):
        # validate nerd path exists
        if nerd_name and not self.get_nerd_base_path(nerd_name=nerd_name).exists():
            raise ValueError(
                f"Nerd '{nerd_name}' does not exist. Create it first. See chatnerds nerd --help."
            )

        # save active nerd
        os.environ["ACTIVE_NERD"] = nerd_name
        self.ACTIVE_NERD = nerd_name

        # write to disk runtime .env file
        self.dump_runtime_dotenv()

    def get_active_nerd(self) -> str | None:
        return self.ACTIVE_NERD

    def dump_runtime_dotenv(self):
        with open(_RUNTIME_DOTENV_PATH, "w") as file_handler:
            file_handler.write("%s=%s\n" % ("ACTIVE_NERD", self.ACTIVE_NERD or ""))

    @classmethod
    def read_nerd_config(cls, path: Union[Path, str]) -> Dict[str, Any]:
        path = Path(path)
        if path.is_dir():
            path = path / cls._NERD_CONFIG_FILENAME
        with open(path) as f:
            return yaml.safe_load(f)

    @classmethod
    def merge_config(cls, a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
        c = {}
        always_merger.merge(c, a)
        always_merger.merge(c, b)
        return c

    @classmethod
    def strip_source_url(cls, line: str) -> Union[str, None]:
        line = line.strip()
        if line.startswith("#"):  # Comment as the first character
            return None

        line = line.split(" #")[0]  # Comment after the source url
        line = line.strip()
        if line != "":
            return line

        return None
