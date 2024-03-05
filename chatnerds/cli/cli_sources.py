import logging
import typer
from pathlib import Path
from chatnerds.enums import SourceEnum
from chatnerds.cli.cli_utils import (
    SourceOption,
    DirectoryFilterArgument,
    validate_confirm_active_nerd,
    TqdmHolder,
)
from chatnerds.utils import get_source_directory_paths
from chatnerds.config import Config

_global_config = Config.environment_instance()
app = typer.Typer()


@app.command(
    "download-sources", help="Download audio files from sources (youtube and podcasts)"
)
def download_sources(source: SourceOption = None):
    validate_confirm_active_nerd()

    # Youtube downloader
    from chatnerds.tools.youtube_downloader import YoutubeDownloader

    if not source or source == SourceEnum.youtube:
        youtube_downloader = YoutubeDownloader(
            source_urls=_global_config.get_nerd_youtube_sources(), config=_global_config
        )

        tqdm_holder = TqdmHolder(desc="Downloading youtube sources", ncols=80)
        youtube_downloader.on("start", tqdm_holder.start)
        youtube_downloader.on("update", tqdm_holder.update)
        youtube_downloader.on("end", tqdm_holder.close)

        results, errors = youtube_downloader.run(
            output_path=str(
                Path(_global_config.get_nerd_base_path(), "downloads", "youtube")
            ),
        )

        tqdm_holder.close()
        logging.info(
            f"{len(results)} youtube audio files downloaded successfully with {len(errors)} errors...."
        )

    # Padcasts downloader
    from chatnerds.tools.podcast_downloader import PodcastDownloader

    if not source or source == SourceEnum.podcasts:
        podcast_downloader = PodcastDownloader(
            feeds=_global_config.get_nerd_podcast_sources(), config=_global_config
        )
        podcast_downloader.run(
            output_path=str(
                Path(_global_config.get_nerd_base_path(), "downloads", "podcasts")
            )
        )


@app.command("transcribe-sources", help="Transcribe audio files")
def transcribe_sources(
    directory_filter: DirectoryFilterArgument = None,
    source: SourceOption = None,
):
    validate_confirm_active_nerd()

    source_directories = get_source_directory_paths(
        directory_filter=directory_filter,
        source=source,
        base_path=_global_config.get_nerd_base_path(),
    )

    # Audio transcriber
    from chatnerds.tools.audio_transcriber import AudioTranscriber

    for source_directory in source_directories:
        audio_transcriber = AudioTranscriber(
            source_directory=str(source_directory), config=_global_config
        )

        tqdm_holder = TqdmHolder(desc="Transcribing sources", ncols=80)
        audio_transcriber.on("start", tqdm_holder.start)
        audio_transcriber.on("update", tqdm_holder.update)
        audio_transcriber.on("end", tqdm_holder.close)

        results, errors = audio_transcriber.run()

        tqdm_holder.close()
        logging.info(
            f"{len(results)} audios transcribed successfully with {len(errors)} errors...."
        )
