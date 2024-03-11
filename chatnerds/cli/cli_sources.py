import logging
import typer
from pathlib import Path
from chatnerds.enums import DownloadSourceEnum
from chatnerds.cli.cli_utils import (
    DownloadSourceOption,
    DirectoryFilterArgument,
    DryRunOption,
    validate_confirm_active_nerd,
    TqdmHolder,
)
from chatnerds.utils import get_filtered_directories, copy_files_between_directories
from chatnerds.config import Config


_global_config = Config.environment_instance()
app = typer.Typer()


@app.command(
    "download-sources",
    help="Download audio files (.mp3) of youtube and podcast sources",
)
def download_sources(source: DownloadSourceOption = None):
    validate_confirm_active_nerd()

    # Local import of YoutubeDownloader
    from chatnerds.tools.youtube_downloader import YoutubeDownloader

    if not source or source == DownloadSourceEnum.youtube:
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

    # Local import of PodcastDownloader
    from chatnerds.tools.podcast_downloader import PodcastDownloader

    if not source or source == DownloadSourceEnum.podcasts:
        podcast_downloader = PodcastDownloader(
            feeds=_global_config.get_nerd_podcast_sources(), config=_global_config
        )
        podcast_downloader.run(
            output_path=str(
                Path(_global_config.get_nerd_base_path(), "downloads", "podcasts")
            )
        )


@app.command(
    "transcribe-downloads",
    help="Transcribe downloaded audio files into transcript files (.transcript)",
)
def transcribe_downloads(
    directory_filter: DirectoryFilterArgument = None,
    dry_run: DryRunOption = False,
):
    validate_confirm_active_nerd()

    source_directories = get_filtered_directories(
        directory_filter=directory_filter,
        base_path=Path(_global_config.get_nerd_base_path(), "downloads"),
    )

    # Local import of AudioTranscriber
    from chatnerds.tools.audio_transcriber import AudioTranscriber

    for source_directory in source_directories:
        audio_transcriber = AudioTranscriber(config=_global_config)

        tqdm_holder = TqdmHolder(desc="Transcribing sources", ncols=80)
        audio_transcriber.on("start", tqdm_holder.start)
        audio_transcriber.on("update", tqdm_holder.update)
        audio_transcriber.on("end", tqdm_holder.close)

        results, errors = audio_transcriber.run(
            source_directory=str(source_directory), dry_run=dry_run
        )

        tqdm_holder.close()
        logging.info(
            f"{'[Dry run] ' if dry_run else ''}{len(results)} audios transcribed successfully with {len(errors)} errors...."
        )

        if dry_run == True:
            return

        logging.info("Copying transcript files to source_documents/ ....")

        source_directory_relative = source_directory.relative_to(
            Path(_global_config.get_nerd_base_path(), "downloads")
        )

        try:
            copy_files_between_directories(
                "**/*.transcript",
                src_dir=Path(
                    _global_config.get_nerd_base_path(),
                    "downloads",
                    source_directory_relative,
                ),
                dst_dir=Path(
                    _global_config.get_nerd_base_path(),
                    "source_documents",
                    source_directory_relative,
                ),
            )
        except Exception as e:
            logging.error(
                "Error moving transcript files to directory 'source_documents'",
                exc_info=e,
            )
