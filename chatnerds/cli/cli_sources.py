import logging
import typer
from pathlib import Path
from chatnerds.lib.enums import DownloadSourceEnum
from chatnerds.cli.cli_utils import (
    DownloadSourceOption,
    LimitOption,
    DirectoryFilterArgument,
    DryRunOption,
    validate_confirm_active_nerd,
    TqdmHolder,
)
from chatnerds.lib.helpers import (
    get_filtered_directories,
    copy_files_between_directories,
)
from chatnerds.config import Config


_global_config = Config.environment_instance()
app = typer.Typer()


@app.command(
    "download-sources",
    help="Download audio files (.mp3) of youtube and podcast sources",
)
def download_sources(source: DownloadSourceOption = None, limit: LimitOption = None):

    validate_confirm_active_nerd()

    # Local import of YoutubeDownloader and PodcastDownloader
    from chatnerds.tools.youtube_downloader import YoutubeDownloader
    from chatnerds.tools.podcast_downloader import PodcastDownloader

    try:
        if not source or source == DownloadSourceEnum.youtube:
            youtube_downloader = YoutubeDownloader(
                source_urls=_global_config.get_nerd_youtube_sources(),
                config=_global_config,
            )

            tqdm_holder = TqdmHolder(desc="Downloading youtube sources", ncols=80)
            youtube_downloader.on("start", tqdm_holder.start)
            youtube_downloader.on("update", tqdm_holder.update)
            youtube_downloader.on("end", tqdm_holder.close)

            results, errors = youtube_downloader.run(
                output_path=str(
                    Path(_global_config.get_nerd_base_path(), "downloads", "youtube")
                ),
                limit=limit,
            )

            tqdm_holder.close()
            logging.info(
                f"{len(results)} youtube audio files downloaded successfully with {len(errors)} errors...."
            )

            if len(errors) > 0:
                logging.error(
                    "Errors occurred while downloading audio files from youtube sources. Last error:\n",
                    exc_info=errors[-1],
                )

        if not source or source == DownloadSourceEnum.podcasts:
            podcast_downloader = PodcastDownloader(
                feeds=_global_config.get_nerd_podcast_sources(), config=_global_config
            )
            podcast_downloader.run(
                output_path=str(
                    Path(_global_config.get_nerd_base_path(), "downloads", "podcasts")
                )
            )
    except Exception as e:
        logging.error("Error downloading audios from sources")
        raise e
    except SystemExit:
        raise typer.Abort()


@app.command(
    "transcribe-downloads",
    help="Transcribe downloaded audio files into transcript files (.transcript)",
)
def transcribe_downloads(
    directory_filter: DirectoryFilterArgument = None,
    dry_run: DryRunOption = False,
    limit: LimitOption = None,
):
    validate_confirm_active_nerd()

    # Local import of AudioTranscriber
    from chatnerds.tools.audio_transcriber import AudioTranscriber

    try:
        source_directories = get_filtered_directories(
            directory_filter=directory_filter,
            base_path=Path(_global_config.get_nerd_base_path(), "downloads"),
        )

        for source_directory in source_directories:
            audio_transcriber = AudioTranscriber(config=_global_config)

            tqdm_holder = TqdmHolder(desc="Transcribing sources", ncols=80)
            audio_transcriber.on("start", tqdm_holder.start)
            audio_transcriber.on("update", tqdm_holder.update)
            audio_transcriber.on("write", tqdm_holder.write)
            audio_transcriber.on("end", tqdm_holder.close)

            results, errors = audio_transcriber.run(
                source_directory=str(source_directory),
                dry_run=dry_run,
                limit=limit,
            )

            tqdm_holder.close()
            logging.info(
                f"{'[Dry run] ' if dry_run else ''}{len(results)} audios transcribed successfully with {len(errors)} errors...."
            )

            # Despite dry_run, continue copying .transcript files to source_documents/ directory
            # if dry_run == True:
            #     return

            logging.info("Copying transcript files to source_documents/ ....")

            source_directory_relative = source_directory.relative_to(
                Path(_global_config.get_nerd_base_path(), "downloads")
            )

            try:
                num_files_copied = copy_files_between_directories(
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

                logging.info(f"{num_files_copied} transcript files copied successfully")
            except Exception as e:
                logging.error(
                    "Error moving transcript files to directory 'source_documents'",
                    exc_info=e,
                )
    except Exception as e:
        logging.error("Error transcribing downloaded audio files")
        raise e
    except SystemExit:
        raise typer.Abort()
