# Resources:
# Download youtube video: https://dev.to/stokry/download-youtube-video-to-mp3-with-python-26p
# video attributes: title, description, views, rating, length, keywords, thumbnail_url, video_id, age_restricted, channel_id, channel_url, watch_url, captions, publish_date, start_time, end_time, category, tags

import re
import unicodedata
from pathlib import Path
from time import sleep
from dateutil.parser import parse as dateparse
import traceback
# from pytube import YouTube, Playlist
from pytubefix import YouTube, Playlist
import music_tag
from typing import List, Optional, Union
import os
import logging
from .config import Config

class YoutubeDownloader:
    source_urls: set
    config: Config = Optional[Config]

    def __init__(self, source_urls: List[str] = [], config: Config = None):
        if config:
            self.config = config
        else:
            self.config = Config.environment_instance()

        self.source_urls = set()
        self.add_sources(source_urls)

    def add_sources(self, source_urls: Union[str, List]):
        if source_urls is str:
            source_urls = [source_urls]
        
        for source_url in source_urls:
            if "watch?v=" in source_url:
                self.source_urls.add(source_url)
            elif "playlist?list=" in source_url:
                playlist = Playlist(source_url)
                self.source_urls = self.source_urls.union(playlist.video_urls)
            elif "channel/" in source_url:     # See: https://github.com/pytube/pytube/issues/619
                raise ValueError("Channel URLs not supported yet. See: https://github.com/pytube/pytube/issues/619")
                # channel = Channel(source_url)
                # self.source_urls = self.source_urls.union(channel.video_urls)
    
    def run(self, output_path = ".") -> List[str]:
        logging.info(f"Running youtube downloader with {len(self.source_urls)} urls")
        
        audio_output_file_paths = []
        for source_url in self.source_urls:
            logging.debug(f"  - downloading {source_url}")
            audio_output_file_path = self.download(url=source_url, output_path=output_path)
            if audio_output_file_path:
                audio_output_file_paths.append(audio_output_file_path)
            logging.debug(f"  ...waiting {self.config.YOUTUBE_SLEEP_SECONDS_BETWEEN_DOWNLOADS} seconds")
            sleep(self.config.YOUTUBE_SLEEP_SECONDS_BETWEEN_DOWNLOADS)

        logging.info("\nyoutube downloader finished....")
        return audio_output_file_paths

    def download(self, url: str, output_path = ".") -> Union[str, None]:
        """Download audio from YouTube video url."""
        yt = YouTube(url)

        channel_title = yt.author
        video_title = yt.title

        # Audio output file
        audio_output_filename = f"{video_title}.mp3"
        if self.config.YOUTUBE_ADD_DATE_PREFIX is True and yt.publish_date is not None:
            date_str = yt.publish_date.strftime("%Y-%m-%d")
            audio_output_filename = f"{date_str} {audio_output_filename}"
        elif yt.publish_date is None:
            logging.info(f"Video {video_title} has no publish date. Skipping date prefix.")
        
        # Slugify / fix titles
        if self.config.YOUTUBE_SLUGIFY_PATHS:
            audio_output_filename = self.slugifyString(audio_output_filename)
            channel_title = self.slugifyString(channel_title)
        else:
            audio_output_filename = audio_output_filename.replace(os.path.pathsep, "_")
            audio_output_filename = audio_output_filename.replace(os.path.sep, "_")
            channel_title = channel_title.replace(os.path.pathsep, "_")
            channel_title = channel_title.replace(os.path.sep, "_")
        
        # Output path
        if self.config.YOUTUBE_GROUP_BY_AUTHOR:
            output_path = str(Path(output_path, channel_title))

        audio_output_file_path = os.path.join(output_path, audio_output_filename)
        # logging.debug(f"audio_output_file_path: {audio_output_file_path}")
        if os.path.exists(audio_output_file_path):
            logging.debug(f"Audio file {audio_output_file_path} already exists. Skipping download.")
            YoutubeDownloader.write_tags(audio_output_file_path, yt)

            return audio_output_file_path

        # Extract audio from video
        audio_output_file_path = None
        try:
            video = yt.streams.get_audio_only()
            audio_output_file_path = video.download(
                mp3=True,
                output_path = output_path,
                filename = audio_output_filename,
                filename_prefix = None,
                skip_existing = True,
                # timeout: Optional[int] = None,
                # max_retries: Optional[int] = 0
            )
        except Exception as err:
            logging.error(f"Could not download audio from video {url}: {str(err)}")
            traceback.print_exc()
            return None

        file_stats = os.stat(audio_output_file_path)
        if file_stats.st_size > 100000000:
            logging.warning(f"Size of audio file {file_stats.st_size / 1024}MB exceeds maximum of 100MB.")
            return
        
        # Validate downloaded audio file
        if os.path.exists(audio_output_file_path):
            logging.debug(f"Video downloaded successfully in '{audio_output_file_path}'")

            YoutubeDownloader.write_tags(audio_output_file_path, yt)
            return audio_output_file_path
        else:
            logging.error(f"{yt.title} could not be downloaded!")
            return None


    @staticmethod
    def slugifyString(filename: str) -> str:
        filename = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore")
        filename = re.sub("[^\w\s\-\.]", "", filename.decode("ascii")).strip()
        filename = re.sub("[-\s]+", "-", filename)

        return filename

    @staticmethod
    def write_tags(filepath: str, yt: YouTube) -> None:
        music_tag_file = music_tag.load_file(filepath)
        music_tag_file["title"] = yt.title
        music_tag_file["album"] = yt.author
        music_tag_file["artist"] = yt.author
        music_tag_file["comment"] = yt.watch_url
        music_tag_file.save()
