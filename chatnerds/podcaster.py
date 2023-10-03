from __future__ import annotations

import re
from typing import List, Optional
import unicodedata
import urllib.error
import xml.etree.ElementTree as etree
from contextlib import nullcontext
from os import makedirs, path, remove
# from typing import TYPE_CHECKING
from urllib.parse import urlparse
import logging
import feedparser
import requests
from dateutil.parser import parse as dateparse
from feedparser import CharacterEncodingOverride
from tqdm import tqdm
from .config import Config

# podcast_archiver settings: https://github.com/janw/podcast-archiver/blob/main/podcast_archiver/config.py
USER_AGENT = f"podcast-archiver/dev (https://github.com/janw/podcast-archiver)"

# if TYPE_CHECKING:
#     from podcast_archiver.config import Settings

class PodcastDownloader:
    _global_info_keys = [
        "author",
        "language",
        "link",
        "subtitle",
        "title",
    ]
    _episode_info_keys = [
        "author",
        "link",
        "subtitle",
        "title",
    ]
    _date_keys = [
        "published",
    ]

    feedlist: set
    config: Config = Optional[Config]
    def __init__(self, feeds: List[str] = [], opml_files: List[str] = [], config: Config = None):
        if config:
            self.config = config
        else:
            self.config = Config.environment_instance()

        self.feedlist = set()
        for feed in feeds or []:
            self.addFeed(feed)
        for opml in opml_files or []:
            self.parseOpmlFile(opml)

        self.session = requests.Session()
        self.session.headers.update({"user-agent": USER_AGENT})

    def addFeed(self, feed):
        if path.isfile(feed):
            with open(feed, "r") as fp:
                self.feedlist = self.feedlist.union(set(fp.read().strip().splitlines()))
        else:
            self.feedlist.add(feed)

    def parseOpmlFile(self, opml):
        with opml.open("r") as file:
            tree = etree.fromstringlist(file)

        for feed in [
            node.get("xmlUrl") for node in tree.findall("*/outline/[@type='rss']") if node.get("xmlUrl") is not None
        ]:
            self.addFeed(feed)

    def processFeed(self, feed_url, output_path="."):
        if self.config.VERBOSE > 0:
            print(f"\nDownloading archive for: {feed_url}\n1. Gathering link list ...", end="", flush=True)

        linklist, feed_info = self.processPodcastLink(feed_url, output_path=output_path)
        if self.config.VERBOSE == 1:
            print("%d episodes to process" % len(linklist))
        if self.config.VERBOSE > 2:
            print("\n\tFeed info:")
            for key, value in feed_info.items():
                print("\t * %10s: %s" % (key, value))
            print()
        if not linklist:
            return
        if self.config.VERBOSE == 1:
            print("2. Downloading content ... ", end="")
        elif self.config.VERBOSE > 1:
            print("2. Downloading content ...")
        self.downloadEpisodes(linklist, feed_info, output_path=output_path)

    def run(self, output_path="."):
        if self.config.VERBOSE > 0 and self.config.PODCAST_UPDATE_ARCHIVE:
            print("Updating archive")
        for feed_url in self.feedlist:
            self.processFeed(feed_url, output_path=output_path)
        if self.config.VERBOSE > 0:
            print("\nDone.")

    @staticmethod
    def slugifyString(filename):
        filename = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore")
        filename = re.sub("[^\w\s\-\.]", "", filename.decode("ascii")).strip()
        filename = re.sub("[-\s]+", "-", filename)

        return filename

    def linkToTargetFilename(self, link, feed_info, must_have_ext=False, episode_info=None, output_path="."):
        linkpath = urlparse(link).path
        basename = path.basename(linkpath)
        feed_title = feed_info["title"]

        if self.config.PODCAST_ADD_DATE_PREFIX is True and episode_info:
            date_str = dateparse(episode_info["published"]).strftime("%Y-%m-%d")
            basename = f"{date_str} {basename}"

        _, ext = path.splitext(basename)
        if must_have_ext and not ext:
            return None

        if self.config.PODCAST_SLUGIFY_PATHS:
            basename = self.slugifyString(basename)
            feed_title = self.slugifyString(feed_title)
        else:
            basename.replace(path.pathsep, "_")
            basename.replace(path.sep, "_")
            feed_title.replace(path.pathsep, "_")
            feed_title.replace(path.sep, "_")

        if self.config.PODCAST_GROUP_BY_AUTHOR:
            filename = path.join(output_path, feed_title, basename)
        else:
            filename = path.join(output_path, basename)

        return filename

    def parseFeedToNextPage(self, feedobj):
        # Assuming there will only be one link declared as 'next'
        feed_next_page = [link["href"] for link in feedobj["feed"]["links"] if link["rel"] == "next"]
        if len(feed_next_page) > 0:
            return feed_next_page[0]

    def parseFeedToLinks(self, feedobj):
        # Try different feed episode layouts: 'items' or 'entries'
        episodeList = feedobj.get("items", False) or feedobj.get("entries", False)
        if episodeList:
            linklist = [self.parseEpisode(episode) for episode in episodeList]
            linklist = [link for link in linklist if len(link) > 0]
        else:
            linklist = []

        return linklist

    def parseEpisode(self, episode):
        url = None
        episode_info = {}
        for link in episode["links"]:
            if "type" in link:
                if link["type"].startswith("audio") or link["type"].startswith("video"):
                    url = link["href"]

                if url is not None:
                    for key in self._episode_info_keys + self._date_keys:
                        episode_info[key] = episode.get(key, None)
                    episode_info["url"] = url

        return episode_info

    def getFeedObj(self, feed_url):
        response = self.session.get(feed_url, allow_redirects=True)

        # Escape improper feed-URL
        if not response.ok:
            print("\nQuery returned HTTP error", response.status_code)
            return None

        feedobj = feedparser.parse(response.content)
        # Escape malformatted XML; If the character encoding is wrong, continue as long as the reparsing succeeded
        if feedobj["bozo"] == 1 and not isinstance(feedobj["bozo_exception"], CharacterEncodingOverride):
            print("\nDownloaded feed is malformatted on", feed_url)
            return None

        return feedobj

    def truncateLinkList(self, linklist, feed_info, output_path="."):
        # On given option, run an update, break at first existing episode
        if self.config.PODCAST_UPDATE_ARCHIVE:
            for index, episode_dict in enumerate(linklist):
                link = episode_dict["url"]
                filename = self.linkToTargetFilename(link, feed_info, output_path=output_path)

                if path.isfile(filename):
                    del linklist[index:]
                    if self.config.VERBOSE > 1:
                        print(f" found existing episodes, {len(linklist)} new to process")
                    return True, linklist

        # On given option, crop linklist to maximum number of episodes
        if (max_count := self.config.PODCAST_MAXIMUM_EPISODE_COUNT) > 0 and max_count < len(linklist):
            linklist = linklist[0:max_count]
            if self.config.VERBOSE > 1:
                print(f" reached maximum episode count of {max_count}")
            return True, linklist

        return False, linklist

    def parseFeedInfo(self, feedobj):
        feed_header = feedobj.get("feed", {})
        feed_info = {key: feed_header.get(key, None) for key in self._global_info_keys}
        if feed_info.get("title"):
            return feed_info

        print("✗ Feed is missing title information.")
        return None

    def processPodcastLink(self, feed_next_page, output_path="."):
        feed_info = None
        linklist = []
        while True:
            if not (feedobj := self.getFeedObj(feed_next_page)):
                break

            if not feed_info:
                feed_info = self.parseFeedInfo(feedobj)
                if not feed_info:
                    return [], {}

            # Parse the feed object for episodes and the next page
            linklist += self.parseFeedToLinks(feedobj)
            feed_next_page = self.parseFeedToNextPage(feedobj)
            was_truncated, linklist = self.truncateLinkList(linklist, feed_info, output_path=output_path)

            if not feed_next_page or was_truncated:
                break

            if self.config.VERBOSE > 0:
                print(".", end="", flush=True)

        if self.config.VERBOSE == 1:
            print(" ", end="", flush=True)
        linklist.reverse()
        return linklist, feed_info

    def checkEpisodeExistsPreflight(self, link, *, feed_info, episode_dict, output_path="."):
        # Check existence once ...
        filename = self.linkToTargetFilename(link, feed_info=feed_info, episode_info=episode_dict, output_path=output_path)

        if self.config.VERBOSE > 1:
            print("\tLocal filename:", filename)

        if path.isfile(filename):
            if self.config.VERBOSE > 1:
                print("\t✓ Already exists.")
            return None

        return filename

    def logDownloadHeader(self, link, episode_dict, *, index, total):
        if self.config.VERBOSE == 1:
            print("\r2. Downloading episodes ... {0}/{1}".format(index + 1, total), end="", flush=True)
        elif self.config.VERBOSE > 1:
            print("\n\tDownloading episode no. {0}/{1}:\n\t{2}".format(index + 1, total, link))
        if self.config.VERBOSE > 2:
            print("\tEpisode info:")
            for key, value in episode_dict.items():
                print("\t * %10s: %s" % (key, value))

    def processResponse(self, response, *, filename, feed_info, episode_dict, output_path="."):
        # Check existence another time, with resolved link
        new_filename = self.linkToTargetFilename(response.url, feed_info, must_have_ext=True, episode_info=episode_dict, output_path=output_path)

        if new_filename and new_filename != filename:
            filename = new_filename
            if self.config.VERBOSE > 1:
                print("\tResolved filename:", filename)

            if path.isfile(filename):
                if self.config.VERBOSE > 1:
                    print("\t✓ Already exists.")
                return

        # Create the subdir, if it does not exist
        if target_dir := path.dirname(filename):
            makedirs(target_dir, exist_ok=True)

        if self.config.PODCAST_SHOW_PROGRESS_BAR:
            if self.config.VERBOSE < 2:
                print(f"\nDownloading {filename}")
            total_size = int(response.headers.get("content-length", "0"))
            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024)
            callback = progress_bar.update
        else:
            progress_bar = nullcontext()
            callback = None

        with progress_bar, open(filename, "wb") as outfile:
            self.prettyCopyfileobj(response, outfile, callback=callback)

    def downloadEpisode(self, link, *, feed_info, episode_dict, output_path="."):
        filename = self.checkEpisodeExistsPreflight(link, feed_info=feed_info, episode_dict=episode_dict, output_path=output_path)
        if not filename:
            return
        try:
            response = self.session.get(link, stream=True, allow_redirects=True)
            self.processResponse(response, filename=filename, feed_info=feed_info, episode_dict=episode_dict, output_path=output_path)
            if self.config.VERBOSE > 1:
                print("\t✓ Download successful.")
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            if self.config.VERBOSE > 1:
                logging.error("\t✗ Download failed. Query returned '%s'" % error)
        except KeyboardInterrupt:
            if self.config.VERBOSE > 0:
                logging.error("\n\t✗ Unexpected interruption. Deleting unfinished file.")

            remove(filename)
            raise

    def downloadEpisodes(self, linklist, feed_info, output_path="."):
        nlinks = len(linklist)
        for cnt, episode_dict in enumerate(linklist):
            link = episode_dict["url"]

            self.logDownloadHeader(link, episode_dict, index=cnt, total=nlinks)
            self.downloadEpisode(link, feed_info=feed_info, episode_dict=episode_dict, output_path=output_path)

    def prettyCopyfileobj(self, fsrc, fdst, callback, block_size=512 * 1024):
        for chunk in fsrc.iter_content(block_size):
            fdst.write(chunk)
            if callback:
                callback(len(chunk))
