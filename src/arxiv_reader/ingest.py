"""Reads the papers arXiv announced today from its RSS feed."""

import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime

FEED = "https://rss.arxiv.org/rss/cs.AI+cs.CL+cs.LG+cs.IR+cs.SE+cs.MA"
NS = {"arxiv": "http://arxiv.org/schemas/atom"}


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    categories: str

    @property
    def text(self) -> str:
        """What the interest memory embeds."""
        return f"{self.title}\n{self.abstract}"

    @property
    def link(self) -> str:
        return f"https://arxiv.org/abs/{self.id}"

    def first_sentences(self, n: int) -> list[str]:
        return re.split(r"(?<=\.)\s+", self.abstract)[:n]


def parse(xml: bytes) -> tuple[date | None, list[Paper]]:
    """Returns the announcement date and its new and cross-listed papers, deduped by id."""
    channel = ET.fromstring(xml).find("channel")
    pub = channel.findtext("pubDate")
    papers: dict[str, Paper] = {}
    for item in channel.iter("item"):
        if item.findtext("arxiv:announce_type", namespaces=NS) not in ("new", "cross"):
            continue
        link, title = item.findtext("link", ""), item.findtext("title", "")
        if not link or not title:
            continue  # one malformed item should not cost the whole day
        pid = link.rsplit("/", 1)[-1]
        abstract = item.findtext("description", "").split("Abstract:", 1)[-1]
        papers.setdefault(pid, Paper(
            id=pid,
            title=" ".join(title.split()),
            abstract=" ".join(abstract.split()),
            categories=", ".join(c.text for c in item.iter("category")),
        ))
    return (parsedate_to_datetime(pub).date() if pub else None), list(papers.values())


def fetch() -> tuple[date | None, list[Paper]]:
    with urllib.request.urlopen(FEED, timeout=60) as resp:
        return parse(resp.read())
