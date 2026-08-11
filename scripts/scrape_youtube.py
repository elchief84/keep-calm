"""YouTube comment scraper for Keep Calm dataset.

Extracts comments from English and Italian YouTube videos across multiple
categories to build a diverse corpus of real human communication.

Usage:
    python scripts/scrape_youtube.py --api-key YOUR_KEY [--max-videos 20] [--max-comments 200]

Requirements:
    - YouTube Data API v3 key (https://console.cloud.google.com/)
    - Quota: 10,000 units/day. Each search ~100 units, each comment page ~1 unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE_EN = DATA_DIR / "youtube_en.jsonl"
OUTPUT_FILE_IT = DATA_DIR / "youtube_it.jsonl"
CHECKPOINT_FILE = DATA_DIR / "youtube_checkpoint.json"


# ---- Configuration ----

@dataclass
class SearchConfig:
    """A search configuration for a category + language combination."""

    query: str
    language: str  # "en" or "it"
    relevance_language: str  # ISO 639-1 for YouTube API
    region_code: str  # ISO 3166-1 alpha-2
    category: str  # keep_calm domain tag
    max_videos: int = 10
    max_comments_per_video: int = 100


# Diverse categories to capture varied communication styles
SEARCH_CONFIGS: list[SearchConfig] = [
    # English
    SearchConfig("tech review 2025", "en", "en", "US", "tech", max_videos=8),
    SearchConfig("software engineering debate", "en", "en", "US", "tech", max_videos=6),
    SearchConfig("open source controversy", "en", "en", "US", "tech", max_videos=5),
    SearchConfig("politics debate 2025", "en", "en", "US", "news", max_videos=5),
    SearchConfig("gaming review drama", "en", "en", "US", "gaming", max_videos=5),
    SearchConfig("music reaction 2025", "en", "en", "US", "entertainment", max_videos=5),
    SearchConfig("science documentary", "en", "en", "US", "education", max_videos=5),
    # Italian
    SearchConfig("recensione tech 2025", "it", "it", "IT", "tech", max_videos=8),
    SearchConfig("sviluppo software opinioni", "it", "it", "IT", "tech", max_videos=6),
    SearchConfig("dibattito politico 2025", "it", "it", "IT", "news", max_videos=5),
    SearchConfig("gaming recensione polemica", "it", "it", "IT", "gaming", max_videos=5),
    SearchConfig("musica reazione", "it", "it", "IT", "entertainment", max_videos=5),
    SearchConfig("documentario scienza", "it", "it", "IT", "education", max_videos=5),
    SearchConfig(
        "calcio discussione", "it", "it", "IT", "sports", max_videos=5
    ),
]

# Comment quality filters
MIN_COMMENT_LENGTH = 10
MAX_COMMENT_LENGTH = 2000
BLOCKED_PATTERNS = [
    "http://",
    "https://",
    "check out my channel",
    "subscribe to my",
    "follow me on",
]


# ---- Data model ----

def make_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---- YouTube API ----

class YouTubeScraper:
    def __init__(self, api_key: str) -> None:
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.quota_used = 0

    def search_videos(
        self,
        config: SearchConfig,
    ) -> list[dict]:
        """Search for videos matching a config. Returns video metadata."""
        print(f"  Searching: '{config.query}' ({config.language})...", end=" ")

        try:
            request = self.youtube.search().list(
                q=config.query,
                part="snippet",
                type="video",
                relevanceLanguage=config.relevance_language,
                regionCode=config.region_code,
                maxResults=min(config.max_videos, 50),
                order="relevance",
                safeSearch="none",
            )
            response = request.execute()
            self.quota_used += 100  # search.list costs 100 units

        except HttpError as e:
            print(f"API error: {e}")
            return []

        videos = []
        for item in response.get("items", []):
            snippet = item["snippet"]
            videos.append({
                "video_id": item["id"]["videoId"],
                "title": snippet["title"],
                "description": snippet["description"][:500],
                "channel_title": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
            })

        print(f"{len(videos)} videos found")
        return videos

    def get_comments(
        self,
        video_id: str,
        max_comments: int,
    ) -> list[dict]:
        """Fetch comments for a video. Returns raw comment data."""
        comments = []
        page_token = None

        while len(comments) < max_comments:
            try:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=page_token,
                    order="relevance",
                    textFormat="plainText",
                )
                response = request.execute()
                self.quota_used += 1  # commentThreads.list costs 1 unit

            except HttpError as e:
                if "commentsDisabled" in str(e):
                    break
                if "quotaExceeded" in str(e):
                    raise
                break

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet["textDisplay"].strip()
                if self._is_valid_comment(text):
                    comments.append({
                        "comment_id": item["id"],
                        "text": text,
                        "author": snippet["authorDisplayName"],
                        "likes": snippet["likeCount"],
                        "published_at": snippet["publishedAt"],
                        "updated_at": snippet["updatedAt"],
                    })

            page_token = response.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.3)  # Rate limiting

        return comments

    @staticmethod
    def _is_valid_comment(text: str) -> bool:
        if len(text) < MIN_COMMENT_LENGTH:
            return False
        if len(text) > MAX_COMMENT_LENGTH:
            return False
        text_lower = text.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern in text_lower:
                return False
        return True


# ---- Data pipeline ----

def video_to_examples(
    video: dict,
    comments: list[dict],
    config: SearchConfig,
) -> list[dict]:
    """Convert video + comments into Keep Calm annotation-ready examples."""
    examples = []
    video_context = f"Video: {video['title']} — Channel: {video['channel_title']}"

    for comment in comments:
        examples.append({
            "id": f"yt-{make_id(comment['comment_id'])}",
            "text": comment["text"],
            "language": config.language,
            "domain": f"youtube_{config.category}",
            "source": "youtube",
            "source_id": comment["comment_id"],
            "context": video_context,
            "metadata": {
                "video_id": video["video_id"],
                "video_title": video["title"],
                "channel": video["channel_title"],
                "category": config.category,
                "comment_likes": comment["likes"],
                "comment_published_at": comment["published_at"],
            },
            "annotations": {
                "communication_risk": None,
                "tones": [],
                "intent": None,
                "explanation": None,
                "needs_attention": None,
            },
        })

    return examples


def save_examples(examples: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint() -> dict | None:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None


# ---- Main ----

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape YouTube comments for Keep Calm dataset")
    parser.add_argument("--api-key", required=True, help="YouTube Data API v3 key")
    parser.add_argument("--max-videos", type=int, default=0, help="Override max videos per config")
    parser.add_argument("--max-comments", type=int, default=0, help="Override max comments per video")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Show configs without scraping")
    args = parser.parse_args()

    if args.dry_run:
        print("Search configurations:\n")
        for i, cfg in enumerate(SEARCH_CONFIGS):
            print(f"  [{i}] {cfg.query!r} | lang={cfg.language} | "
                  f"videos={cfg.max_videos} | comments={cfg.max_comments_per_video}")
        return

    scraper = YouTubeScraper(args.api_key)

    # Resume from checkpoint
    checkpoint = load_checkpoint() if args.resume else None
    completed_queries: set[str] = set(checkpoint.get("completed_queries", [])) if checkpoint else set()

    total_videos = 0
    total_comments = 0
    en_examples: list[dict] = []
    it_examples: list[dict] = []

    # Clear output files on fresh run
    if not args.resume:
        for f in [OUTPUT_FILE_EN, OUTPUT_FILE_IT]:
            if f.exists():
                f.unlink()

    for cfg in SEARCH_CONFIGS:
        if cfg.query in completed_queries:
            print(f"\n[{cfg.query}] Skipping (already completed)")
            continue

        mv = args.max_videos or cfg.max_videos
        mc = args.max_comments or cfg.max_comments_per_video
        print(f"\n{'='*60}")
        print(f"Config: {cfg.query!r} | Lang: {cfg.language} | "
              f"Category: {cfg.category} | Max videos: {mv} | Max comments/video: {mc}")
        print(f"Quota used so far: {scraper.quota_used}")

        videos = scraper.search_videos(cfg)
        if args.max_videos:
            videos = videos[: args.max_videos]
        else:
            videos = videos[: cfg.max_videos]

        for video in videos:
            print(f"  Video: {video['title'][:80]}... ({video['video_id']})")

            try:
                comments = scraper.get_comments(video["video_id"], mc)
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print(f"\nQuota exceeded! Saving progress...")
                    save_checkpoint({
                        "completed_queries": list(completed_queries),
                        "quota_used": scraper.quota_used,
                        "total_videos": total_videos,
                        "total_comments": total_comments,
                    })
                    sys.exit(0)
                raise

            print(f"    {len(comments)} valid comments")

            if comments:
                examples = video_to_examples(video, comments, cfg)
                if cfg.language == "en":
                    en_examples.extend(examples)
                else:
                    it_examples.extend(examples)

            total_videos += 1
            total_comments += len(comments)

        completed_queries.add(cfg.query)

    # Save outputs
    if en_examples:
        save_examples(en_examples, OUTPUT_FILE_EN)
    if it_examples:
        save_examples(it_examples, OUTPUT_FILE_IT)

    # Save final checkpoint
    save_checkpoint({
        "completed_queries": list(completed_queries),
        "quota_used": scraper.quota_used,
        "total_videos": total_videos,
        "total_comments": total_comments,
    })

    # Summary
    print(f"\n{'='*60}")
    print("SCRAPE COMPLETE")
    print(f"{'='*60}")
    print(f"Videos processed:     {total_videos}")
    print(f"Comments extracted:   {total_comments}")
    print(f"EN examples saved:    {len(en_examples)} -> {OUTPUT_FILE_EN.name}")
    print(f"IT examples saved:    {len(it_examples)} -> {OUTPUT_FILE_IT.name}")
    print(f"Total quota used:     {scraper.quota_used} / 10,000")
    print(f"Quota remaining:      {10000 - scraper.quota_used}")

    if total_comments == 0:
        print("\nNo comments extracted. Check your API key and quota.")
    else:
        print(f"\nNext step: run annotation pipeline on these {total_comments} unlabeled examples.")


if __name__ == "__main__":
    main()
