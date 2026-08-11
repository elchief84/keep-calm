"""Scrape heated Reddit threads and comments.

Extracts medium/high-risk communication from debate-focused subreddits.
Uses Reddit's free JSON API (no auth needed, just add .json to URLs).

Output: data/reddit_en.jsonl and data/reddit_it.jsonl
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_EN = PROJECT_ROOT / "data" / "reddit_en.jsonl"
OUTPUT_IT = PROJECT_ROOT / "data" / "reddit_it.jsonl"

REDDIT_JSON = "https://www.reddit.com"

SUBREDDITS_EN = [
    "changemyview",
    "unpopularopinion",
    "AmItheAsshole",
    "programming",
    "ExperiencedDevs",
    "sysadmin",
    "opensource",
]

SUBREDDITS_IT = [
    "italy",
    "italia",
    "Universitaly",
    "ItaliaCareerAdvice",
    "ItalyMotori",
]


def fetch_posts(subreddit: str, limit: int = 15) -> list[dict]:
    """Fetch hot/controversial posts from a subreddit."""
    posts = []
    for sort in ["controversial", "hot"]:
        url = f"{REDDIT_JSON}/r/{subreddit}/{sort}.json"
        headers = {"User-Agent": "KeepCalm/1.0 (research dataset)"}
        resp = requests.get(url, params={"limit": limit}, headers=headers, timeout=15)
        if resp.status_code != 200:
            continue
        data = resp.json()
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            if post_data.get("stickied"):
                continue
            posts.append({
                "id": post_data["id"],
                "title": post_data["title"],
                "selftext": post_data.get("selftext", ""),
                "subreddit": subreddit,
                "permalink": post_data.get("permalink", ""),
                "num_comments": post_data.get("num_comments", 0),
            })
        time.sleep(0.5)
    return posts


def fetch_comments(permalink: str, max_comments: int = 30) -> list[str]:
    """Fetch comments from a post."""
    url = f"{REDDIT_JSON}{permalink}.json"
    headers = {"User-Agent": "KeepCalm/1.0 (research dataset)"}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        return []

    comments = []

    def extract(children, depth=0):
        if len(comments) >= max_comments:
            return
        for child in children:
            if child["kind"] != "t1":
                continue
            data = child["data"]
            body = data.get("body", "").strip()
            if 20 < len(body) < 2000 and not body.startswith(">") and not body.startswith("/"):
                comments.append(body)
            if data.get("replies") and isinstance(data["replies"], dict):
                extract(data["replies"]["data"]["children"], depth + 1)
            if len(comments) >= max_comments:
                return

    try:
        listing = resp.json()
        for post_listing in listing:
            if isinstance(post_listing, dict):
                extract(post_listing["data"]["children"])
    except Exception:
        pass

    return comments[:max_comments]


def make_id(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def scrape_subreddits(subreddits: list[str], language: str, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    seen = set()

    for sub in subreddits:
        print(f"\n  r/{sub}...", end=" ", flush=True)
        try:
            posts = fetch_posts(sub, limit=10)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"{len(posts)} posts", end=" ", flush=True)

        for post in posts:
            # Post body
            if post["selftext"] and 30 < len(post["selftext"]) < 2000 and post["selftext"] not in seen:
                seen.add(post["selftext"])
                examples.append({
                    "id": f"rd-{make_id(post['selftext'])}",
                    "text": post["selftext"],
                    "language": language,
                    "domain": f"reddit_{post['subreddit']}",
                    "source": "reddit_post",
                    "context": f"r/{post['subreddit']}: {post['title'][:100]}",
                    "metadata": {"subreddit": post["subreddit"], "type": "post_body"},
                    "annotations": {"communication_risk": None, "tones": [], "intent": None, "explanation": None, "needs_attention": None},
                })

            # Comments
            if post["num_comments"] > 0 and post["permalink"]:
                time.sleep(0.3)
                comments = fetch_comments(post["permalink"], max_comments=25)
                for c in comments:
                    if c not in seen:
                        seen.add(c)
                        examples.append({
                            "id": f"rd-{make_id(c)}",
                            "text": c,
                            "language": language,
                            "domain": f"reddit_{post['subreddit']}",
                            "source": "reddit_comment",
                            "context": f"r/{post['subreddit']}: {post['title'][:100]}",
                            "metadata": {"subreddit": post["subreddit"], "type": "comment"},
                            "annotations": {"communication_risk": None, "tones": [], "intent": None, "explanation": None, "needs_attention": None},
                        })

            if len(examples) % 100 == 0:
                print(f"({len(examples)})", end=" ", flush=True)

    with open(output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return len(examples)


def main() -> None:
    print("Reddit Scraper for Keep Calm")
    print("=" * 60)

    print(f"\n--- English subreddits ---")
    en_count = scrape_subreddits(SUBREDDITS_EN, "en", OUTPUT_EN)

    print(f"\n\n--- Italian subreddits ---")
    it_count = scrape_subreddits(SUBREDDITS_IT, "it", OUTPUT_IT)

    print(f"\n\n{'='*60}")
    print(f"Reddit scrape complete")
    print(f"  EN: {en_count} examples -> {OUTPUT_EN}")
    print(f"  IT: {it_count} examples -> {OUTPUT_IT}")
    print(f"  Total: {en_count + it_count}")


if __name__ == "__main__":
    main()
