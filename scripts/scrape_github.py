"""Scrape controversial GitHub issues — simplified version.

Focused on high-comment, locked, or heated issues from popular repos.
GitHub API: 60 req/hour unauthenticated.

Output: data/github_en.jsonl
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "github_en.jsonl"

QUERIES = [
    ("is:issue is:locked comments:>30", "locked"),
    ("is:issue is:locked label:controversial", "controversial"),
    ("is:issue label:wontfix comments:>20", "wontfix"),
    ("is:pr is:unmerged comments:>30", "rejected_pr"),
    ("\"this is unacceptable\" is:issue", "unacceptable"),
    ("\"completely useless\" is:issue", "useless"),
    ("\"waste of time\" is:issue", "waste_of_time"),
    ("\"terrible idea\" is:issue", "terrible"),
    ("is:issue label:invalid comments:>20", "invalid"),
]


def make_id(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    seen = set()

    for query, category in QUERIES:
        print(f"Query: {query[:60]}...", end=" ", flush=True)

        try:
            url = "https://api.github.com/search/issues"
            resp = requests.get(url, params={"q": query, "per_page": 10, "sort": "comments", "order": "desc"},
                headers={"Accept": "application/vnd.github.v3+json"}, timeout=15)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                time.sleep(2)
                continue
            data = resp.json()
        except Exception as e:
            print(f"Error: {e}")
            continue

        items = data.get("items", [])
        print(f"{len(items)} issues")

        for issue in items:
            title = issue.get("title", "")
            body = (issue.get("body") or "").strip()
            repo_url = issue.get("repository_url", "")
            repo_name = repo_url.split("repos/")[-1] if "repos/" in repo_url else repo_url
            issue_url = issue.get("html_url", "")

            # Issue body
            if body and 30 < len(body) < 2000 and body not in seen:
                seen.add(body)
                examples.append({
                    "id": f"gh-{make_id(body)}",
                    "text": body,
                    "language": "en",
                    "domain": f"github_{category}",
                    "source": "github_issue",
                    "context": f"Issue: {title} — {repo_name}",
                    "metadata": {"url": issue_url, "category": category},
                    "annotations": {"communication_risk": None, "tones": [], "intent": None, "explanation": None, "needs_attention": None},
                })

            # Comments (separate API call per issue)
            comments_url = issue.get("comments_url", "")
            if comments_url and issue.get("comments", 0) > 5:
                time.sleep(0.5)
                try:
                    cresp = requests.get(f"{comments_url}?per_page=20",
                        headers={"Accept": "application/vnd.github.v3+json"}, timeout=15)
                    if cresp.status_code == 200:
                        for c in cresp.json():
                            cbody = c.get("body", "").strip()
                            if 20 < len(cbody) < 2000 and not cbody.startswith(">") and cbody not in seen:
                                seen.add(cbody)
                                examples.append({
                                    "id": f"gh-{make_id(cbody)}",
                                    "text": cbody,
                                    "language": "en",
                                    "domain": f"github_{category}",
                                    "source": "github_comment",
                                    "context": f"Issue: {title} — {repo_name}",
                                    "metadata": {"url": issue_url, "category": category},
                                    "annotations": {"communication_risk": None, "tones": [], "intent": None, "explanation": None, "needs_attention": None},
                                })
                except Exception:
                    pass

            time.sleep(0.5)

        if len(examples) >= 1000:
            break

    with open(OUTPUT, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nGitHub: {len(examples)} examples -> {OUTPUT}")


if __name__ == "__main__":
    main()
