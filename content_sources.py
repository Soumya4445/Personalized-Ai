

from __future__ import annotations

import time
import urllib.parse
from typing import Dict, List, Optional

import pandas as pd
import requests

CATEGORIES = [
    "confidence_building", "stress_relief", "career_motivation",
    "physical_fitness", "academic_motivation", "mindfulness",
    "leadership", "relationships",
]


def _yt_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)


def _spotify_search_url(query: str, kind: str = "episode") -> str:
    return f"https://open.spotify.com/search/{urllib.parse.quote(query)}/{kind}s"


# --------------------------------------------------------------------------- #
# 1. Curated, real, no-key-required content
# --------------------------------------------------------------------------- #
class CuratedContentProvider:
   

    VIDEOS: Dict[str, List[str]] = {
        "confidence_building": ["Mel Robbins confidence", "Les Brown you have greatness in you"],
        "stress_relief": ["Headspace guided meditation stress relief", "how to calm anxiety Andrew Huberman"],
        "career_motivation": ["Gary Vaynerchuk career advice motivation", "Simon Sinek find your why"],
        "physical_fitness": ["David Goggins motivation running", "fitness motivation workout speech"],
        "academic_motivation": ["study motivation speech students", "how to stay focused studying Ali Abdaal"],
        "mindfulness": ["Jon Kabat-Zinn mindfulness talk", "5 minute mindfulness meditation"],
        "leadership": ["Simon Sinek leadership talk", "leadership lessons TED talk"],
        "relationships": ["Esther Perel relationships talk", "building healthy relationships TED"],
    }

    PODCAST_SHOWS: Dict[str, List[str]] = {
        "confidence_building": ["The Confidence Podcast", "The Self Esteem podcast"],
        "stress_relief": ["Ten Percent Happier", "The Anxiety Coaches Podcast"],
        "career_motivation": ["The Tim Ferriss Show", "How I Built This"],
        "physical_fitness": ["Huberman Lab", "Mark Bell's Power Project"],
        "academic_motivation": ["The Study Skills Podcast", "Deep Questions with Cal Newport"],
        "mindfulness": ["The Mindful Kind", "Untangle"],
        "leadership": ["Dare to Lead with Brene Brown", "WorkLife with Adam Grant"],
        "relationships": ["Where Should We Begin with Esther Perel", "The Relationship School Podcast"],
    }

    QUOTES: Dict[str, List[Dict[str, str]]] = {
        "confidence_building": [
            {"quote": "Believe you can and you're halfway there.", "author": "Theodore Roosevelt"},
            {"quote": "You gain strength, courage, and confidence by every experience.", "author": "Eleanor Roosevelt"},
        ],
        "stress_relief": [
            {"quote": "Peace comes from within. Do not seek it without.", "author": "Buddha"},
            {"quote": "You don't have to control your thoughts. You just have to stop letting them control you.", "author": "Dan Millman"},
        ],
        "career_motivation": [
            {"quote": "The only way to do great work is to love what you do.", "author": "Steve Jobs"},
            {"quote": "Opportunities don't happen. You create them.", "author": "Chris Grosser"},
        ],
        "physical_fitness": [
            {"quote": "Take care of your body. It's the only place you have to live.", "author": "Jim Rohn"},
            {"quote": "The pain you feel today will be the strength you feel tomorrow.", "author": "Anonymous"},
        ],
        "academic_motivation": [
            {"quote": "Education is the most powerful weapon which you can use to change the world.", "author": "Nelson Mandela"},
            {"quote": "The beautiful thing about learning is that no one can take it away from you.", "author": "B.B. King"},
        ],
        "mindfulness": [
            {"quote": "Wherever you are, be all there.", "author": "Jim Elliot"},
            {"quote": "The present moment is the only time over which we have dominion.", "author": "Thich Nhat Hanh"},
        ],
        "leadership": [
            {"quote": "A leader is one who knows the way, goes the way, and shows the way.", "author": "John C. Maxwell"},
            {"quote": "Leadership is not about being in charge. It's about taking care of those in your charge.", "author": "Simon Sinek"},
        ],
        "relationships": [
            {"quote": "The best thing to hold onto in life is each other.", "author": "Audrey Hepburn"},
            {"quote": "Love is composed of a single soul inhabiting two bodies.", "author": "Aristotle"},
        ],
    }

    ARTICLES: Dict[str, List[str]] = {
        "confidence_building": ["how to build self confidence psychology today"],
        "stress_relief": ["evidence based stress management techniques"],
        "career_motivation": ["career growth tips harvard business review"],
        "physical_fitness": ["benefits of exercise on mental health"],
        "academic_motivation": ["study motivation tips for students"],
        "mindfulness": ["beginners guide to mindfulness"],
        "leadership": ["leadership skills harvard business review"],
        "relationships": ["how to build healthy relationships psychology today"],
    }

    def _google_search_url(self, query: str) -> str:
        return "https://www.google.com/search?q=" + urllib.parse.quote(query)

    def build_media_catalog(self) -> pd.DataFrame:
        rows = []
        idx = 0
        for category in CATEGORIES:
            for q in self.VIDEOS.get(category, []):
                rows.append({
                    "media_id": f"V{idx:04d}", "title": q.title(),
                    "media_type": "video", "category": category,
                    "tags": [category], "description": f"Real YouTube results for: {q}",
                    "duration_min": 12.0, "url": _yt_search_url(q),
                })
                idx += 1
            for show in self.PODCAST_SHOWS.get(category, []):
                rows.append({
                    "media_id": f"P{idx:04d}", "title": show,
                    "media_type": "podcast", "category": category,
                    "tags": [category], "description": f"Real podcast show: {show} (Spotify)",
                    "duration_min": 35.0, "url": _spotify_search_url(show, "show"),
                })
                idx += 1
            for q in self.QUOTES.get(category, []):
                rows.append({
                    "media_id": f"Q{idx:04d}", "title": f'"{q["quote"]}" — {q["author"]}',
                    "media_type": "quote", "category": category,
                    "tags": [category], "description": q["author"],
                    "duration_min": 0.5, "url": "",
                })
                idx += 1
            for q in self.ARTICLES.get(category, []):
                rows.append({
                    "media_id": f"A{idx:04d}", "title": q.title(),
                    "media_type": "article", "category": category,
                    "tags": [category], "description": f"Real article results for: {q}",
                    "duration_min": 6.0, "url": self._google_search_url(q),
                })
                idx += 1
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2a. YouTube Data API v3 (needs an API key - free tier available)
#     Get one at: https://console.cloud.google.com/apis/credentials
#     (enable "YouTube Data API v3" on the project first)
# --------------------------------------------------------------------------- #
class YouTubeLiveSource:
    BASE_URL = "https://www.googleapis.com/youtube/v3/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search_videos(self, query: str, max_results: int = 5) -> List[Dict]:
        params = {
            "part": "snippet", "q": query, "type": "video",
            "maxResults": max_results, "key": self.api_key,
            "relevanceLanguage": "en", "safeSearch": "strict",
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("items", []):
            vid = item["id"]["videoId"]
            snippet = item["snippet"]
            results.append({
                "title": snippet["title"],
                "description": snippet.get("description", "")[:200],
                "url": f"https://www.youtube.com/watch?v={vid}",
                "channel": snippet.get("channelTitle", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            })
        return results


# --------------------------------------------------------------------------- #
# 2b. Spotify Web API (Client Credentials flow - needs Client ID + Secret)
#     Get credentials at: https://developer.spotify.com/dashboard
# --------------------------------------------------------------------------- #
class SpotifyLiveSource:
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    SEARCH_URL = "https://api.spotify.com/v1/search"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._token_expiry = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry:
            return self._token
        resp = requests.post(
            self.TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expiry = time.time() + payload.get("expires_in", 3600) - 30
        return self._token

    def search_shows(self, query: str, max_results: int = 5) -> List[Dict]:
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": query, "type": "show", "limit": max_results}
        resp = requests.get(self.SEARCH_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("shows", {}).get("items", []):
            results.append({
                "title": item["name"],
                "description": (item.get("description") or "")[:200],
                "url": item["external_urls"]["spotify"],
                "publisher": item.get("publisher", ""),
                "thumbnail": (item.get("images") or [{}])[0].get("url", ""),
            })
        return results


# --------------------------------------------------------------------------- #
# 2c. Free quotes API - NO KEY REQUIRED. Quotable primary, ZenQuotes fallback.
# --------------------------------------------------------------------------- #
class QuoteLiveSource:
    QUOTABLE_URL = "https://api.quotable.io/quotes/random"
    ZENQUOTES_URL = "https://zenquotes.io/api/random"

    CATEGORY_TAGS = {
        "confidence_building": "inspirational",
        "stress_relief": "wisdom",
        "career_motivation": "success",
        "physical_fitness": "motivational",
        "academic_motivation": "wisdom",
        "mindfulness": "wisdom",
        "leadership": "leadership",
        "relationships": "love",
    }

    def get_quote(self, category: str = None) -> Optional[Dict[str, str]]:
        tag = self.CATEGORY_TAGS.get(category, "inspirational")
        try:
            resp = requests.get(self.QUOTABLE_URL, params={"tags": tag, "limit": 1}, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            if data:
                q = data[0]
                return {"quote": q["content"], "author": q["author"]}
        except Exception:
            pass
        # Fallback: ZenQuotes (no tag filtering on free tier, but no key needed)
        try:
            resp = requests.get(self.ZENQUOTES_URL, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            if data:
                q = data[0]
                return {"quote": q["q"], "author": q["a"]}
        except Exception:
            pass
        return None


# --------------------------------------------------------------------------- #
# Unified fetcher: builds a live media catalog from whichever APIs are
# configured, falling back to the curated dataset for anything missing.
# --------------------------------------------------------------------------- #
class LiveContentFetcher:
    def __init__(self, youtube_api_key: str = "", spotify_client_id: str = "",
                 spotify_client_secret: str = ""):
        self.youtube = YouTubeLiveSource(youtube_api_key) if youtube_api_key else None
        self.spotify = (
            SpotifyLiveSource(spotify_client_id, spotify_client_secret)
            if spotify_client_id and spotify_client_secret else None
        )
        self.quotes = QuoteLiveSource()
        self.curated = CuratedContentProvider()

    def fetch_for_category(self, category: str, per_type: int = 2) -> pd.DataFrame:
        rows = []
        idx = 0

        # Videos
        if self.youtube:
            query = category.replace("_", " ") + " motivation"
            try:
                for v in self.youtube.search_videos(query, max_results=per_type):
                    rows.append({
                        "media_id": f"LV{category[:3]}{idx}", "title": v["title"],
                        "media_type": "video", "category": category, "tags": [category],
                        "description": v["description"], "duration_min": 10.0, "url": v["url"],
                    })
                    idx += 1
            except Exception:
                pass
        if not any(r["media_type"] == "video" for r in rows):
            for q in self.curated.VIDEOS.get(category, [])[:per_type]:
                rows.append({
                    "media_id": f"CV{category[:3]}{idx}", "title": q.title(),
                    "media_type": "video", "category": category, "tags": [category],
                    "description": f"Real YouTube results for: {q}", "duration_min": 12.0,
                    "url": _yt_search_url(q),
                })
                idx += 1

        # Podcasts
        if self.spotify:
            query = category.replace("_", " ")
            try:
                for s in self.spotify.search_shows(query, max_results=per_type):
                    rows.append({
                        "media_id": f"LP{category[:3]}{idx}", "title": s["title"],
                        "media_type": "podcast", "category": category, "tags": [category],
                        "description": s["description"], "duration_min": 35.0, "url": s["url"],
                    })
                    idx += 1
            except Exception:
                pass
        if not any(r["media_type"] == "podcast" for r in rows):
            for show in self.curated.PODCAST_SHOWS.get(category, [])[:per_type]:
                rows.append({
                    "media_id": f"CP{category[:3]}{idx}", "title": show,
                    "media_type": "podcast", "category": category, "tags": [category],
                    "description": f"Real podcast show: {show} (Spotify)", "duration_min": 35.0,
                    "url": _spotify_search_url(show, "show"),
                })
                idx += 1

        # Quotes (always live - no key needed)
        for _ in range(per_type):
            q = self.quotes.get_quote(category)
            if q:
                rows.append({
                    "media_id": f"LQ{category[:3]}{idx}", "title": f'"{q["quote"]}" — {q["author"]}',
                    "media_type": "quote", "category": category, "tags": [category],
                    "description": q["author"], "duration_min": 0.5, "url": "",
                })
                idx += 1

        return pd.DataFrame(rows)

    def fetch_full_catalog(self, per_type: int = 2) -> pd.DataFrame:
        frames = [self.fetch_for_category(c, per_type) for c in CATEGORIES]
        return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("=== Curated (offline, real links) ===")
    curated = CuratedContentProvider().build_media_catalog()
    print(curated.shape)
    print(curated.head(8)[["title", "media_type", "category", "url"]])

    print("\n=== Live quotes (no API key needed) ===")
    qsrc = QuoteLiveSource()
    for cat in ["stress_relief", "career_motivation"]:
        print(cat, "->", qsrc.get_quote(cat))
