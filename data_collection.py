

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class UserProfile:
    user_id: str
    age: int
    gender: str
    profession: str
    interests: List[str]


@dataclass
class MediaItem:
    media_id: str
    title: str
    media_type: str            # video, podcast, speech, quote, article
    category: str              # confidence, stress_relief, career, fitness, academic ...
    tags: List[str]
    description: str
    duration_min: float


@dataclass
class Interaction:
    user_id: str
    media_id: str
    timestamp: datetime
    rating: Optional[float]        # 1-5, None if not rated
    liked: bool
    watch_time_pct: float          # 0-1, fraction of media watched
    skipped: bool
    comment: Optional[str] = None  # free-text feedback used for sentiment analysis
    mood_input: Optional[str] = None  # free-text mood the user typed in


# --------------------------------------------------------------------------- #
# Data Collector
# --------------------------------------------------------------------------- #

class DataCollector:
    """Collects / synthesizes user, media and interaction data."""

    CATEGORIES = [
        "confidence_building", "stress_relief", "career_motivation",
        "physical_fitness", "academic_motivation", "mindfulness",
        "leadership", "relationships",
    ]
    MEDIA_TYPES = ["video", "podcast", "speech", "quote", "article"]
    PROFESSIONS = ["student", "engineer", "entrepreneur", "athlete",
                   "teacher", "healthcare_worker", "artist", "unemployed"]
    GENDERS = ["male", "female", "non_binary"]

    MOOD_PHRASES_POS = [
        "I feel great and ready to take on the day",
        "Feeling motivated and confident today",
        "I'm excited about my new project",
    ]
    MOOD_PHRASES_NEG = [
        "I feel stressed and overwhelmed with work",
        "I'm anxious about my upcoming exam",
        "Feeling low on energy and unmotivated",
        "I am sad and need something uplifting",
    ]
    MOOD_PHRASES_NEU = [
        "Just checking in, nothing special today",
        "Looking for something interesting to watch",
    ]

    COMMENTS_POS = [
        "This video really boosted my confidence, loved it!",
        "Exactly what I needed, very inspiring speech.",
        "Great content, helped me stay focused.",
    ]
    COMMENTS_NEG = [
        "This did not help me at all, too generic.",
        "I found this boring and repetitive.",
        "Not relevant to what I was feeling.",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # ---------------------------- Users ----------------------------- #
    def generate_users(self, n: int = 200) -> pd.DataFrame:
        users = []
        for i in range(n):
            interests = self.rng.sample(self.CATEGORIES, k=self.rng.randint(2, 4))
            users.append(UserProfile(
                user_id=f"U{i:04d}",
                age=self.rng.randint(16, 55),
                gender=self.rng.choice(self.GENDERS),
                profession=self.rng.choice(self.PROFESSIONS),
                interests=interests,
            ))
        return pd.DataFrame([u.__dict__ for u in users])

    # ---------------------------- Media ------------------------------ #
    def generate_media(self, n: int = 500) -> pd.DataFrame:
        items = []
        for i in range(n):
            category = self.rng.choice(self.CATEGORIES)
            media_type = self.rng.choice(self.MEDIA_TYPES)
            tags = self.rng.sample(self.CATEGORIES, k=self.rng.randint(1, 3))
            items.append(MediaItem(
                media_id=f"M{i:04d}",
                title=f"{category.replace('_', ' ').title()} {media_type.title()} #{i}",
                media_type=media_type,
                category=category,
                tags=tags,
                description=(
                    f"A {media_type} about {category.replace('_', ' ')} "
                    f"designed to help you with {', '.join(tags).replace('_', ' ')}."
                ),
                duration_min=round(self.rng.uniform(1, 45), 1),
            ))
        return pd.DataFrame([m.__dict__ for m in items])

    # ------------------------ Interactions ---------------------------- #
    def generate_interactions(self, users: pd.DataFrame, media: pd.DataFrame,
                               n: int = 4000) -> pd.DataFrame:
        interactions = []
        start = datetime.now() - timedelta(days=90)
        for _ in range(n):
            user = users.sample(1, random_state=self.rng.randint(0, 10_000)).iloc[0]
            # Bias media choice toward user's interests (simulates real behaviour)
            if self.rng.random() < 0.7:
                candidates = media[media["category"].isin(user["interests"])]
                if candidates.empty:
                    candidates = media
            else:
                candidates = media
            m = candidates.sample(1, random_state=self.rng.randint(0, 10_000)).iloc[0]

            liked = self.rng.random() < 0.55
            skipped = self.rng.random() < 0.15
            watch_time_pct = 0.0 if skipped else round(self.rng.uniform(0.2, 1.0), 2)
            rating = self.rng.choices(
                [None, 1, 2, 3, 4, 5], weights=[0.4, 0.03, 0.05, 0.12, 0.2, 0.2], k=1
            )[0]

            comment = None
            if self.rng.random() < 0.25:
                comment = self.rng.choice(self.COMMENTS_POS if liked else self.COMMENTS_NEG)

            mood_input = None
            if self.rng.random() < 0.3:
                mood_input = self.rng.choice(
                    self.MOOD_PHRASES_POS + self.MOOD_PHRASES_NEG + self.MOOD_PHRASES_NEU
                )

            interactions.append(Interaction(
                user_id=user["user_id"],
                media_id=m["media_id"],
                timestamp=start + timedelta(minutes=self.rng.randint(0, 90 * 24 * 60)),
                rating=rating,
                liked=liked,
                watch_time_pct=watch_time_pct,
                skipped=skipped,
                comment=comment,
                mood_input=mood_input,
            ))
        df = pd.DataFrame([i.__dict__ for i in interactions])
        return df

    def collect_all(self, n_users=200, n_media=500, n_interactions=4000
                     ) -> Dict[str, pd.DataFrame]:
        """Convenience entry point returning the full raw dataset."""
        users = self.generate_users(n_users)
        media = self.generate_media(n_media)
        interactions = self.generate_interactions(users, media, n_interactions)
        return {"users": users, "media": media, "interactions": interactions}


if __name__ == "__main__":
    collector = DataCollector()
    data = collector.collect_all()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
        print(df.head(3), "\n")
