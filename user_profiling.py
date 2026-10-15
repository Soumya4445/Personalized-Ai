
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sentiment_emotion import SentimentEmotionAnalyzer


class UserProfiler:
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.analyzer = SentimentEmotionAnalyzer()

    def build_profiles(self, users: pd.DataFrame, media: pd.DataFrame,
                        interactions: pd.DataFrame) -> pd.DataFrame:
        media_cat = media.set_index("media_id")["category"]
        profiles = []

        for _, user in users.iterrows():
            uid = user["user_id"]
            u_inter = interactions[interactions["user_id"] == uid]

            # --- behavioral category affinity vector -------------------
            cat_scores = {c: 0.0 for c in self.categories}
            for _, row in u_inter.iterrows():
                cat = media_cat.get(row["media_id"])
                if cat is None:
                    continue
                weight = (
                    1.0
                    + (row["liked_flag"] if "liked_flag" in row else int(row.get("liked", False)))
                    + row.get("watch_time_pct", 0)
                    - (0.5 * row.get("skipped_flag", 0))
                )
                cat_scores[cat] = cat_scores.get(cat, 0.0) + max(weight, 0)

            # blend with declared interests (cold-start friendly)
            for interest in user["interests"]:
                cat_scores[interest] = cat_scores.get(interest, 0.0) + 1.5

            total = sum(cat_scores.values()) or 1.0
            cat_scores = {k: v / total for k, v in cat_scores.items()}

            # --- most recent mood / emotion -----------------------------
            recent_mood_text = ""
            mood_rows = u_inter[u_inter["mood_input"].astype(bool)]
            if not mood_rows.empty:
                recent_mood_text = mood_rows.sort_values("timestamp").iloc[-1]["mood_input"]
            sentiment_result = self.analyzer.analyze(recent_mood_text)

            avg_rating = u_inter["rating"].dropna().mean() if not u_inter.empty else np.nan
            engagement = u_inter["watch_time_pct"].mean() if not u_inter.empty else 0.0

            profile = {
                "user_id": uid,
                "age": user["age"],
                "profession": user["profession"],
                "top_category": max(cat_scores, key=cat_scores.get),
                "category_affinity": cat_scores,
                "current_sentiment": sentiment_result.sentiment,
                "current_emotion": sentiment_result.emotion,
                "recommended_bias_category": self.analyzer.category_for_emotion(
                    sentiment_result.emotion
                ),
                "avg_rating": avg_rating,
                "engagement_score": engagement,
                "n_interactions": len(u_inter),
            }
            profiles.append(profile)

        return pd.DataFrame(profiles)


if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor

    raw = DataCollector().collect_all(n_users=20, n_media=100, n_interactions=500)
    pre = DataPreprocessor()
    users, media, interactions = pre.clean_dataframes(**raw)
    interactions = pre.build_interaction_features(interactions)

    profiler = UserProfiler(categories=DataCollector.CATEGORIES)
    profiles = profiler.build_profiles(users, media, interactions)
    print(profiles[["user_id", "top_category", "current_emotion",
                     "recommended_bias_category", "engagement_score"]].head(10))
