

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class RankingEngine:
    def __init__(self, w_hybrid: float = 0.55, w_popularity: float = 0.2,
                 w_engagement: float = 0.25):
        assert abs(w_hybrid + w_popularity + w_engagement - 1.0) < 1e-6
        self.w_hybrid = w_hybrid
        self.w_popularity = w_popularity
        self.w_engagement = w_engagement
        self.engagement_model: LogisticRegression | None = None
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------ #
    def compute_popularity(self, interactions: pd.DataFrame) -> pd.Series:
        """Popularity = normalized blend of interaction count and average
        implicit rating for each media item."""
        grp = interactions.groupby("media_id").agg(
            n=("user_id", "count"),
            avg_rating=("rating_filled", "mean"),
        )
        grp["n_norm"] = (grp["n"] - grp["n"].min()) / (grp["n"].max() - grp["n"].min() + 1e-9)
        grp["rating_norm"] = (grp["avg_rating"] - grp["avg_rating"].min()) / (
            grp["avg_rating"].max() - grp["avg_rating"].min() + 1e-9
        )
        grp["popularity"] = 0.6 * grp["n_norm"] + 0.4 * grp["rating_norm"]
        return grp["popularity"]

    # ------------------------------------------------------------------ #
    def fit_engagement_model(self, interactions: pd.DataFrame, media: pd.DataFrame):
        """Trains a simple logistic regression predicting P(user engages
        i.e. does not skip & watches >50%) from media/user behavioral features."""
        df = interactions.merge(media[["media_id", "duration_min"]], on="media_id", how="left")
        df["target"] = ((df["skipped"] == False) & (df["watch_time_pct"] >= 0.5)).astype(int)

        feature_cols = ["duration_min", "rating_filled"]
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
        X = self.scaler.fit_transform(df[feature_cols])
        y = df["target"]

        self.engagement_model = LogisticRegression(max_iter=500)
        self.engagement_model.fit(X, y)
        return self

    def predict_engagement(self, media_ids: List[str], media: pd.DataFrame,
                            default_rating: float = 0.6) -> Dict[str, float]:
        m = media.set_index("media_id").loc[media_ids, ["duration_min"]].copy()
        m["rating_filled"] = default_rating
        X = self.scaler.transform(m[["duration_min", "rating_filled"]])
        probs = self.engagement_model.predict_proba(X)[:, 1]
        return dict(zip(media_ids, probs))

    # ------------------------------------------------------------------ #
    def rank(self, candidates: List[Tuple[str, float]], interactions: pd.DataFrame,
              media: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        media_ids = [m for m, _ in candidates]
        hybrid_scores = {m: s for m, s in candidates}

        popularity = self.compute_popularity(interactions)
        pop_scores = {m: popularity.get(m, 0.0) for m in media_ids}

        if self.engagement_model is None:
            self.fit_engagement_model(interactions, media)
        eng_scores = self.predict_engagement(media_ids, media)

        rows = []
        for m in media_ids:
            final_score = (
                self.w_hybrid * hybrid_scores[m]
                + self.w_popularity * pop_scores.get(m, 0.0)
                + self.w_engagement * eng_scores.get(m, 0.0)
            )
            rows.append({
                "media_id": m,
                "hybrid_score": hybrid_scores[m],
                "popularity": pop_scores.get(m, 0.0),
                "engagement_probability": eng_scores.get(m, 0.0),
                "final_score": final_score,
            })

        df = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(top_n)
        display_cols = ["media_id", "title", "category", "media_type"]
        if "url" in media.columns:
            display_cols.append("url")
        return df.merge(media[display_cols], on="media_id", how="left").reset_index(drop=True)


if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    from recommendation_model import CollaborativeFiltering, ContentBasedFiltering, HybridRecommender

    raw = DataCollector().collect_all(n_users=50, n_media=150, n_interactions=1500)
    pre = DataPreprocessor()
    users, media, interactions = pre.clean_dataframes(**raw)
    tfidf_matrix = pre.build_media_text_features(media)
    feats = pre.build_interaction_features(interactions)

    cf = CollaborativeFiltering(n_factors=15).fit(feats)
    cbf = ContentBasedFiltering(tfidf_matrix, pre.media_ids_order)
    hybrid = HybridRecommender(cf, cbf)

    test_user = users.iloc[0]["user_id"]
    candidates = hybrid.recommend(test_user, feats, media, top_n=30)

    ranker = RankingEngine()
    top_ranked = ranker.rank(candidates, feats, media, top_n=10)
    print(top_ranked)
