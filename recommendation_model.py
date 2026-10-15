

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------- #
# 1. Collaborative Filtering
# --------------------------------------------------------------------------- #
class CollaborativeFiltering:
    """Matrix-factorization based collaborative filtering (implicit + explicit
    ratings blended into a single interaction-strength matrix)."""

    def __init__(self, n_factors: int = 20, random_state: int = 42):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_index: Dict[str, int] = {}
        self.item_index: Dict[str, int] = {}
        self.item_index_inv: Dict[int, str] = {}
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0

    def fit(self, interactions: pd.DataFrame):
        users = interactions["user_id"].unique()
        items = interactions["media_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {m: i for i, m in enumerate(items)}
        self.item_index_inv = {i: m for m, i in self.item_index.items()}

        rows = interactions["user_id"].map(self.user_index)
        cols = interactions["media_id"].map(self.item_index)
        strength = (
            interactions["rating_filled"].fillna(0) * 0.7
            + interactions["watch_time_pct"].fillna(0) * 0.3
        )
        self.global_mean = strength.mean()

        matrix = csr_matrix(
            (strength, (rows, cols)),
            shape=(len(users), len(items)),
        )
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        return self

    def predict_score(self, user_id: str, media_id: str) -> float:
        if user_id not in self.user_index or media_id not in self.item_index:
            return self.global_mean
        u = self.user_factors[self.user_index[user_id]]
        i = self.item_factors[self.item_index[media_id]]
        return float(np.dot(u, i))

    def recommend(self, user_id: str, top_n: int = 10,
                   exclude: List[str] = None) -> List[Tuple[str, float]]:
        exclude = set(exclude or [])
        if user_id not in self.user_index:
            return []
        u = self.user_factors[self.user_index[user_id]]
        scores = self.item_factors @ u
        ranked = sorted(
            ((self.item_index_inv[i], s) for i, s in enumerate(scores)
             if self.item_index_inv[i] not in exclude),
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_n]


# --------------------------------------------------------------------------- #
# 2. Content-Based Filtering
# --------------------------------------------------------------------------- #
class ContentBasedFiltering:
    """Recommends media whose TF-IDF content vector is closest to a user's
    aggregated 'liked content' vector."""

    def __init__(self, media_tfidf_matrix, media_ids: List[str]):
        self.media_tfidf_matrix = media_tfidf_matrix
        self.media_ids = media_ids
        self.media_id_to_row = {m: i for i, m in enumerate(media_ids)}

    def build_user_vector(self, user_id: str, interactions: pd.DataFrame) -> np.ndarray:
        liked = interactions[
            (interactions["user_id"] == user_id) &
            ((interactions["liked"] == True) | (interactions["rating"].fillna(0) >= 4))
        ]
        rows = [self.media_id_to_row[m] for m in liked["media_id"] if m in self.media_id_to_row]
        if not rows:
            # cold start -> neutral (mean) vector
            return np.asarray(self.media_tfidf_matrix.mean(axis=0))
        return np.asarray(self.media_tfidf_matrix[rows].mean(axis=0))

    def recommend(self, user_id: str, interactions: pd.DataFrame, top_n: int = 10,
                   exclude: List[str] = None) -> List[Tuple[str, float]]:
        exclude = set(exclude or [])
        user_vec = self.build_user_vector(user_id, interactions)
        sims = cosine_similarity(user_vec, self.media_tfidf_matrix).flatten()
        ranked = sorted(
            ((self.media_ids[i], s) for i, s in enumerate(sims)
             if self.media_ids[i] not in exclude),
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_n]


# --------------------------------------------------------------------------- #
# 3. Hybrid Recommender
# --------------------------------------------------------------------------- #
class HybridRecommender:
    """Weighted blend of collaborative + content-based scores. The weight
    can be dynamically adjusted based on data sparsity (cold-start users
    lean more on content-based) and on the user's current detected emotion."""

    def __init__(self, cf: CollaborativeFiltering, cbf: ContentBasedFiltering,
                 cf_weight: float = 0.5):
        self.cf = cf
        self.cbf = cbf
        self.cf_weight = cf_weight

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return scores
        vals = np.array(list(scores.values()))
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-9:
            return {k: 0.5 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def recommend(self, user_id: str, interactions: pd.DataFrame, media: pd.DataFrame,
                   top_n: int = 10, exclude: List[str] = None,
                   bias_category: str = None, bias_strength: float = 0.15
                   ) -> List[Tuple[str, float]]:
        exclude = set(exclude or [])
        n_user_inter = len(interactions[interactions["user_id"] == user_id])

        # Cold-start users get more weight on content-based filtering
        cf_weight = self.cf_weight if n_user_inter >= 5 else 0.15
        cbf_weight = 1 - cf_weight

        cf_scores = dict(self.cf.recommend(user_id, top_n=200, exclude=list(exclude)))
        cbf_scores = dict(self.cbf.recommend(user_id, interactions, top_n=200,
                                              exclude=list(exclude)))

        cf_scores = self._normalize(cf_scores)
        cbf_scores = self._normalize(cbf_scores)

        all_media = set(cf_scores) | set(cbf_scores)
        combined = {}
        cat_map = media.set_index("media_id")["category"]
        for m in all_media:
            score = cf_weight * cf_scores.get(m, 0) + cbf_weight * cbf_scores.get(m, 0)
            # Emotion-aware boost: nudge score for media matching current mood category
            if bias_category is not None and cat_map.get(m) == bias_category:
                score += bias_strength
            combined[m] = score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor

    raw = DataCollector().collect_all(n_users=50, n_media=150, n_interactions=1500)
    pre = DataPreprocessor()
    users, media, interactions = pre.clean_dataframes(**raw)
    tfidf_matrix = pre.build_media_text_features(media)
    feats = pre.build_interaction_features(interactions)

    cf = CollaborativeFiltering(n_factors=15).fit(feats)
    cbf = ContentBasedFiltering(tfidf_matrix, pre.media_ids_order)
    hybrid = HybridRecommender(cf, cbf)

    test_user = users.iloc[0]["user_id"]
    recs = hybrid.recommend(test_user, feats, media, top_n=5, bias_category="stress_relief")
    print(f"Top recommendations for {test_user}:")
    for media_id, score in recs:
        title = media.set_index("media_id").loc[media_id, "title"]
        print(f"  {media_id} | {title:45s} | score={score:.3f}")
