from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def precision_recall_f1_at_k(recommended: List[str], relevant: set, k: int = 10):
    top_k = recommended[:k]
    if not top_k:
        return 0.0, 0.0, 0.0
    hits = len(set(top_k) & relevant)
    precision = hits / len(top_k)
    recall = hits / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def ndcg_at_k(recommended: List[str], relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    dcg = sum(
        (1.0 if item in relevant else 0.0) / np.log2(idx + 2)
        for idx, item in enumerate(top_k)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


class Evaluator:
    """Evaluates a recommender ('recommend_fn') against a held-out test set
    of interactions, treating rating>=4 or liked==True as 'relevant'."""

    def __init__(self, k: int = 10):
        self.k = k

    def build_relevance_sets(self, test_interactions: pd.DataFrame) -> Dict[str, set]:
        relevant = test_interactions[
            (test_interactions["rating"].fillna(0) >= 4) | (test_interactions["liked"] == True)
        ]
        out: Dict[str, set] = {}
        for uid, grp in relevant.groupby("user_id"):
            out[uid] = set(grp["media_id"])
        return out

    def evaluate_ranking(self, recommend_fn, test_interactions: pd.DataFrame,
                          sample_users: List[str] = None) -> pd.DataFrame:
        relevance = self.build_relevance_sets(test_interactions)
        users = sample_users or list(relevance.keys())
        rows = []
        for uid in users:
            relevant = relevance.get(uid, set())
            if not relevant:
                continue
            recs = [m for m, _ in recommend_fn(uid, top_n=self.k)]
            p, r, f1 = precision_recall_f1_at_k(recs, relevant, self.k)
            ndcg = ndcg_at_k(recs, relevant, self.k)
            rows.append({"user_id": uid, "precision": p, "recall": r, "f1": f1, "ndcg": ndcg})
        return pd.DataFrame(rows)

    def evaluate_user_models(self, recommenders: Dict[str, object], user_id: str,
                             test_interactions: pd.DataFrame, k: int = None) -> pd.DataFrame:
        """Evaluate all recommenders for one selected user only.

        Relevance is taken from that user's held-out test interactions, so the
        values change when the selected user changes.
        """
        k = k or self.k
        relevance = self.build_relevance_sets(test_interactions).get(user_id, set())
        columns = ["model", "user_id", "precision", "recall", "f1", "ndcg"]
        if not relevance:
            return pd.DataFrame(columns=columns)

        rows = []
        for name, recommend_fn in recommenders.items():
            recs = [m for m, _ in recommend_fn(user_id, top_n=k)]
            p, r, f1 = precision_recall_f1_at_k(recs, relevance, k)
            ndcg = ndcg_at_k(recs, relevance, k)
            rows.append({
                "model": name,
                "user_id": user_id,
                "precision": p,
                "recall": r,
                "f1": f1,
                "ndcg": ndcg,
            })
        return pd.DataFrame(rows)

    def evaluate_recommendation_list(self, user_id: str, recommended: List[str],
                                     test_interactions: pd.DataFrame, k: int = None) -> Dict[str, float]:
        """Evaluate the recommendations currently displayed to one user."""
        k = k or self.k
        relevance = self.build_relevance_sets(test_interactions).get(user_id, set())
        if not relevance:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0,
                    "relevant_count": 0, "evaluated": False}

        p, r, f1 = precision_recall_f1_at_k(recommended, relevance, k)
        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "ndcg": ndcg_at_k(recommended, relevance, k),
            "relevant_count": len(relevance),
            "evaluated": True,
        }

    def summarize(self, results: pd.DataFrame) -> Dict[str, float]:
        if results.empty:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0}
        return {
            "precision@k": round(results["precision"].mean(), 4),
            "recall@k": round(results["recall"].mean(), 4),
            "f1@k": round(results["f1"].mean(), 4),
            "ndcg@k": round(results["ndcg"].mean(), 4),
        }


if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    from recommendation_model import CollaborativeFiltering, ContentBasedFiltering, HybridRecommender

    raw = DataCollector().collect_all(n_users=60, n_media=150, n_interactions=2500)
    pre = DataPreprocessor()
    users, media, interactions = pre.clean_dataframes(**raw)
    tfidf_matrix = pre.build_media_text_features(media)
    feats = pre.build_interaction_features(interactions)
    splits = pre.train_val_test_split(feats)

    cf = CollaborativeFiltering(n_factors=15).fit(splits["train"])
    cbf = ContentBasedFiltering(tfidf_matrix, pre.media_ids_order)
    hybrid = HybridRecommender(cf, cbf)

    evaluator = Evaluator(k=10)

    def hybrid_recommend(uid, top_n=10):
        return hybrid.recommend(uid, splits["train"], media, top_n=top_n)

    def cf_recommend(uid, top_n=10):
        return cf.recommend(uid, top_n=top_n)

    def cbf_recommend(uid, top_n=10):
        return cbf.recommend(uid, splits["train"], top_n=top_n)

    test_users = splits["test"]["user_id"].unique().tolist()[:30]

    for name, fn in [("Collaborative Filtering", cf_recommend),
                      ("Content-Based Filtering", cbf_recommend),
                      ("Hybrid", hybrid_recommend)]:
        res = evaluator.evaluate_ranking(fn, splits["test"], sample_users=test_users)
        summary = evaluator.summarize(res)
        print(f"{name:26s} -> {summary}")